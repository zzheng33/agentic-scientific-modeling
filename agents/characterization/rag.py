"""Hierarchical, auditable retrieval over scientific papers."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]")
_SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def _character_ngrams(text: str, size: int = 3) -> Iterable[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    for start in range(max(0, len(compact) - size + 1)):
        yield compact[start : start + size]


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    title: str
    path: str
    page: int | None
    text: str


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    parent_id: str
    title: str
    path: str
    page: int | None
    text: str
    parent_start: int
    parent_end: int

    def citation(self) -> str:
        location = f", page {self.page}" if self.page is not None else ""
        return f"{self.chunk_id}: {self.title} ({self.path}{location})"


@dataclass(frozen=True)
class SearchResult:
    chunk: PaperChunk
    score: float
    bm25_score: float
    semantic_score: float
    rerank_score: float

    # Preserve tuple unpacking used by the first RAG implementation.
    def __iter__(self):
        yield self.chunk
        yield self.score


class InMemoryTestRetriever:
    """Dependency-free reference retriever used only by unit tests.

    Production characterization uses PersistentPaperRetriever from rag_store.py.
    """

    def __init__(
        self,
        parents: Iterable[ParentChunk],
        chunks: Iterable[PaperChunk],
        *,
        dense_dimensions: int = 512,
        bm25_weight: float = 0.55,
        semantic_weight: float = 0.45,
        candidate_multiplier: int = 6,
        diversity_lambda: float = 0.82,
    ) -> None:
        if dense_dimensions < 64:
            raise ValueError("RAG dense_dimensions must be at least 64")
        if bm25_weight < 0 or semantic_weight < 0 or bm25_weight + semantic_weight <= 0:
            raise ValueError("RAG hybrid weights must be non-negative and not both zero")
        if candidate_multiplier < 1:
            raise ValueError("RAG candidate_multiplier must be positive")
        if not 0 <= diversity_lambda <= 1:
            raise ValueError("RAG diversity_lambda must be between 0 and 1")
        self.parents = {parent.parent_id: parent for parent in parents}
        self.chunks = list(chunks)
        self.dense_dimensions = dense_dimensions
        total_weight = bm25_weight + semantic_weight
        self.bm25_weight = bm25_weight / total_weight
        self.semantic_weight = semantic_weight / total_weight
        self.candidate_multiplier = candidate_multiplier
        self.diversity_lambda = diversity_lambda
        self._term_frequencies = [Counter(_tokens(chunk.text)) for chunk in self.chunks]
        self._token_sets = [set(frequencies) for frequencies in self._term_frequencies]
        self._lengths = [sum(frequencies.values()) for frequencies in self._term_frequencies]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            self._document_frequency.update(frequencies.keys())
        self._dense_vectors = [self._dense_vector(chunk.text) for chunk in self.chunks]

    @classmethod
    def from_directory(
        cls,
        corpus_root: str | Path,
        *,
        parent_chunk_chars: int = 6000,
        child_chunk_chars: int = 1200,
        child_overlap_chars: int = 180,
        dense_dimensions: int = 512,
        bm25_weight: float = 0.55,
        semantic_weight: float = 0.45,
        candidate_multiplier: int = 6,
        diversity_lambda: float = 0.82,
        # Backward-compatible alias from the initial implementation.
        chunk_chars: int | None = None,
    ) -> "InMemoryTestRetriever":
        if chunk_chars is not None:
            child_chunk_chars = chunk_chars
        root = Path(corpus_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"RAG paper corpus does not exist: {root}")
        if parent_chunk_chars < child_chunk_chars:
            raise ValueError("RAG parent chunks must be at least as large as child chunks")
        if child_chunk_chars < 400:
            raise ValueError("RAG child_chunk_chars must be at least 400")
        if child_overlap_chars < 0 or child_overlap_chars >= child_chunk_chars:
            raise ValueError("RAG child overlap must be smaller than the child chunk")

        parents, children = chunk_corpus(
            root,
            parent_chunk_chars=parent_chunk_chars,
            child_chunk_chars=child_chunk_chars,
            child_overlap_chars=child_overlap_chars,
        )
        return cls(
            parents,
            children,
            dense_dimensions=dense_dimensions,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            candidate_multiplier=candidate_multiplier,
            diversity_lambda=diversity_lambda,
        )


    def _dense_vector(self, text: str) -> dict[int, float]:
        vector: defaultdict[int, float] = defaultdict(float)
        for ngram in _character_ngrams(text):
            digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
            encoded = int.from_bytes(digest, "big")
            index = encoded % self.dense_dimensions
            sign = -1.0 if encoded & 1 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            return {index: value / norm for index, value in vector.items()}
        return {}

    @staticmethod
    def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())

    def _bm25_scores(self, query_terms: Counter[str]) -> list[float]:
        count = len(self.chunks)
        k1, b = 1.5, 0.75
        scores: list[float] = []
        for frequencies, length in zip(self._term_frequencies, self._lengths):
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + k1 * (
                    1.0 - b + b * length / max(self._average_length, 1.0)
                )
                score += (
                    query_frequency * inverse_frequency * frequency * (k1 + 1.0)
                    / normalization
                )
            scores.append(score)
        return scores

    @staticmethod
    def _reciprocal_ranks(scores: list[float], limit: int, offset: int = 60) -> dict[int, float]:
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return {
            index: 1.0 / (offset + rank)
            for rank, index in enumerate(ranked[:limit], start=1)
            if scores[index] > 0
        }

    def search(self, query: str, *, top_k: int = 6) -> list[SearchResult]:
        if top_k < 1 or not self.chunks:
            return []
        query_terms = Counter(_tokens(query))
        query_vector = self._dense_vector(query)
        if not query_terms and not query_vector:
            return []
        candidate_limit = min(len(self.chunks), max(top_k, top_k * self.candidate_multiplier))
        bm25_scores = self._bm25_scores(query_terms)
        semantic_scores = [
            max(0.0, self._cosine(query_vector, vector))
            for vector in self._dense_vectors
        ]
        bm25_ranks = self._reciprocal_ranks(bm25_scores, candidate_limit)
        semantic_ranks = self._reciprocal_ranks(semantic_scores, candidate_limit)
        candidates = set(bm25_ranks) | set(semantic_ranks)
        hybrid = {
            index: self.bm25_weight * bm25_ranks.get(index, 0.0)
            + self.semantic_weight * semantic_ranks.get(index, 0.0)
            for index in candidates
        }

        parent_support: defaultdict[str, float] = defaultdict(float)
        for index, score in hybrid.items():
            parent_support[self.chunks[index].parent_id] += score
        max_parent_support = max(parent_support.values(), default=1.0)
        query_set = set(query_terms)
        query_bigrams = set(zip(_tokens(query), _tokens(query)[1:]))
        relevance: dict[int, float] = {}
        for index in candidates:
            token_set = self._token_sets[index]
            coverage = len(query_set & token_set) / max(1, len(query_set))
            child_tokens = _tokens(self.chunks[index].text)
            child_bigrams = set(zip(child_tokens, child_tokens[1:]))
            phrase_coverage = len(query_bigrams & child_bigrams) / max(1, len(query_bigrams))
            support = parent_support[self.chunks[index].parent_id] / max_parent_support
            relevance[index] = hybrid[index] + 0.006 * coverage + 0.003 * phrase_coverage + 0.002 * support

        selected: list[int] = []
        remaining = set(candidates)
        while remaining and len(selected) < top_k:
            def mmr(index: int) -> tuple[float, str]:
                redundancy = max(
                    (
                        self._cosine(self._dense_vectors[index], self._dense_vectors[chosen])
                        for chosen in selected
                    ),
                    default=0.0,
                )
                if any(
                    self.chunks[index].parent_id == self.chunks[chosen].parent_id
                    for chosen in selected
                ):
                    redundancy = max(redundancy, 0.9)
                value = (
                    self.diversity_lambda * relevance[index]
                    - (1.0 - self.diversity_lambda) * redundancy * 0.02
                )
                return value, self.chunks[index].chunk_id

            best = max(remaining, key=mmr)
            selected.append(best)
            remaining.remove(best)

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=relevance[index],
                bm25_score=bm25_scores[index],
                semantic_score=semantic_scores[index],
                rerank_score=relevance[index],
            )
            for index in selected
        ]

    def render_context(
        self,
        query: str,
        *,
        top_k: int = 6,
        max_chars: int = 12000,
        parent_context_chars: int = 2600,
    ) -> str:
        blocks: list[str] = []
        used = 0
        for result in self.search(query, top_k=top_k):
            chunk = result.chunk
            parent = self.parents[chunk.parent_id]
            before, after = _parent_surroundings(
                parent.text, chunk, parent_context_chars
            )
            block = (
                f"[paper_source {chunk.citation()}, parent_id={chunk.parent_id}, "
                f"hybrid_score={result.score:.6f}, bm25={result.bm25_score:.4f}, "
                f"semantic={result.semantic_score:.4f}]\n"
                f"<matched_child>\n{chunk.text.strip()}\n</matched_child>\n"
                f"<parent_context_before>\n{before}\n</parent_context_before>\n"
                f"<parent_context_after>\n{after}\n</parent_context_after>"
            )
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rstrip()
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)


def chunk_corpus(
    corpus_root: str | Path,
    *,
    parent_chunk_chars: int = 6000,
    child_chunk_chars: int = 1200,
    child_overlap_chars: int = 180,
) -> tuple[list[ParentChunk], list[PaperChunk]]:
    """Parse a corpus into deterministic parent and child chunks."""
    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"RAG paper corpus does not exist: {root}")
    if parent_chunk_chars < child_chunk_chars:
        raise ValueError("RAG parent chunks must be at least as large as child chunks")
    if child_chunk_chars < 400:
        raise ValueError("RAG child_chunk_chars must be at least 400")
    if child_overlap_chars < 0 or child_overlap_chars >= child_chunk_chars:
        raise ValueError("RAG child overlap must be smaller than the child chunk")
    parents: list[ParentChunk] = []
    children: list[PaperChunk] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.suffix.lower() not in _SUPPORTED_SUFFIXES
        ):
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            sources = _pdf_sources(path, relative)
        else:
            sources = [
                (
                    path.stem,
                    relative,
                    None,
                    path.read_text(encoding="utf-8", errors="replace"),
                )
            ]
        for title, source_path, page, text in sources:
            for parent in _parent_chunks(
                text,
                title=title,
                relative_path=source_path,
                page=page,
                parent_chunk_chars=parent_chunk_chars,
            ):
                parents.append(parent)
                children.extend(
                    _child_chunks(
                        parent,
                        child_chunk_chars=child_chunk_chars,
                        overlap_chars=child_overlap_chars,
                    )
                )
    return parents, children


def _pdf_sources(path: Path, relative_path: str) -> list[tuple[str, str, int | None, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF RAG requires pypdf; install setup/requirements.txt") from exc
    reader = PdfReader(path)
    metadata_title = str((reader.metadata or {}).get("/Title") or "").strip()
    title = metadata_title or path.stem
    return [
        (title, relative_path, page_number, page.extract_text() or "")
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def _parent_chunks(
    text: str,
    *,
    title: str,
    relative_path: str,
    page: int | None,
    parent_chunk_chars: int,
) -> list[ParentChunk]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    parents: list[ParentChunk] = []
    start = 0
    ordinal = 1
    while start < len(normalized):
        stop = min(start + parent_chunk_chars, len(normalized))
        if stop < len(normalized):
            boundary = normalized.rfind("\n\n", start, stop)
            if boundary > start + parent_chunk_chars // 2:
                stop = boundary
        content = normalized[start:stop].strip()
        page_label = f"p{page:04d}" if page is not None else "text"
        parent_id = f"paper:{relative_path}:{page_label}:parent{ordinal:03d}"
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                title=title,
                path=relative_path,
                page=page,
                text=content,
            )
        )
        ordinal += 1
        start = stop
        while start < len(normalized) and normalized[start].isspace():
            start += 1
    return parents


def _child_chunks(
    parent: ParentChunk,
    *,
    child_chunk_chars: int,
    overlap_chars: int,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    start = 0
    ordinal = 1
    while start < len(parent.text):
        stop = min(start + child_chunk_chars, len(parent.text))
        if stop < len(parent.text):
            boundary = max(
                parent.text.rfind("\n\n", start, stop),
                parent.text.rfind(". ", start, stop),
            )
            if boundary > start + child_chunk_chars // 2:
                stop = boundary + (2 if parent.text[boundary : boundary + 2] == ". " else 0)
        content = parent.text[start:stop].strip()
        if content:
            chunks.append(
                PaperChunk(
                    chunk_id=f"{parent.parent_id}:child{ordinal:03d}",
                    parent_id=parent.parent_id,
                    title=parent.title,
                    path=parent.path,
                    page=parent.page,
                    text=content,
                    parent_start=start,
                    parent_end=stop,
                )
            )
            ordinal += 1
        if stop >= len(parent.text):
            break
        start = max(stop - overlap_chars, start + 1)
    return chunks


def _parent_surroundings(
    parent_text: str,
    child: PaperChunk,
    limit: int,
) -> tuple[str, str]:
    half = max(0, limit // 2)
    before_start = max(0, child.parent_start - half)
    after_stop = min(len(parent_text), child.parent_end + half)
    before = parent_text[before_start : child.parent_start].strip()
    after = parent_text[child.parent_end : after_stop].strip()
    if before_start:
        before = "..." + before
    if after_stop < len(parent_text):
        after += "..."
    return before, after
