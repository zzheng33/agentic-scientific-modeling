"""Persistent BGE-M3/HNSW hybrid retrieval for papers and runbooks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from .rag import (
    PaperChunk,
    ParentChunk,
    SearchResult,
    _parent_surroundings,
    _tokens,
    chunk_corpus,
)


INDEX_SCHEMA_VERSION = "characterization-rag-index-0.1"


class Embedder(Protocol):
    model_name: str

    def encode_documents(self, texts: list[str]): ...
    def encode_query(self, text: str): ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RAGIndexSettings:
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    parent_chunk_chars: int = 6000
    child_chunk_chars: int = 1200
    child_overlap_chars: int = 180
    embedding_batch_size: int = 8
    embedding_max_length: int = 8192
    reranker_batch_size: int = 8
    reranker_max_length: int = 2048
    use_fp16: bool = False
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 96
    bm25_top_n: int = 50
    dense_top_n: int = 50
    fusion_top_n: int = 50
    rerank_top_n: int = 15
    rrf_offset: int = 60
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    diversity_lambda: float = 0.82
    same_parent_penalty: float = 0.12
    max_children_per_paper: int = 3

    def validate(self) -> None:
        if self.parent_chunk_chars < self.child_chunk_chars:
            raise ValueError("RAG parent chunks must be at least as large as child chunks")
        if self.child_chunk_chars < 400:
            raise ValueError("RAG child chunks must be at least 400 characters")
        if not 0 <= self.child_overlap_chars < self.child_chunk_chars:
            raise ValueError("RAG child overlap must be smaller than the child chunk")
        if min(
            self.embedding_batch_size,
            self.embedding_max_length,
            self.reranker_batch_size,
            self.reranker_max_length,
            self.hnsw_m,
            self.hnsw_ef_construction,
            self.hnsw_ef_search,
            self.bm25_top_n,
            self.dense_top_n,
            self.fusion_top_n,
            self.rerank_top_n,
            self.rrf_offset,
            self.max_children_per_paper,
        ) < 1:
            raise ValueError("RAG index and retrieval limits must be positive")
        if self.rerank_top_n > self.fusion_top_n:
            raise ValueError("rerank_top_n cannot exceed fusion_top_n")
        if self.hnsw_ef_search < self.dense_top_n:
            raise ValueError("hnsw_ef_search must be at least dense_top_n")
        if self.bm25_weight < 0 or self.dense_weight < 0:
            raise ValueError("RAG fusion weights cannot be negative")
        if self.bm25_weight + self.dense_weight <= 0:
            raise ValueError("At least one RAG fusion weight must be positive")
        if not 0 <= self.diversity_lambda <= 1:
            raise ValueError("RAG diversity_lambda must be between zero and one")
        if self.same_parent_penalty < 0:
            raise ValueError("RAG same_parent_penalty cannot be negative")


def rag_settings_from_mapping(config: dict[str, Any]) -> RAGIndexSettings:
    """Parse one TOML RAG table while keeping defaults in a single place."""
    defaults = RAGIndexSettings()
    return RAGIndexSettings(
        embedding_model=str(config.get("embedding_model", defaults.embedding_model)),
        reranker_model=str(config.get("reranker_model", defaults.reranker_model)),
        parent_chunk_chars=int(config.get("parent_chunk_chars", defaults.parent_chunk_chars)),
        child_chunk_chars=int(config.get("child_chunk_chars", defaults.child_chunk_chars)),
        child_overlap_chars=int(config.get("child_overlap_chars", defaults.child_overlap_chars)),
        embedding_batch_size=int(config.get("embedding_batch_size", defaults.embedding_batch_size)),
        embedding_max_length=int(config.get("embedding_max_length", defaults.embedding_max_length)),
        reranker_batch_size=int(config.get("reranker_batch_size", defaults.reranker_batch_size)),
        reranker_max_length=int(config.get("reranker_max_length", defaults.reranker_max_length)),
        use_fp16=bool(config.get("use_fp16", defaults.use_fp16)),
        hnsw_m=int(config.get("hnsw_m", defaults.hnsw_m)),
        hnsw_ef_construction=int(config.get("hnsw_ef_construction", defaults.hnsw_ef_construction)),
        hnsw_ef_search=int(config.get("hnsw_ef_search", defaults.hnsw_ef_search)),
        bm25_top_n=int(config.get("bm25_top_n", defaults.bm25_top_n)),
        dense_top_n=int(config.get("dense_top_n", defaults.dense_top_n)),
        fusion_top_n=int(config.get("fusion_top_n", defaults.fusion_top_n)),
        rerank_top_n=int(config.get("rerank_top_n", defaults.rerank_top_n)),
        rrf_offset=int(config.get("rrf_offset", defaults.rrf_offset)),
        bm25_weight=float(config.get("bm25_weight", defaults.bm25_weight)),
        dense_weight=float(config.get("dense_weight", defaults.dense_weight)),
        diversity_lambda=float(config.get("diversity_lambda", defaults.diversity_lambda)),
        same_parent_penalty=float(config.get("same_parent_penalty", defaults.same_parent_penalty)),
        max_children_per_paper=int(
            config.get("max_children_per_paper", defaults.max_children_per_paper)
        ),
    )


class BGEM3Embedder:
    """Lazy FlagEmbedding wrapper producing normalized BGE-M3 dense vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int,
        max_length: int,
        use_fp16: bool,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "BGE-M3 indexing requires FlagEmbedding; install setup/requirements.txt"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def _encode(self, texts: list[str]):
        result = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = result["dense_vecs"]
        return _normalize_rows(vectors)

    def encode_documents(self, texts: list[str]):
        return self._encode(texts)

    def encode_query(self, text: str):
        return self._encode([text])[0]


class BGEV2M3Reranker:
    """Lazy cross-encoder wrapper around bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int,
        max_length: int,
        use_fp16: bool,
    ) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "BGE reranking requires FlagEmbedding; install setup/requirements.txt"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self.model.compute_score(
            [[query, passage] for passage in passages],
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=False,
        )
        if isinstance(raw, (float, int)):
            return [float(raw)]
        return [float(value) for value in raw]


def corpus_fingerprint(corpus_root: str | Path) -> tuple[str, list[dict[str, Any]]]:
    root = Path(corpus_root).expanduser().resolve(strict=True)
    records: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".pdf", ".md", ".txt", ".sh", ".bash"}:
            continue
        digest = _sha256(path)
        relative = path.relative_to(root).as_posix()
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": digest}
        records.append(record)
        combined.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return combined.hexdigest(), records


def build_persistent_index(
    corpus_root: str | Path,
    index_root: str | Path,
    settings: RAGIndexSettings,
    *,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Build all generated files in a temporary directory, then publish atomically per file."""
    settings.validate()
    corpus = Path(corpus_root).expanduser().resolve(strict=True)
    destination = Path(index_root).expanduser().resolve()
    parents, children = chunk_corpus(
        corpus,
        parent_chunk_chars=settings.parent_chunk_chars,
        child_chunk_chars=settings.child_chunk_chars,
        child_overlap_chars=settings.child_overlap_chars,
    )
    if not children:
        raise ValueError(f"RAG corpus contains no indexable text: {corpus}")
    embedder = embedder or BGEM3Embedder(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
        use_fp16=settings.use_fp16,
    )
    vectors = embedder.encode_documents([child.text for child in children])
    np = _numpy()
    vectors = _normalize_rows(np.asarray(vectors, dtype=np.float32))
    if vectors.ndim != 2 or vectors.shape[0] != len(children):
        raise ValueError("Embedding backend returned an invalid child-vector matrix")
    fingerprint, files = corpus_fingerprint(corpus)
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(corpus),
        "corpus_sha256": fingerprint,
        "corpus_files": files,
        "parent_count": len(parents),
        "child_count": len(children),
        "embedding_model": embedder.model_name,
        "embedding_dimensions": int(vectors.shape[1]),
        "embedding_normalized": True,
        "distance": "cosine",
        "settings": asdict(settings),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_jsonl(temporary / "parents.jsonl", (asdict(parent) for parent in parents))
        _write_jsonl(
            temporary / "children.jsonl",
            ({"vector_id": index, **asdict(child)} for index, child in enumerate(children)),
        )
        (temporary / "bm25.json").write_text(
            json.dumps(_build_bm25_payload(children), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        with (temporary / "embeddings.npy").open("wb") as stream:
            np.save(stream, vectors, allow_pickle=False)
        hnswlib = _hnswlib()
        index = hnswlib.Index(space="cosine", dim=int(vectors.shape[1]))
        index.init_index(
            max_elements=len(children),
            ef_construction=settings.hnsw_ef_construction,
            M=settings.hnsw_m,
        )
        index.add_items(vectors, np.arange(len(children), dtype=np.int64))
        index.set_ef(settings.hnsw_ef_search)
        index.save_index(str(temporary / "dense.hnsw"))
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "parents.jsonl",
            "children.jsonl",
            "bm25.json",
            "embeddings.npy",
            "dense.hnsw",
            "manifest.json",
        ):
            os.replace(temporary / name, destination / name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


class PersistentCorpusRetriever:
    """BM25 + BGE-M3/HNSW + RRF + cross-encoder + MMR retrieval."""

    def __init__(
        self,
        corpus_root: str | Path,
        index_root: str | Path,
        settings: RAGIndexSettings,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        verify_corpus: bool = True,
        source_label: str = "paper_source",
    ) -> None:
        settings.validate()
        self.settings = settings
        if not source_label or any(character.isspace() for character in source_label):
            raise ValueError("RAG source_label must be a non-empty token")
        self.source_label = source_label
        self.corpus_root = Path(corpus_root).expanduser().resolve(strict=True)
        try:
            self.index_root = Path(index_root).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            build_command = (
                "./agentic jlse-rag build"
                if source_label == "operational_source"
                else "./agentic rag-index build"
            )
            raise ValueError(
                f"RAG index does not exist: {index_root}; run '{build_command}'"
            ) from exc
        self.manifest = json.loads(
            (self.index_root / "manifest.json").read_text(encoding="utf-8")
        )
        self._validate_manifest(verify_corpus)
        self.parents = {
            item["parent_id"]: ParentChunk(**item)
            for item in _read_jsonl(self.index_root / "parents.jsonl")
        }
        child_records = list(_read_jsonl(self.index_root / "children.jsonl"))
        child_records.sort(key=lambda item: int(item["vector_id"]))
        self.chunks = [
            PaperChunk(**{key: value for key, value in item.items() if key != "vector_id"})
            for item in child_records
        ]
        np = _numpy()
        self.embeddings = np.load(
            self.index_root / "embeddings.npy", allow_pickle=False, mmap_mode="r"
        )
        if self.embeddings.shape != (
            len(self.chunks), int(self.manifest["embedding_dimensions"])
        ):
            raise ValueError("RAG embeddings do not match child metadata")
        hnswlib = _hnswlib()
        self.hnsw = hnswlib.Index(
            space="cosine", dim=int(self.manifest["embedding_dimensions"])
        )
        self.hnsw.load_index(str(self.index_root / "dense.hnsw"), max_elements=len(self.chunks))
        self.hnsw.set_ef(settings.hnsw_ef_search)
        self.embedder = embedder or BGEM3Embedder(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
            use_fp16=settings.use_fp16,
        )
        self.reranker = reranker or BGEV2M3Reranker(
            settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            use_fp16=settings.use_fp16,
        )
        bm25 = json.loads((self.index_root / "bm25.json").read_text(encoding="utf-8"))
        self._lengths = [int(value) for value in bm25["document_lengths"]]
        self._average_length = float(bm25["average_document_length"])
        self._inverted_index = {
            term: [(int(document), int(frequency)) for document, frequency in postings]
            for term, postings in bm25["inverted_index"].items()
        }

    def _validate_manifest(self, verify_corpus: bool) -> None:
        if self.manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported RAG index schema; rebuild the index")
        indexed = self.manifest.get("settings", {})
        for key in (
            "parent_chunk_chars", "child_chunk_chars", "child_overlap_chars",
            "embedding_model", "embedding_max_length", "use_fp16",
            "hnsw_m", "hnsw_ef_construction",
        ):
            if indexed.get(key) != getattr(self.settings, key):
                raise ValueError(f"RAG index setting changed ({key}); rebuild the index")
        if verify_corpus:
            actual, _ = corpus_fingerprint(self.corpus_root)
            if actual != self.manifest.get("corpus_sha256"):
                raise ValueError("RAG paper corpus changed; rebuild the index")

    def _bm25(self, query: str) -> tuple[list[int], dict[int, float]]:
        query_terms = Counter(_tokens(query))
        count = len(self.chunks)
        scores: dict[int, float] = Counter()
        for term, query_frequency in query_terms.items():
            postings = self._inverted_index.get(term, [])
            document_frequency = len(postings)
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for index, frequency in postings:
                length = self._lengths[index]
                denominator = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * length / max(self._average_length, 1.0)
                )
                scores[index] += (
                    query_frequency * inverse_frequency * frequency * 2.5 / denominator
                )
        ranked = sorted(scores, key=lambda index: (-scores[index], index))
        return ranked[: self.settings.bm25_top_n], scores

    def _dense(self, query: str) -> tuple[list[int], dict[int, float]]:
        if not self.chunks:
            return [], {}
        vector = self.embedder.encode_query(query)
        count = min(len(self.chunks), self.settings.dense_top_n)
        labels, distances = self.hnsw.knn_query(vector, k=count)
        ranked = [int(label) for label in labels[0]]
        return ranked, {
            int(label): 1.0 - float(distance)
            for label, distance in zip(labels[0], distances[0])
        }

    def search(self, query: str, *, top_k: int = 6) -> list[SearchResult]:
        if top_k < 1 or not self.chunks:
            return []
        bm25_ranked, bm25_scores = self._bm25(query)
        dense_ranked, dense_scores = self._dense(query)
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_ranked, start=1)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_ranked, start=1)}
        candidates = set(bm25_ranks) | set(dense_ranks)
        rrf = {
            index: (
                self.settings.bm25_weight
                / (self.settings.rrf_offset + bm25_ranks[index])
                if index in bm25_ranks else 0.0
            ) + (
                self.settings.dense_weight
                / (self.settings.rrf_offset + dense_ranks[index])
                if index in dense_ranks else 0.0
            )
            for index in candidates
        }
        fused = sorted(candidates, key=lambda index: (-rrf[index], index))[
            : self.settings.fusion_top_n
        ]
        passages = [
            f"Title: {self.chunks[index].title}\nPage: {self.chunks[index].page}\n"
            f"Passage: {self.chunks[index].text}"
            for index in fused
        ]
        reranker_scores = self.reranker.score(query, passages)
        if len(reranker_scores) != len(fused):
            raise ValueError("Reranker returned the wrong number of scores")
        reranked = sorted(
            zip(fused, reranker_scores), key=lambda item: (-item[1], item[0])
        )[: self.settings.rerank_top_n]
        normalized_relevance = _minmax([score for _, score in reranked])

        selected: list[tuple[int, float, float]] = []
        remaining = [
            (index, raw_score, normalized)
            for (index, raw_score), normalized in zip(reranked, normalized_relevance)
        ]
        paper_counts: Counter[str] = Counter()
        while remaining and len(selected) < top_k:
            eligible = [
                item for item in remaining
                if paper_counts[self.chunks[item[0]].path] < self.settings.max_children_per_paper
            ]
            if not eligible:
                eligible = remaining

            def mmr(item: tuple[int, float, float]) -> tuple[float, str]:
                index, _raw_relevance, relevance = item
                redundancy = max(
                    0.0,
                    max((
                        float(self.embeddings[index] @ self.embeddings[chosen])
                        for chosen, _, _ in selected
                    ), default=0.0),
                )
                same_parent = any(
                    self.chunks[index].parent_id == self.chunks[chosen].parent_id
                    for chosen, _, _ in selected
                )
                value = (
                    self.settings.diversity_lambda * relevance
                    - (1.0 - self.settings.diversity_lambda) * redundancy
                    - (self.settings.same_parent_penalty if same_parent else 0.0)
                )
                return value, self.chunks[index].chunk_id

            chosen = max(eligible, key=mmr)
            selected.append(chosen)
            paper_counts[self.chunks[chosen[0]].path] += 1
            remaining.remove(chosen)

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=rrf[index],
                bm25_score=bm25_scores.get(index, 0.0),
                semantic_score=dense_scores.get(index, 0.0),
                rerank_score=float(rerank_score),
            )
            for index, rerank_score, _normalized in selected
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
            child = result.chunk
            parent = self.parents[child.parent_id]
            before, after = _parent_surroundings(parent.text, child, parent_context_chars)
            block = (
                f"[{self.source_label} {child.citation()}, parent_id={child.parent_id}, "
                f"rrf={result.score:.6f}, bm25={result.bm25_score:.4f}, "
                f"dense_cosine={result.semantic_score:.4f}, "
                f"cross_encoder={result.rerank_score:.4f}]\n"
                f"<matched_child>\n{child.text}\n</matched_child>\n"
                f"<parent_context_before>\n{before}\n</parent_context_before>\n"
                f"<parent_context_after>\n{after}\n</parent_context_after>"
            )
            remaining = max_chars - used
            if remaining <= 0:
                break
            blocks.append(block[:remaining].rstrip())
            used += len(blocks[-1]) + 2
        return "\n\n".join(blocks)


# Backward-compatible domain-specific name used by characterization.
PersistentPaperRetriever = PersistentCorpusRetriever


def index_status(
    corpus_root: str | Path,
    index_root: str | Path,
    settings: RAGIndexSettings | None = None,
) -> dict[str, Any]:
    index = Path(index_root).expanduser().resolve()
    manifest_path = index / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "missing", "index_root": str(index)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual, _ = corpus_fingerprint(corpus_root)
    reasons: list[str] = []
    if actual != manifest.get("corpus_sha256"):
        reasons.append("corpus_changed")
    if settings is not None:
        indexed = manifest.get("settings", {})
        for key in (
            "parent_chunk_chars", "child_chunk_chars", "child_overlap_chars",
            "embedding_model", "embedding_max_length", "use_fp16",
            "hnsw_m", "hnsw_ef_construction",
        ):
            if indexed.get(key) != getattr(settings, key):
                reasons.append(f"setting_changed:{key}")
    return {
        "status": "stale" if reasons else "ready",
        "stale_reasons": reasons,
        "index_root": str(index),
        "parent_count": manifest.get("parent_count"),
        "child_count": manifest.get("child_count"),
        "embedding_model": manifest.get("embedding_model"),
        "created_at": manifest.get("created_at"),
    }


def _normalize_rows(vectors):
    np = _numpy()
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding backend returned a zero vector")
    return array / norms


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _build_bm25_payload(children: list[PaperChunk]) -> dict[str, Any]:
    inverted: dict[str, list[list[int]]] = {}
    lengths: list[int] = []
    for document, child in enumerate(children):
        frequencies = Counter(_tokens(child.text))
        lengths.append(sum(frequencies.values()))
        for term, frequency in frequencies.items():
            inverted.setdefault(term, []).append([document, frequency])
    return {
        "algorithm": "okapi-bm25",
        "k1": 1.5,
        "b": 0.75,
        "document_lengths": lengths,
        "average_document_length": sum(lengths) / max(1, len(lengths)),
        "inverted_index": inverted,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield value


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Persistent RAG requires numpy") from exc
    return np


def _hnswlib():
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError(
            "Persistent RAG requires hnswlib; install setup/requirements.txt"
        ) from exc
    return hnswlib
