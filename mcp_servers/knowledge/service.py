"""Configuration and structured read-only access to persistent RAG indexes."""

from __future__ import annotations

import json
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from agents.characterization.rag_store import (
    PersistentCorpusRetriever,
    RAGIndexSettings,
    index_status,
    rag_settings_from_mapping,
)


CorpusName = Literal["papers", "jlse"]


@dataclass(frozen=True)
class CorpusConfiguration:
    name: CorpusName
    corpus_root: Path
    index_root: Path
    settings: RAGIndexSettings
    source_label: str
    default_top_k: int
    default_parent_context_chars: int


def _resolved_path(config_path: Path, value: Any, *, field: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Missing {field} in {config_path}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def load_corpus_configurations(config_path: str | Path) -> dict[CorpusName, CorpusConfiguration]:
    source = Path(config_path).expanduser().resolve(strict=True)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    mappings: list[tuple[CorpusName, dict[str, Any], str]] = [
        (
            "papers",
            document.get("characterization", {}).get("rag", {}),
            "paper_source",
        ),
        ("jlse", document.get("planner", {}).get("rag", {}), "operational_source"),
    ]
    configurations: dict[CorpusName, CorpusConfiguration] = {}
    for name, mapping, source_label in mappings:
        settings = rag_settings_from_mapping(mapping)
        settings.validate()
        top_k = int(mapping.get("top_k", 6))
        parent_chars = int(mapping.get("parent_context_chars", 2600))
        if top_k < 1 or parent_chars < 200:
            raise ValueError(f"Invalid retrieval limits for {name}")
        configurations[name] = CorpusConfiguration(
            name=name,
            corpus_root=_resolved_path(
                source, mapping.get("corpus_path"), field=f"{name}.corpus_path"
            ),
            index_root=_resolved_path(
                source, mapping.get("index_path"), field=f"{name}.index_path"
            ),
            settings=settings,
            source_label=source_label,
            default_top_k=top_k,
            default_parent_context_chars=parent_chars,
        )
    return configurations


class KnowledgeService:
    """Lazy, thread-safe access to the two configured persistent indexes."""

    def __init__(
        self,
        configurations: dict[CorpusName, CorpusConfiguration],
        *,
        retriever_factory: Callable[..., PersistentCorpusRetriever] = PersistentCorpusRetriever,
    ) -> None:
        if set(configurations) != {"papers", "jlse"}:
            raise ValueError("Knowledge MCP requires exactly the papers and jlse corpora")
        self.configurations = configurations
        self.retriever_factory = retriever_factory
        self._retrievers: dict[CorpusName, PersistentCorpusRetriever] = {}
        self._lock = threading.Lock()

    def _configuration(self, corpus: CorpusName) -> CorpusConfiguration:
        if corpus not in self.configurations:
            raise ValueError("corpus must be 'papers' or 'jlse'")
        return self.configurations[corpus]

    def _retriever(self, corpus: CorpusName) -> PersistentCorpusRetriever:
        configuration = self._configuration(corpus)
        with self._lock:
            retriever = self._retrievers.get(corpus)
            if retriever is None:
                retriever = self.retriever_factory(
                    configuration.corpus_root,
                    configuration.index_root,
                    configuration.settings,
                    source_label=configuration.source_label,
                )
                self._retrievers[corpus] = retriever
            return retriever

    def status(self, corpus: CorpusName) -> dict[str, Any]:
        configuration = self._configuration(corpus)
        result = index_status(
            configuration.corpus_root,
            configuration.index_root,
            configuration.settings,
        )
        return {"corpus": corpus, **result}

    def search(
        self,
        corpus: CorpusName,
        query: str,
        *,
        top_k: int | None = None,
        parent_context_chars: int | None = None,
    ) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if len(normalized_query) > 4000:
            raise ValueError("query must not exceed 4000 characters")
        configuration = self._configuration(corpus)
        requested_top_k = configuration.default_top_k if top_k is None else top_k
        requested_parent_chars = (
            configuration.default_parent_context_chars
            if parent_context_chars is None
            else parent_context_chars
        )
        if not 1 <= requested_top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 200 <= requested_parent_chars <= 8000:
            raise ValueError("parent_context_chars must be between 200 and 8000")

        retriever = self._retriever(corpus)
        matches: list[dict[str, Any]] = []
        half = requested_parent_chars // 2
        for result in retriever.search(normalized_query, top_k=requested_top_k):
            child = result.chunk
            parent = retriever.parents[child.parent_id]
            before = parent.text[max(0, child.parent_start - half) : child.parent_start]
            after = parent.text[child.parent_end : child.parent_end + half]
            matches.append(
                {
                    "chunk_id": child.chunk_id,
                    "parent_id": child.parent_id,
                    "title": child.title,
                    "path": child.path,
                    "page": child.page,
                    "citation": child.citation(),
                    "scores": {
                        "rrf": result.score,
                        "bm25": result.bm25_score,
                        "dense_cosine": result.semantic_score,
                        "cross_encoder": result.rerank_score,
                    },
                    "matched_child": child.text,
                    "parent_context_before": before,
                    "parent_context_after": after,
                }
            )
        return {
            "corpus": corpus,
            "query": normalized_query,
            "retrieval": (
                "BM25 + BGE-M3/HNSW cosine + weighted RRF + "
                "bge-reranker-v2-m3 + MMR"
            ),
            "matches": matches,
        }

    def parent_context(
        self,
        corpus: CorpusName,
        parent_id: str,
        *,
        start_char: int = 0,
        max_chars: int = 4000,
    ) -> dict[str, Any]:
        """Read one bounded parent directly from metadata without loading ML models."""
        configuration = self._configuration(corpus)
        if start_char < 0:
            raise ValueError("start_char must be non-negative")
        if not 200 <= max_chars <= 12000:
            raise ValueError("max_chars must be between 200 and 12000")
        parents_path = configuration.index_root / "parents.jsonl"
        if not parents_path.is_file():
            raise ValueError(f"RAG index is missing: {parents_path}")
        with parents_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("parent_id") != parent_id:
                    continue
                text = str(record["text"])
                if start_char > len(text):
                    raise ValueError(
                        f"start_char exceeds the {len(text)} character parent"
                    )
                end = min(len(text), start_char + max_chars)
                return {
                    "corpus": corpus,
                    "parent_id": parent_id,
                    "title": record["title"],
                    "path": record["path"],
                    "page": record.get("page"),
                    "start_char": start_char,
                    "end_char": end,
                    "total_chars": len(text),
                    "text": text[start_char:end],
                    "truncated": end < len(text),
                }
        raise ValueError(f"Unknown parent_id in {corpus} index: {parent_id}")
