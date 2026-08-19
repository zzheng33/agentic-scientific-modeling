from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp import Client

from agents.characterization.rag import ParentChunk, PaperChunk, SearchResult
from agents.characterization.rag_store import RAGIndexSettings
from mcp_servers.codebase.server import create_server as create_codebase_server
from mcp_servers.knowledge.server import create_server as create_knowledge_server
from mcp_servers.knowledge.service import CorpusConfiguration, KnowledgeService


class _FakeRetriever:
    def __init__(self) -> None:
        parent = ParentChunk(
            parent_id="parent-1",
            title="Ptychography test",
            path="paper.pdf",
            page=2,
            text="before context matched child after context",
        )
        self.parents = {parent.parent_id: parent}
        self.chunk = PaperChunk(
            chunk_id="child-1",
            parent_id=parent.parent_id,
            title=parent.title,
            path=parent.path,
            page=parent.page,
            text="matched child",
            parent_start=15,
            parent_end=28,
        )

    def search(self, _query: str, *, top_k: int):
        return [
            SearchResult(
                chunk=self.chunk,
                score=0.02,
                bm25_score=1.2,
                semantic_score=0.8,
                rerank_score=0.9,
            )
        ][:top_k]


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_codebase_server_is_read_only_and_root_confined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
            server = create_codebase_server(root)
            async with Client(server) as client:
                listed = await client.list_tools()
                self.assertEqual(
                    {tool.name for tool in listed.tools},
                    {"codebase_list_files", "codebase_read_file", "codebase_search"},
                )
                self.assertTrue(
                    all(tool.annotations.read_only_hint for tool in listed.tools)
                )
                result = await client.call_tool(
                    "codebase_search",
                    {"query": "return", "path": ".", "file_pattern": "*.py"},
                )
                self.assertFalse(result.is_error)
                self.assertEqual(result.structured_content["matches"][0]["path"], "main.py")
                denied = await client.call_tool(
                    "codebase_read_file", {"path": ".env"}
                )
                self.assertTrue(denied.is_error)

    async def test_knowledge_server_returns_structured_hybrid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            index = root / "index"
            corpus.mkdir()
            index.mkdir()
            parent_record = {
                "parent_id": "parent-1",
                "title": "Ptychography test",
                "path": "paper.pdf",
                "page": 2,
                "text": "before context matched child after context",
            }
            (index / "parents.jsonl").write_text(
                json.dumps(parent_record) + "\n", encoding="utf-8"
            )
            settings = RAGIndexSettings()
            configurations = {
                name: CorpusConfiguration(
                    name=name,
                    corpus_root=corpus,
                    index_root=index,
                    settings=settings,
                    source_label=("paper_source" if name == "papers" else "operational_source"),
                    default_top_k=6,
                    default_parent_context_chars=400,
                )
                for name in ("papers", "jlse")
            }
            fake = _FakeRetriever()
            service = KnowledgeService(
                configurations,
                retriever_factory=lambda *_args, **_kwargs: fake,
            )
            server = create_knowledge_server(service)
            async with Client(server) as client:
                listed = await client.list_tools()
                self.assertEqual(
                    {tool.name for tool in listed.tools},
                    {
                        "knowledge_index_status",
                        "knowledge_search",
                        "knowledge_get_parent_context",
                    },
                )
                self.assertTrue(
                    all(tool.annotations.read_only_hint for tool in listed.tools)
                )
                result = await client.call_tool(
                    "knowledge_search",
                    {"corpus": "papers", "query": "memory scaling", "top_k": 1},
                )
                self.assertFalse(result.is_error)
                structured = result.structured_content
                self.assertEqual(structured["matches"][0]["chunk_id"], "child-1")
                self.assertIn("BGE-M3/HNSW", structured["retrieval"])
                parent = await client.call_tool(
                    "knowledge_get_parent_context",
                    {"corpus": "papers", "parent_id": "parent-1", "max_chars": 200},
                )
                self.assertFalse(parent.is_error)
                self.assertEqual(parent.structured_content["parent_id"], "parent-1")


if __name__ == "__main__":
    unittest.main()
