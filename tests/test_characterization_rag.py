from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.characterization.rag import InMemoryTestRetriever
from agents.characterization.rag_store import (
    RAGIndexSettings,
    corpus_fingerprint,
    index_status,
)


class PaperRetrieverTests(unittest.TestCase):
    def test_parent_child_chunking_and_stable_citation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = (
                "Ptychographic reconstruction applies repeated two-dimensional FFT "
                "operations to diffraction patterns during iterative updates. "
                + "Supporting context about the forward model and probe update. " * 80
            )
            (root / "fft-notes.txt").write_text(text, encoding="utf-8")
            retriever = InMemoryTestRetriever.from_directory(
                root,
                parent_chunk_chars=1800,
                child_chunk_chars=500,
                child_overlap_chars=80,
            )

            results = retriever.search("ptychography FFT diffraction", top_k=1)

            self.assertEqual(len(results), 1)
            chunk = results[0].chunk
            self.assertEqual(chunk.path, "fft-notes.txt")
            self.assertIn(":parent001:child001", chunk.chunk_id)
            self.assertIn(chunk.parent_id, retriever.parents)
            self.assertLessEqual(len(chunk.text), 502)

    def test_hybrid_search_matches_morphological_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "method.txt").write_text(
                "Ptychographic imaging reconstructs an object from overlapping scans.",
                encoding="utf-8",
            )
            (root / "unrelated.txt").write_text(
                "A short note about laboratory scheduling.", encoding="utf-8"
            )
            retriever = InMemoryTestRetriever.from_directory(
                root, parent_chunk_chars=800, child_chunk_chars=400
            )

            results = retriever.search("ptychography reconstruction", top_k=1)

            self.assertEqual(results[0].chunk.path, "method.txt")
            self.assertGreater(results[0].semantic_score, 0)
            self.assertGreater(results[0].score, 0)

    def test_reranker_diversifies_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paper-a.txt").write_text(
                ("FFT diffraction reconstruction compute complexity. " * 80)
                + "\n\n"
                + ("FFT diffraction reconstruction probe update. " * 80),
                encoding="utf-8",
            )
            (root / "paper-b.txt").write_text(
                "Independent FFT diffraction reconstruction evidence from another paper. " * 20,
                encoding="utf-8",
            )
            retriever = InMemoryTestRetriever.from_directory(
                root,
                parent_chunk_chars=1600,
                child_chunk_chars=450,
                child_overlap_chars=50,
                diversity_lambda=0.65,
            )

            results = retriever.search("FFT diffraction reconstruction", top_k=3)

            self.assertGreaterEqual(len({result.chunk.parent_id for result in results}), 2)

    def test_rendered_context_contains_child_and_bounded_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paper.txt").write_text(
                "Background sentence. " * 60
                + "The detector shape controls FFT work and diffraction bytes. "
                + "Conclusion sentence. " * 60,
                encoding="utf-8",
            )
            retriever = InMemoryTestRetriever.from_directory(
                root, parent_chunk_chars=3000, child_chunk_chars=500
            )

            context = retriever.render_context(
                "detector shape FFT diffraction bytes",
                top_k=1,
                max_chars=1800,
                parent_context_chars=900,
            )

            self.assertIn("<matched_child>", context)
            self.assertIn("<parent_context_before>", context)
            self.assertIn("<parent_context_after>", context)
            self.assertIn("parent_id=", context)
            self.assertLessEqual(len(context), 1800)

    def test_empty_corpus_returns_empty_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = InMemoryTestRetriever.from_directory(
                directory, child_chunk_chars=400
            )
            self.assertEqual(retriever.render_context("ptychography"), "")

    def test_corpus_fingerprint_changes_with_paper_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper = root / "paper.txt"
            paper.write_text("first ptychography version", encoding="utf-8")
            first, records = corpus_fingerprint(root)
            paper.write_text("second ptychography version", encoding="utf-8")
            second, _ = corpus_fingerprint(root)
            self.assertNotEqual(first, second)
            self.assertEqual(records[0]["path"], "paper.txt")

    def test_shell_runbook_is_chunked_and_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "run_gh200.sh"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "module load cuda/12.9.1\n"
                "conda activate ptychopinn_torch_arm\n"
                + "python benchmark.py --device cuda\n" * 30,
                encoding="utf-8",
            )

            retriever = InMemoryTestRetriever.from_directory(
                root, parent_chunk_chars=1000, child_chunk_chars=400
            )
            _fingerprint, records = corpus_fingerprint(root)
            results = retriever.search("GH200 CUDA conda environment", top_k=1)

            self.assertEqual(records[0]["path"], "run_gh200.sh")
            self.assertEqual(results[0].chunk.path, "run_gh200.sh")

    def test_missing_persistent_index_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = index_status(root, root / "index")
            self.assertEqual(status["status"], "missing")

    def test_invalid_retrieval_funnel_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rerank_top_n"):
            RAGIndexSettings(fusion_top_n=10, rerank_top_n=15).validate()


if __name__ == "__main__":
    unittest.main()
