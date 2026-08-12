# Ptychography paper corpus

Place locally licensed or openly available ptychography papers in `papers/`.
The characterization RAG loader accepts `.pdf`, `.md`, and `.txt` files.

PDF chunks retain their source filename and one-based page number. Text
extraction can lose equation layout, symbols, captions, and figures, so claims
derived from those elements must remain subject to human review.

Do not place credentials, private notes, or papers that may not be processed by
the configured model service in this directory.

After changing papers, rebuild the generated index from the project root:

```bash
./agentic rag-index build
```

Generated BM25, BGE-M3 embedding, HNSW, and metadata files live in `index/` and
are intentionally ignored by Git.
