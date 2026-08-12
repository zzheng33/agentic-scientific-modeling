# Application Characterization Agent

This agent uses GPT-5.6 Sol through Argo's OpenAI-compatible Chat Completions API and
root-confined, read-only local tools to analyze an arbitrary scientific
application codebase. Its design, schemas, prompt, and runtime all live
in this directory.

The primary entry point is the persistent LangGraph CLI documented in
`RUN_WORKFLOW.txt`. It first pauses for candidate-input review, then derives
FLOP and I/O formulas from the approved inputs and pauses for final
characterization review. The commands below keep the original standalone agent
available for debugging.

## Setup

From the repository root, create the virtual environment, install dependencies,
and download the RAG models:

```bash
./setup/setup_venv.sh
```

Python 3.11 or newer is required. If `python3` resolves to an older interpreter,
set `AGENTIC_PYTHON` before running the setup script.

The setup script stores `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` under
`models/huggingface/` in the repository. That directory is intentionally ignored
by Git. To download or verify the models separately, run:

```bash
setup/venv/bin/python setup/download_models.py
setup/venv/bin/python setup/download_models.py --local-files-only
```

Set `AGENTIC_SKIP_MODEL_DOWNLOAD=1` only when setting up an environment that
will not build or query the literature index. `./agentic` automatically uses the
project-local model cache; `AGENTIC_HF_HOME` can override its location.

The repository includes `config.toml` as the runtime configuration:

```bash
editor config.toml
```

Set `openai.api_key` before running the agent. You may also configure the
application codebase:

```toml
[application]
path = "/path/to/application"

[output]
# Leave empty to use agents/characterization/output.
path = ""
```

A relative path is resolved from the directory containing `config.toml`.

## Run

```bash
venv/bin/python -m agents.characterization.cli \
  --context 'Optional entry-point or application hints'
```

You can override the configured application for one run:

```bash
venv/bin/python -m agents.characterization.cli /other/application \
  --output /path/to/new/analysis-directory
```

Command-line application and output paths take precedence over their configured
values. When `output.path` is empty or omitted, generated files are written to
`agents/characterization/output`. Repeated runs update the generated files in
that directory.

Use `--config /other/path/config.toml` to select a different config file.

The output directory contains:

- `application_characterization.yaml`
- `analysis_report.md`
- `human_review.yaml`

## Ptychography literature RAG

The persistent and standalone characterization paths can retrieve supporting
evidence from a local paper corpus. Configure it in `config.toml`:

```toml
[characterization.rag]
enabled = true
corpus_path = "knowledge/ptychography/papers"
index_path = "knowledge/ptychography/index"
embedding_model = "BAAI/bge-m3"
reranker_model = "BAAI/bge-reranker-v2-m3"
top_k = 6
parent_chunk_chars = 6000
child_chunk_chars = 1200
child_overlap_chars = 180
max_context_chars = 12000
parent_context_chars = 2600
bm25_top_n = 50
dense_top_n = 50
fusion_top_n = 50
rerank_top_n = 15
rrf_offset = 60
bm25_weight = 0.55
dense_weight = 0.45
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 96
diversity_lambda = 0.82
same_parent_penalty = 0.12
max_children_per_paper = 3
```

Add locally licensed or open papers as PDF, Markdown, or plain text files, then
build the persistent index before starting characterization:

```bash
./agentic rag-index build
./agentic rag-index status
./agentic rag-index search \
  --query "Which inputs determine ptychography FFT work and I/O bytes?"
```

The build command extracts PDF text page by page, creates parent and child
chunks, writes a persistent Okapi-BM25 inverted index, embeds every child with
`BAAI/bge-m3`, and builds a cosine HNSW index. It also records parents,
children, embeddings, corpus checksums, model names, and index settings under
`index_path`. Characterization refuses a stale index after a paper, chunking
parameter, embedding model, or structural HNSW setting changes.

Online retrieval runs BM25 top-N and BGE-M3/HNSW top-N in parallel, combines
their ranks with weighted Reciprocal Rank Fusion, and sends the fused candidates
through the `BAAI/bge-reranker-v2-m3` cross-encoder. MMR then uses normalized
cross-encoder relevance and BGE embedding cosine redundancy, with same-parent
and per-paper controls, to select the final children. Bounded windows from their
parents restore context without repeating the matched child.

Retrieved passages retain source IDs, parent IDs, filenames, and PDF page
numbers. Literature remains secondary evidence: application source is
authoritative for implementation claims, and layout-sensitive equations or
figures require human verification.

Despite the `.yaml` suffix, machine-readable artifacts are currently emitted as
JSON, which is a valid YAML subset. This avoids adding a YAML runtime dependency.

## Security boundary

The model cannot read files directly. It can only call three local tools:

- `list_files`
- `read_file`
- `search_code`

All paths are resolved beneath the supplied application root. The tools reject
path traversal, absolute paths, common secret files, private keys, oversized
files, binary files, generated directories, VCS internals, and vendored
dependencies.
