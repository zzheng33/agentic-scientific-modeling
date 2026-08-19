# Read-only MCP servers

The project exposes two local Model Context Protocol servers over `stdio`.
They are adapters over the existing implementations; MCP does not replace the
retrieval algorithms, LangGraph state, artifacts, or human-review gates.

## Codebase MCP

Start it from the repository root:

```bash
./agentic mcp-server codebase
```

It resolves `[application].path` from `config.toml`. Override that root only at
server startup when inspecting a different application:

```bash
./agentic mcp-server codebase --root /absolute/path/to/application
```

Tools:

- `codebase_list_files`
- `codebase_read_file`
- `codebase_search`

The root is immutable for the life of the process. Absolute paths, `..`,
symlinks, credentials, dependency trees, generated trees, unsupported binary
files, oversized files, and unbounded reads are rejected by `CodebaseTools`.

## Knowledge MCP

Build both indexes before starting the server:

```bash
./agentic rag-index build
./agentic jlse-rag build
./agentic mcp-server knowledge
```

Tools:

- `knowledge_index_status(corpus="papers" | "jlse")`
- `knowledge_search(corpus, query, top_k, parent_context_chars)`
- `knowledge_get_parent_context(corpus, parent_id, start_char, max_chars)`

`knowledge_search` uses the existing persistent retrieval pipeline:

```text
BM25 + BGE-M3 query embedding/HNSW cosine
→ weighted RRF
→ bge-reranker-v2-m3
→ MMR and per-paper diversity
→ matched child plus bounded parent context
```

The embedding and reranker models load lazily on the first search. Index status
and direct bounded parent reads do not load either model. Launching through
`./agentic` sets Hugging Face offline mode and loads only the project-local
snapshots under `models/huggingface/`.

## MCP host configuration

An MCP host should launch each command as a child process. Replace the project
path below with the absolute path on that computer:

```json
{
  "mcpServers": {
    "agentic-codebase": {
      "command": "/absolute/path/agentic-scientific-modeling/agentic",
      "args": ["mcp-server", "codebase"]
    },
    "agentic-knowledge": {
      "command": "/absolute/path/agentic-scientific-modeling/agentic",
      "args": ["mcp-server", "knowledge"]
    }
  }
}
```

`stdout` is reserved for the MCP wire protocol. Server diagnostics must go to
`stderr`. Both servers declare all tools as read-only and closed-world, but
those annotations are descriptive hints; the root confinement and input checks
are the actual enforcement boundary.
