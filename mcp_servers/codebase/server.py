"""Read-only MCP interface over the existing root-confined codebase tools."""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from agents.characterization.tools import CodebaseTools


PROJECT_ROOT = Path(__file__).resolve().parents[2]
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


class FileEntry(BaseModel):
    path: str
    bytes: int


class FileListResult(BaseModel):
    entries: list[FileEntry]
    truncated: bool


class FileReadResult(BaseModel):
    path: str
    start_line: int
    end_line: int
    content: str
    truncated: bool


class CodeMatch(BaseModel):
    path: str
    line: int
    text: str


class CodeSearchResult(BaseModel):
    matches: list[CodeMatch]
    truncated: bool


def configured_application_root(config_path: str | Path) -> Path:
    """Resolve application.path without loading API credentials or an LLM config."""
    source = Path(config_path).expanduser().resolve(strict=True)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    raw = str(document.get("application", {}).get("path", "")).strip()
    if not raw:
        raise ValueError(f"Set application.path in {source}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return candidate.resolve(strict=True)


def create_server(root: str | Path) -> MCPServer:
    """Create a server permanently confined to one application source tree."""
    tools = CodebaseTools(root)
    server = MCPServer(
        "agentic-codebase",
        version="0.1.0",
        instructions=(
            "Read-only inspection of one configured application source tree. "
            "Paths are relative to that root; traversal, secrets, dependencies, "
            "generated files, symlinks, and unsupported binary files are denied."
        ),
    )

    @server.tool(
        title="List application files",
        annotations=READ_ONLY,
    )
    def codebase_list_files(
        path: Annotated[str, Field(description="Relative directory; use . for the root")] = ".",
        max_depth: Annotated[int, Field(ge=0, le=6)] = 2,
        pattern: Annotated[str, Field(description="Filename glob such as *.py")] = "*",
    ) -> FileListResult:
        """List bounded, non-sensitive files below the configured application root."""
        return FileListResult.model_validate(tools.list_files(path, max_depth, pattern))

    @server.tool(
        title="Read application source",
        annotations=READ_ONLY,
    )
    def codebase_read_file(
        path: Annotated[str, Field(description="File path relative to the application root")],
        start_line: Annotated[int, Field(ge=1)] = 1,
        end_line: Annotated[int, Field(ge=1)] = 200,
    ) -> FileReadResult:
        """Read a bounded line range from a non-sensitive text source file."""
        return FileReadResult.model_validate(tools.read_file(path, start_line, end_line))

    @server.tool(
        title="Search application source",
        annotations=READ_ONLY,
    )
    def codebase_search(
        query: Annotated[str, Field(min_length=1, description="Literal case-sensitive text")],
        path: Annotated[str, Field(description="Relative directory; use . for the root")] = ".",
        file_pattern: Annotated[str, Field(description="Filename glob such as *.py")] = "*",
    ) -> CodeSearchResult:
        """Search bounded application text files for a literal string."""
        return CodeSearchResult.model_validate(
            tools.search_code(query, path, file_pattern)
        )

    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the read-only Codebase MCP server")
    parser.add_argument("--root", type=Path, help="Application root to expose")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.toml")
    return parser


def main() -> None:
    args = _parser().parse_args()
    environment_root = os.environ.get("AGENTIC_APPLICATION_ROOT", "").strip()
    root = args.root or (Path(environment_root) if environment_root else None)
    if root is None:
        root = configured_application_root(args.config)
    create_server(root).run()


if __name__ == "__main__":
    main()
