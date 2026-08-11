"""Read-only, root-confined tools exposed to the characterization model."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any


class ToolAccessError(ValueError):
    """Raised when a requested codebase operation is unsafe or invalid."""


_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "secrets.json",
}

_TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".f",
    ".f90",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".ipynb",
    ".java",
    ".jl",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "vendor",
}


class CodebaseTools:
    """Safe implementations and function schemas for local source inspection."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = 512_000,
        max_read_lines: int = 400,
        max_list_entries: int = 500,
        max_search_results: int = 100,
    ) -> None:
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ToolAccessError(f"Application root is not a directory: {resolved}")
        self.root = resolved
        self.max_file_bytes = max_file_bytes
        self.max_read_lines = max_read_lines
        self.max_list_entries = max_list_entries
        self.max_search_results = max_search_results

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "list_files",
                "description": (
                    "List application files and directories below a relative path. "
                    "Generated, dependency, VCS, and virtual-environment directories are skipped."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory; use . for root."},
                        "max_depth": {"type": "integer", "minimum": 0, "maximum": 6},
                        "pattern": {"type": "string", "description": "Filename glob, such as *.py or *.yaml."},
                    },
                    "required": ["path", "max_depth", "pattern"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a bounded line range from a text source file relative to the application root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "search_code",
                "description": (
                    "Search text source files for a literal, case-sensitive string and return "
                    "relative paths, line numbers, and matching lines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "description": "Relative directory; use . for root."},
                        "file_pattern": {"type": "string", "description": "Filename glob such as *.py."},
                    },
                    "required": ["query", "path", "file_pattern"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        methods = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
        }
        if name not in methods:
            raise ToolAccessError(f"Unknown tool: {name}")
        return methods[name](**arguments)

    def _resolve(self, relative_path: str, *, expect_directory: bool | None = None) -> Path:
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ToolAccessError("Tool paths must be relative to the application root")
        if ".." in requested.parts:
            raise ToolAccessError("Parent-directory traversal is not allowed")
        if self._is_sensitive(requested):
            raise ToolAccessError(f"Access to sensitive path is denied: {relative_path}")
        try:
            candidate = (self.root / requested).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ToolAccessError(f"Requested path does not exist: {relative_path}") from exc
        if not candidate.is_relative_to(self.root):
            raise ToolAccessError("Requested path escapes the application root")
        if expect_directory is True and not candidate.is_dir():
            raise ToolAccessError(f"Not a directory: {relative_path}")
        if expect_directory is False and not candidate.is_file():
            raise ToolAccessError(f"Not a file: {relative_path}")
        return candidate

    @staticmethod
    def _is_sensitive(path: Path) -> bool:
        lowered = [part.lower() for part in path.parts]
        return any(part in _SENSITIVE_NAMES for part in lowered) or any(
            part.endswith((".pem", ".key", ".p12", ".pfx")) for part in lowered
        )

    @staticmethod
    def _is_skipped(path: Path) -> bool:
        return any(part in _SKIP_DIRS for part in path.parts)

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        return path.suffix.lower() in _TEXT_SUFFIXES and not CodebaseTools._is_sensitive(path)

    def list_files(self, path: str, max_depth: int, pattern: str) -> dict[str, Any]:
        directory = self._resolve(path, expect_directory=True)
        max_depth = min(max(max_depth, 0), 6)
        entries: list[dict[str, Any]] = []
        base_depth = len(directory.parts)

        for current, dirs, files in os.walk(directory):
            current_path = Path(current)
            depth = len(current_path.parts) - base_depth
            dirs[:] = sorted(
                d for d in dirs if d not in _SKIP_DIRS and not (current_path / d).is_symlink()
            )
            if depth >= max_depth:
                dirs[:] = []
            for filename in sorted(files):
                candidate = current_path / filename
                relative = candidate.relative_to(self.root)
                if candidate.is_symlink():
                    continue
                if self._is_skipped(relative) or self._is_sensitive(relative):
                    continue
                if not fnmatch.fnmatch(filename, pattern):
                    continue
                entries.append({"path": relative.as_posix(), "bytes": candidate.stat().st_size})
                if len(entries) >= self.max_list_entries:
                    return {"entries": entries, "truncated": True}
        return {"entries": entries, "truncated": False}

    def read_file(self, path: str, start_line: int, end_line: int) -> dict[str, Any]:
        source = self._resolve(path, expect_directory=False)
        if not self._is_text_file(source):
            raise ToolAccessError(f"Unsupported or non-text source file: {path}")
        size = source.stat().st_size
        if size > self.max_file_bytes:
            raise ToolAccessError(f"File exceeds {self.max_file_bytes} byte limit: {path}")
        if start_line < 1 or end_line < start_line:
            raise ToolAccessError("Invalid line range")
        requested_lines = min(end_line - start_line + 1, self.max_read_lines)
        actual_end = start_line + requested_lines - 1
        output: list[str] = []
        with source.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line_number < start_line:
                    continue
                if line_number > actual_end:
                    break
                output.append(f"{line_number}: {line.rstrip()}")
        return {
            "path": source.relative_to(self.root).as_posix(),
            "start_line": start_line,
            "end_line": start_line + len(output) - 1 if output else start_line - 1,
            "content": "\n".join(output),
            "truncated": end_line > actual_end,
        }

    def search_code(self, query: str, path: str, file_pattern: str) -> dict[str, Any]:
        if not query:
            raise ToolAccessError("Search query must not be empty")
        directory = self._resolve(path, expect_directory=True)
        matches: list[dict[str, Any]] = []
        for current, dirs, files in os.walk(directory):
            current_path = Path(current)
            dirs[:] = sorted(
                d for d in dirs if d not in _SKIP_DIRS and not (current_path / d).is_symlink()
            )
            for filename in sorted(files):
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                source = Path(current) / filename
                relative = source.relative_to(self.root)
                if source.is_symlink():
                    continue
                if not self._is_text_file(source) or self._is_skipped(relative):
                    continue
                if source.stat().st_size > self.max_file_bytes:
                    continue
                with source.open("r", encoding="utf-8", errors="replace") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if query in line:
                            matches.append(
                                {
                                    "path": relative.as_posix(),
                                    "line": line_number,
                                    "text": line.rstrip()[:1000],
                                }
                            )
                            if len(matches) >= self.max_search_results:
                                return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}
