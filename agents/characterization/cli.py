"""Command-line entry point for the Application Characterization Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runner import CharacterizationAgent, CharacterizationConfig, write_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a scientific application codebase with GPT-5.6 Sol and read-only local tools."
    )
    parser.add_argument(
        "application_root",
        type=Path,
        nargs="?",
        help="Application root; overrides application.path in config.toml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New artifact directory; overrides output.path in config.toml",
    )
    parser.add_argument("--context", default="", help="Optional human context or entry-point hint")
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Human review file used to revise or approve the current characterization",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config path; defaults to the repository config.toml",
    )
    return parser


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} file {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return document


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = CharacterizationConfig.from_file(args.config)
    application_root = args.application_root or config.application_path
    if application_root is None:
        parser.error(
            "application root is required: pass it as an argument or set application.path in config.toml"
        )
    output_path = args.output or config.output_path
    if output_path is None:
        parser.error("output path is required")
    agent = CharacterizationAgent(config)
    if args.review is not None:
        draft_path = output_path / "application_characterization.yaml"
        draft = load_json_object(draft_path, "characterization")
        human_review = load_json_object(args.review, "human review")
        artifact = agent.revise(application_root, draft, human_review)
        write_artifacts(artifact, output_path, initialize_review=False)
        print(f"Reviewed characterization written to {output_path}")
    else:
        artifact = agent.analyze(application_root, user_context=args.context)
        write_artifacts(artifact, output_path)
        print(f"Characterization draft written to {output_path}")


if __name__ == "__main__":
    main()
