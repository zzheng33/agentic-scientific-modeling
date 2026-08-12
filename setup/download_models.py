#!/usr/bin/env python3
"""Download the local RAG models into this project's Hugging Face cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODELS = (
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download characterization RAG models into the project."
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=project_root / "models" / "huggingface",
        help="Project-local Hugging Face home directory.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model repository to download; repeat to override the defaults.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify the project cache without accessing the network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_home = args.hf_home.expanduser().resolve()
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    from huggingface_hub import snapshot_download

    models = tuple(args.models or DEFAULT_MODELS)
    print(f"Hugging Face home: {hf_home}")
    for model in models:
        print(f"Downloading {model} ...")
        snapshot = snapshot_download(
            repo_id=model,
            cache_dir=hub_cache,
            local_files_only=args.local_files_only,
        )
        print(f"Ready: {model} -> {snapshot}")


if __name__ == "__main__":
    main()
