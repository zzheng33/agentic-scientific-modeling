"""Command-line entry point for the Experiment Planning Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import PlannerConfig, PlanningAgent, load_json_object, write_plan_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a human-review experiment plan from an approved characterization."
    )
    parser.add_argument(
        "--characterization",
        type=Path,
        default=None,
        help="Approved characterization; overrides planner.characterization_path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Planner output directory; overrides planner.output_path",
    )
    parser.add_argument("--context", default="", help="Optional experiment-planning guidance")
    parser.add_argument("--config", type=Path, default=None, help="TOML configuration path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PlannerConfig.from_file(args.config)
    characterization_path = args.characterization or config.characterization_path
    output_path = args.output or config.output_path
    characterization = load_json_object(characterization_path, "characterization")
    plan = PlanningAgent(config).plan(characterization, user_context=args.context)
    write_plan_artifacts(plan, output_path)
    print(f"Experiment plan draft written to {output_path}")


if __name__ == "__main__":
    main()
