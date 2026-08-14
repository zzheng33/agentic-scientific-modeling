"""Single command-line entry point for the persistent scientific workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from langgraph.types import Command

from agents.planner.runner import PlannerConfig
from agents.characterization.runner import CharacterizationConfig
from agents.characterization.rag_store import (
    PersistentCorpusRetriever,
    PersistentPaperRetriever,
    build_persistent_index,
    index_status,
)
from schemas.artifacts import ArtifactRef
from workflow.approvals import validate_review_file
from workflow.artifacts import ArtifactStore, sha256_file
from workflow.graph import open_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent scientific workflow CLI")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Start a new workflow and pause for review")
    start.add_argument("--application", type=Path)
    start.add_argument("--workflow-id")
    start.add_argument(
        "--workflow-type",
        choices=("characterization", "foundation"),
        default="characterization",
    )
    start.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.toml")
    start.add_argument("--context", default="")

    status = commands.add_parser("status", help="Show checkpointed workflow status")
    status.add_argument("--workflow-id")

    template = commands.add_parser("review-template", help="Show the pending review path")
    template.add_argument("--workflow-id")

    resume = commands.add_parser(
        "resume",
        help="Submit the pending YAML review or continue an interrupted node",
    )
    resume.add_argument("--workflow-id")
    resume.add_argument("--review-file", type=Path)

    plan = commands.add_parser(
        "plan",
        help="Start experiment planning from an approved characterization",
    )
    plan.add_argument("--workflow-id")

    benchmark = commands.add_parser(
        "benchmark",
        help="Prepare reviewed remote scripts or execute an approved manifest",
    )
    benchmark.add_argument("--workflow-id")
    benchmark.add_argument(
        "--execute",
        action="store_true",
        help="Execute the human-approved benchmark manifest",
    )

    retry = commands.add_parser("continue", help="Retry/continue a non-review graph node")
    retry.add_argument("--workflow-id")

    hash_command = commands.add_parser("hash-file", help="Print a file SHA-256 for edit review")
    hash_command.add_argument("--path", type=Path, required=True)
    rag = commands.add_parser(
        "rag-index", help="Build, inspect, or query the characterization paper index"
    )
    rag.add_argument("action", choices=("build", "status", "search"))
    rag.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.toml")
    rag.add_argument("--query", help="Required for the search action")
    jlse_rag = commands.add_parser(
        "jlse-rag", help="Build, inspect, or query the JLSE operational runbook index"
    )
    jlse_rag.add_argument("action", choices=("build", "status", "search"))
    jlse_rag.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.toml")
    jlse_rag.add_argument("--query", help="Required for the search action")
    return parser


def workflow_paths(runs_root: Path, workflow_id: str) -> tuple[Path, ArtifactStore]:
    ArtifactStore.validate_workflow_id(workflow_id)
    root = runs_root.expanduser().resolve()
    run_dir = root / workflow_id
    return run_dir, ArtifactStore(run_dir)


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve(strict=True)
    try:
        with config_path.open("rb") as stream:
            return tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML configuration in {config_path}: {exc}") from exc


def configured_workflow_id(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "workflow_id", "") or "").strip()
    if explicit:
        return ArtifactStore.validate_workflow_id(explicit)
    document = load_config(PROJECT_ROOT / "config.toml")
    configured = str(document.get("workflow", {}).get("id", "")).strip()
    if not configured:
        raise ValueError("Set workflow.id in config.toml or pass --workflow-id")
    return ArtifactStore.validate_workflow_id(configured)


def configured_application(config_path: Path) -> Path:
    document = load_config(config_path)
    configured = str(document.get("application", {}).get("path", "")).strip()
    if not configured:
        raise ValueError("Set application.path in config.toml or pass --application")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.expanduser().resolve().parent / candidate
    return candidate.resolve(strict=True)


def graph_config(workflow_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": workflow_id}}


def print_state(values: dict[str, Any], next_nodes: tuple[str, ...] | list[str]) -> None:
    pending = values.get("pending_review") or {}
    print(f"workflow_id: {values.get('workflow_id')}")
    print(f"status: {values.get('workflow_status')}")
    print(f"stage: {values.get('current_stage')}")
    print(f"next: {', '.join(next_nodes) if next_nodes else 'END'}")
    if pending:
        print(f"artifact: {pending['artifact']['path']}")
        print(f"artifact_sha256: {pending['artifact']['sha256']}")
        print(f"review_file: {pending['review_path']}")


def require_existing_run(run_dir: Path) -> None:
    if not run_dir.is_dir():
        raise ValueError(f"Workflow does not exist: {run_dir}")


def workflow_type(store: ArtifactStore) -> str:
    return str(store.load_metadata().get("workflow_type", "foundation"))


def source_revision(application: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(application), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def start_workflow(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    config_path = args.config.expanduser().resolve() if args.config else None
    application = (
        args.application.expanduser().resolve(strict=True)
        if args.application
        else configured_application(config_path)
    )
    if args.workflow_type == "characterization" and not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    store.initialize(
        workflow_id,
        application,
        workflow_type=args.workflow_type,
        config_path=config_path,
        user_context=args.context,
    )
    initial = {
        "workflow_id": workflow_id,
        "workflow_type": args.workflow_type,
        "application_path": str(application),
        "config_path": str(config_path) if config_path else None,
        "user_context": args.context,
        "run_dir": str(run_dir),
        "source_revision": source_revision(application),
        "workflow_status": "starting",
        "input_revision": 0,
        "characterization_revision": 0,
        "review_history": [],
    }
    with open_graph(run_dir, args.workflow_type) as graph:
        graph.invoke(initial, config=graph_config(workflow_id))
        snapshot = graph.get_state(graph_config(workflow_id))
    print_state(snapshot.values, snapshot.next)


def show_status(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    require_existing_run(run_dir)
    with open_graph(run_dir, workflow_type(store)) as graph:
        snapshot = graph.get_state(graph_config(workflow_id))
    print_state(snapshot.values, snapshot.next)


def show_review_template(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    require_existing_run(run_dir)
    with open_graph(run_dir, workflow_type(store)) as graph:
        snapshot = graph.get_state(graph_config(workflow_id))
    pending = snapshot.values.get("pending_review")
    if not pending:
        raise ValueError("Workflow has no pending review")
    print(run_dir / pending["review_path"])


def resume_workflow(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    require_existing_run(run_dir)
    config = graph_config(workflow_id)
    with open_graph(run_dir, workflow_type(store)) as graph:
        before = graph.get_state(config)
        pending = before.values.get("pending_review")
        if pending:
            review_path = args.review_file or (run_dir / pending["review_path"])
            review = validate_review_file(review_path, pending, store)
            graph.invoke(Command(resume=review), config=config)
        else:
            graph.invoke(None, config=config)
        after = graph.get_state(config)
    print_state(after.values, after.next)


def continue_workflow(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    require_existing_run(run_dir)
    config = graph_config(workflow_id)
    with open_graph(run_dir, workflow_type(store)) as graph:
        before = graph.get_state(config)
        if before.values.get("pending_review"):
            raise ValueError("Workflow is awaiting review; use the resume command")
        graph.invoke(None, config=config)
        after = graph.get_state(config)
    print_state(after.values, after.next)


def start_planning(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    require_existing_run(run_dir)
    config = graph_config(workflow_id)
    with open_graph(run_dir, workflow_type(store)) as graph:
        before = graph.get_state(config)
        values = before.values
        if values.get("pending_review"):
            raise ValueError("Complete the current pending review before planning")
        if not values.get("approved_characterization_ref"):
            raise ValueError("Planning requires an approved characterization")
        if values.get("experiment_plan_ref"):
            raise ValueError("Experiment planning has already started")
        graph.update_state(
            config,
            {
                "route": "start_planning",
                "current_stage": "experiment_planning",
                "workflow_status": "planning_starting",
                "planning_revision": 0,
                "planning_feedback": None,
            },
            as_node="apply_characterization_review",
        )
        graph.invoke(None, config=config)
        after = graph.get_state(config)
    print_state(after.values, after.next)


def benchmark_workflow(args: argparse.Namespace) -> None:
    workflow_id = configured_workflow_id(args)
    run_dir, store = workflow_paths(args.runs_root, workflow_id)
    require_existing_run(run_dir)
    config = graph_config(workflow_id)
    with open_graph(run_dir, workflow_type(store)) as graph:
        before = graph.get_state(config)
        values = before.values
        if values.get("pending_review"):
            raise ValueError("Complete the current pending review first")
        if not values.get("config_path"):
            raise ValueError("Benchmark workflow requires the workflow config path")
        if args.execute:
            if not values.get("approved_benchmark_manifest_ref"):
                raise ValueError("Approve the benchmark run manifest before execution")
            if values.get("measurements_ref"):
                raise ValueError("Benchmark measurements already exist for this workflow")
            graph.update_state(
                config,
                {
                    "route": "remote_execute_benchmark",
                    "current_stage": "remote_execution",
                    "workflow_status": "remote_executing",
                },
                as_node="apply_benchmark_review",
            )
        else:
            if not values.get("approved_experiment_plan_ref"):
                raise ValueError("Benchmark preparation requires an approved experiment plan")
            if values.get("benchmark_manifest_ref"):
                raise ValueError("Benchmark preparation has already started")
            graph.update_state(
                config,
                {
                    "route": "start_benchmark",
                    "benchmark_revision": 0,
                    "benchmark_feedback": None,
                    "current_stage": "benchmark_preparation",
                    "workflow_status": "benchmark_preparing",
                },
                as_node="apply_planning_review",
            )
        graph.invoke(None, config=config)
        after = graph.get_state(config)
    print_state(after.values, after.next)


def manage_rag_index(args: argparse.Namespace) -> None:
    config = CharacterizationConfig.from_file(args.config)
    if config.rag_corpus_path is None or config.rag_index_path is None:
        raise ValueError("Configure characterization.rag corpus_path and index_path")
    if args.action == "status":
        print(
            json.dumps(
                index_status(
                    config.rag_corpus_path,
                    config.rag_index_path,
                    config.rag_settings,
                ),
                indent=2,
            )
        )
        return
    if args.action == "build":
        manifest = build_persistent_index(
            config.rag_corpus_path,
            config.rag_index_path,
            config.rag_settings,
        )
        print(json.dumps(manifest, indent=2))
        return
    query = str(args.query or "").strip()
    if not query:
        raise ValueError("rag-index search requires --query")
    retriever = PersistentPaperRetriever(
        config.rag_corpus_path,
        config.rag_index_path,
        config.rag_settings,
    )
    print(
        retriever.render_context(
            query,
            top_k=config.rag_top_k,
            max_chars=config.rag_max_context_chars,
            parent_context_chars=config.rag_parent_context_chars,
        )
    )


def manage_jlse_rag_index(args: argparse.Namespace) -> None:
    config = PlannerConfig.from_file(args.config)
    if config.operational_rag_corpus_path is None or config.operational_rag_index_path is None:
        raise ValueError("Configure planner.rag corpus_path and index_path")
    if args.action == "status":
        print(
            json.dumps(
                index_status(
                    config.operational_rag_corpus_path,
                    config.operational_rag_index_path,
                    config.operational_rag_settings,
                ),
                indent=2,
            )
        )
        return
    if args.action == "build":
        manifest = build_persistent_index(
            config.operational_rag_corpus_path,
            config.operational_rag_index_path,
            config.operational_rag_settings,
        )
        print(json.dumps(manifest, indent=2))
        return
    query = str(args.query or "").strip()
    if not query:
        raise ValueError("jlse-rag search requires --query")
    retriever = PersistentCorpusRetriever(
        config.operational_rag_corpus_path,
        config.operational_rag_index_path,
        config.operational_rag_settings,
        source_label="operational_source",
    )
    print(
        retriever.render_context(
            query,
            top_k=config.operational_rag_top_k,
            max_chars=config.operational_rag_max_context_chars,
            parent_context_chars=config.operational_rag_parent_context_chars,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "start":
            start_workflow(args)
        elif args.command == "status":
            show_status(args)
        elif args.command == "review-template":
            show_review_template(args)
        elif args.command == "resume":
            resume_workflow(args)
        elif args.command == "plan":
            start_planning(args)
        elif args.command == "benchmark":
            benchmark_workflow(args)
        elif args.command == "continue":
            continue_workflow(args)
        elif args.command == "hash-file":
            print(sha256_file(args.path.expanduser().resolve(strict=True)))
        elif args.command == "rag-index":
            manage_rag_index(args)
        elif args.command == "jlse-rag":
            manage_jlse_rag_index(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
