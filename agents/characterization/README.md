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

Create a virtual environment and install the optional agent dependency:

```bash
./setup_venv.sh
```

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
