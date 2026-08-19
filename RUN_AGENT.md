# Agent quick start

This guide runs the complete local-control/remote-execution workflow. Run all
commands from the repository root.

The supported execution path is remote-only:

```text
Characterization Agent
→ Planning Agent
→ Execution Script Agent
→ human script review
→ Remote dataset generation on the JLSE login node
→ Remote Executor (SSH/SCP/Cobalt job)
→ measurement validation and resource-model fitting
→ SystemFlow mapping review
→ SystemFlow integration validation and review
→ automatic publication into SystemFlow
```

There is no local Benchmark Runner and no built-in synthetic dataset generator.
The Execution Script Agent generates application-specific
`dataset_generation.sh` and `benchmark_job.sh` files. Only human-approved
versions are sent to the configured remote system.

## 1. Set up the local Python environment

Python 3.11 or newer is required. First check the interpreter you intend to
use:

```bash
/path/to/python3.11 --version
```

From the project root, create the local virtual environment, install all Python
dependencies, and download the BGE-M3 embedding and reranker models:

```bash
AGENTIC_PYTHON=/path/to/python3.11 ./setup/setup_venv.sh
```

For example, if `python3.11` is already on `PATH`:

```bash
AGENTIC_PYTHON="$(command -v python3.11)" ./setup/setup_venv.sh
```

The setup command creates:

```text
setup/venv/          local Python virtual environment
models/huggingface/  BGE-M3 and reranker model files
```

### Download the Hugging Face models

`setup/setup_venv.sh` downloads the models automatically after installing the
Python packages. It runs the equivalent of:

```bash
setup/venv/bin/python setup/download_models.py \
  --hf-home "$PWD/models/huggingface"
```

The downloader uses `huggingface_hub.snapshot_download` and downloads these
public repositories:

```text
BAAI/bge-m3
BAAI/bge-reranker-v2-m3
```

You do not need to create `models/huggingface/hub` manually. The downloader
creates it and produces a Hugging Face cache layout similar to:

```text
models/huggingface/
└── hub/
    ├── models--BAAI--bge-m3/
    │   ├── blobs/
    │   ├── refs/
    │   └── snapshots/<revision>/
    └── models--BAAI--bge-reranker-v2-m3/
        ├── blobs/
        ├── refs/
        └── snapshots/<revision>/
```

If the environment was created with model downloading disabled, download both
models later with the same command:

```bash
setup/venv/bin/python setup/download_models.py \
  --hf-home "$PWD/models/huggingface"
```

To download explicit model repositories, repeat `--model`:

```bash
setup/venv/bin/python setup/download_models.py \
  --hf-home "$PWD/models/huggingface" \
  --model BAAI/bge-m3 \
  --model BAAI/bge-reranker-v2-m3
```

After copying the cache to another computer, verify that both snapshots are
available without network access:

```bash
setup/venv/bin/python setup/download_models.py \
  --hf-home "$PWD/models/huggingface" \
  --local-files-only
```

This verification must print `Ready` for both models. Because the repositories
are public, a Hugging Face token is normally not required.

The `./agentic` launcher automatically uses `setup/venv/bin/python` and sets
the project-local Hugging Face cache. Runtime model loading is offline-only, so
agents cannot silently download or replace model files. Model downloads happen
only through `setup/download_models.py`. You therefore do not need to activate
the virtual environment for normal commands. To work with its Python directly,
activation is optional:

```bash
source setup/venv/bin/activate
python --version
python -c "import langgraph, hnswlib, torch; print('local environment ready')"
deactivate
```

If the model files have already been copied into `models/huggingface/`, skip
downloading them while rebuilding only the Python environment:

```bash
AGENTIC_PYTHON=/path/to/python3.11 \
AGENTIC_SKIP_MODEL_DOWNLOAD=1 \
./setup/setup_venv.sh
```

Next, build the two indexes:

```bash
./agentic rag-index build
./agentic jlse-rag build

./agentic rag-index status
./agentic jlse-rag status
```

Both status commands should report `ready`. You do not need to rebuild an
index unless its papers, scripts, or runbooks change.

### Optional: expose the local knowledge through MCP

Setup also installs the official MCP Python SDK. Two read-only stdio servers are
available:

```bash
# Root-confined application listing, reading, and literal search
./agentic mcp-server codebase

# Hybrid retrieval over papers and JLSE runbooks
./agentic mcp-server knowledge
```

Normally an MCP host launches these commands as child processes; running one
directly appears to wait because `stdout` is carrying the MCP protocol. The
Codebase MCP reads `[application].path`. The Knowledge MCP reads the two RAG
configurations and uses the already-built indexes. It supports `papers` and
`jlse` as explicit corpus names and loads BGE-M3/reranker weights lazily on the
first search.

These servers do not expose file writes, shell execution, SSH, `qsub`, artifact
approval, or index mutation. See `mcp_servers/README.md` for their tools and an
MCP host configuration example.

## 2. Configure the run

Edit `config.toml` and check at least these values:

```toml
[openai]
api_key = "your-argo-identity-or-key"
base_url = "https://apps.inside.anl.gov/argoapi/v1"

[workflow]
# Use a new ID for every new workflow.
id = "ptychi_002"

[application]
path = "../pty-chi/"

[[machine]]
# Repeat [[machine]] for each JLSE accelerator/queue profile.
accelerator = "GH200"
queue = "gpu_gh200"
nodes = 1
walltime_minutes = 30
module_path = "/soft/modulefiles"
modules = ["cuda/12.9.1", "conda/nvidia/suse15.6/2025.01-11"]
conda_env = "ptychopinn_torch_arm"
remote_monitor_script = "/home/zhong.zheng/PtychoPINN/scripts/monitor_gpu_power.py"
device = "cuda"
power_vendor = "nvidia"
devices = "0"
power_interval_s = 0.2

[remote_executor]
enabled = true
host = "zhong.zheng@login.jlse.anl.gov"
ssh_password = "your-login-password"
ssh_duo_choice = "1"
remote_runs_root = "/home/zhong.zheng/agentic-runs"
remote_application_path = "/home/zhong.zheng/pty-chi"
upload_application = true
qsub_command = "qsub"
qstat_command = "qstat"

[dataset_runtime]
# Login-node environment used before qsub. It must provide numpy and h5py.
python = "/home/zhong.zheng/venv/bin/python"

[systemflow]
path = "../systemflow"
```

With `upload_application = true`, the executor packages the configured local
application source. Set it to false only when intentionally reusing the checkout
at `remote_application_path`.

`ssh_password` and `ssh_duo_choice` automate the two console responses used to
open the reusable SSH control connection. The password is plaintext: keep
`config.toml` private and never commit a real credential.

The platform profiles are fixed configuration, not LLM output. `accelerator` is
the machine identity; `hardware_id` and `vendor` are not required. To add A100,
H100, or another JLSE target, add another `[[machine]]` table following the GH200
format and configure its queue, modules, environment, and monitoring settings.
The planner can then create runs for multiple configured accelerators, and the
executor submits each accelerator's bundle to its configured JLSE queue.

Dataset preparation and benchmark execution deliberately use separate Python
environments. `[dataset_runtime].python` runs on the login node and only needs
dataset dependencies such as NumPy and h5py. The `[[machine]]` modules and Conda
environment are loaded later by the scheduled benchmark job.

## 3. Start characterization

```bash
./agentic start --context "Characterize the production GPU workflow."
```

The graph analyzes the application, writes immutable artifacts under
`runs/<workflow-id>/`, and pauses at a human-review gate.

Check the current stage and locate its review file:

```bash
./agentic status
./agentic review-template
```

Open the reported YAML file. To approve it, set the following review fields:

```yaml
status: completed
decision: approve
reviewer: null
```

`reviewer` is optional. It may be a name or `null`.

Then resume:

```bash
./agentic resume
```

Characterization has two review gates: candidate inputs and the final
characterization. Repeat `status → edit review YAML → resume` for both. Use
`decision: reject` plus `feedback` to request a new version.

## 4. Plan the experiment

After characterization is approved:

```bash
./agentic plan
./agentic status
```

Review and approve the experiment plan using the same YAML procedure, then:

```bash
./agentic resume
```

For a one-point pipeline check instead of the full approved plan, use an
existing plan point and algorithm:

```bash
./agentic plan --smoke --algorithm pie --point-id p001
```

This creates a reviewed one-run plan revision; it does not execute the run.
Datasets are not generated locally. The next stage creates an
application-specific script which runs on the remote login node before qsub.

## 5. Generate and approve execution scripts

```bash
./agentic benchmark
./agentic status
```

The Execution Script Agent reads the application source, approved
characterization and plan, experiment matrix, fixed remote platform profile,
and JLSE RAG. It drafts:

```text
dataset_generation_script.vNNN.sh
benchmark_job_script.vNNN.sh
benchmark_run_manifest.vNNN.yaml
```

Inspect both scripts referenced by the manifest. The LLM determines the
application-specific dataset interface, application entry point, argument
mapping, timing extraction, and standardized result-writing logic. Platform
modules, the Conda environment, SSH/SCP/qsub, artifact hashes, and the result
contract remain fixed by code and configuration.

In short:

| LLM drafts                              | Fixed by the system                       |
| --------------------------------------- | ----------------------------------------- |
| Application dataset-generation commands | SSH/SCP and Cobalt submission             |
| Application entry point and CLI mapping | Queue/modules/Conda platform profile      |
| Per-run timing and result extraction    | Required result paths and CSV fields      |
| Application-specific validation         | Script safety checks and artifact SHA-256 |

Approve the reported review YAML and run:

```bash
./agentic resume
```

The graph now stops at `benchmark_ready`; neither dataset generation nor a GPU
job has run yet.
The old `./agentic datasets` command no longer exists: dataset preparation is
part of the reviewed remote-script contract.

## 6. Execute on JLSE

Run:

```bash
./agentic benchmark --execute
```

When `ssh_password` and `ssh_duo_choice` are configured, the launcher answers
the password and option prompts while opening the SSH control master. You still
need to approve the Duo request on your phone. Without those configured values,
run in an interactive terminal and answer the prompts manually.

The executor opens one SSH multiplexed connection and then automatically:

```text
build versioned bundle
→ SCP upload
→ verify bundle SHA-256
→ run approved dataset_generation.sh with dataset_runtime.python
→ validate the generated dataset manifest
→ qsub to gpu_gh200
→ qstat polling
→ SCP results back
→ measurement extraction
```

The bundle contains the approved plan/matrix, both approved scripts, application
source, and a machine-specific environment wrapper. Dataset generation happens
first on the login node. Only after it succeeds does the wrapper submit the
exact approved `benchmark_job.sh` to the queue. If the valid dataset already
exists, reuse depends on the logic in the human-approved
`dataset_generation.sh`; the executor itself invokes that script on every
execution attempt. Results are downloaded to:

```text
runs/<workflow-id>/remote_results/vNNN/results/
```

The graph rewrites remote log and power-trace paths to their downloaded local
paths, then pauses at `measurement_validation_review`.

Remote execution also writes these versioned workflow artifacts:

```text
raw_measurements.vNNN.csv
remote_execution_summary.vNNN.yaml
```

## 7. Validate measurements and fit the resource model

The remote run stops at `measurement_validation_review`. Inspect its validation
artifact, approve or reject the included measurements, and resume:

```bash
./agentic status
./agentic review-template
# Edit and save the reported review YAML.
./agentic resume
```

After measurement approval, the Modeling Agent fits latency, power, energy,
memory, and throughput targets. The graph then pauses at
`resource_model_review`. Inspect the fitted domain, groups, coefficients, and
validation warnings before approving it:

```bash
./agentic status
./agentic review-template
# Approve resource_model_review.
./agentic resume
```

Do not use a pipeline fixture as a scientific model. For example, five rows
duplicated from one real run can validate the software pipeline, but cannot
measure scaling. That condition is carried into the final deployment as
`scientific_use: false`.

## 8. Complete SystemFlow integration

After model approval, the SystemFlow Integration Agent drafts a declarative
mapping between application inputs, model group selectors, and SystemFlow
outputs. The graph pauses at `systemflow_mapping_review`.

Review the reported mapping artifact and approve it:

```bash
./agentic status
./agentic review-template
# Approve systemflow_mapping_review.
./agentic resume
```

The integration step then:

1. Converts the approved fitted model into the generic SystemFlow model JSON.
2. Loads `systemflow.application_models` from `[systemflow].path` when the
   native runtime is available.
3. Builds and executes validation `ExecutionGraph` instances.
4. Checks that every fitted group is covered and all predictions are positive.
5. Writes `systemflow_integration_report.vNNN.yaml` and pauses for final review.

The SystemFlow repository must provide this generic runtime before final
publication:

```text
<systemflow.path>/systemflow/application_models.py
```

Approve the final `systemflow_integration_review` and resume:

```bash
./agentic status
./agentic review-template
# Set status: completed, decision: approve, and reviewer: null if desired.
./agentic resume
```

Final approval triggers publication. The user does not manually copy model
files. The Integration Agent atomically publishes the approved assets to:

```text
<systemflow.path>/systemflow/application_model_data/<application-id>/
├── manifest.json
├── workflow_application_resource_model.vNNN.json
├── systemflow_application_mapping.vNNN.yaml
└── systemflow_integration_report.vNNN.yaml
```

`manifest.json` is the stable entry point. It contains the application ID,
runtime module, current versioned filenames, SHA-256 hashes, and the
`scientific_use` flag. Existing immutable versions are retained; if another
workflow produces different content with the same local version number, the
publisher adds a content-hash suffix and updates the manifest.

The workflow also records a local immutable
`systemflow_deployment_manifest.vNNN.yaml`. When publication completes:

```bash
./agentic status
```

reports:

```text
status: complete
stage: complete
next: END
```

`END` here means characterization, planning, benchmarking, modeling, mapping,
integration validation, human approval, and SystemFlow publication have all
finished. It does not mean the integration was skipped.

## 9. Load the published prediction model in SystemFlow

SystemFlow should resolve the active model and mapping through `manifest.json`,
instead of hard-coding a workflow run-directory path:

```python
import json
from pathlib import Path

deployment = Path(
    "/path/to/systemflow/systemflow/application_model_data/pty-chi"
)
manifest = json.loads((deployment / "manifest.json").read_text())
model_path = deployment / manifest["assets"]["model"]["path"]
mapping_path = deployment / manifest["assets"]["mapping"]["path"]

from systemflow.application_models import WorkflowApplicationResourceModel

model = WorkflowApplicationResourceModel(model_path)
estimate = model.predict(
    inputs={
        "scan_point_count": 64,
        "detector_shape": [64, 64],
        "num_epochs": 2,
        "batch_size": 1,
    },
    group_selectors={"accelerator": "GH200", "algorithm": "pie"},
)
print(estimate.predictions)
```

Before using a deployment for scientific decisions, require:

```python
if not manifest["scientific_use"]:
    raise RuntimeError("This deployment is only a pipeline-validation fixture")
```

See `systemflow_usage.md` for the complete direct-prediction and SystemFlow
`ExecutionGraph` examples.

## Recovery and useful commands

If a non-review node fails, correct the underlying problem and retry from the
checkpoint:

```bash
./agentic continue
```

Do not use `continue` while a review is pending; edit the review YAML and use
`resume` instead.

```bash
# Inspect workflow state
./agentic status

# Find the pending review
./agentic review-template

# Check whether a knowledge index needs rebuilding
./agentic rag-index status
./agentic jlse-rag status

# Query the operational knowledge manually
./agentic jlse-rag search \
  --query "GH200 module loading, Conda activation, and Cobalt qsub"
```
