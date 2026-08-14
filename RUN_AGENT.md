# Agent quick start

This guide runs the complete local-control/remote-execution workflow. Run all
commands from the repository root.

The supported execution path is remote-only:

```text
Characterization Agent
→ Planning Agent
→ Execution Script Agent
→ human script review
→ Remote Executor (SSH/SCP/Cobalt)
→ measurement validation and resource modeling
→ SystemFlow integration
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
the project-local Hugging Face cache. You therefore do not need to activate
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

[planner]
default_hardware = ["GH200"]

[remote_executor]
enabled = true
host = "zhong.zheng@login.jlse.anl.gov"
remote_runs_root = "/home/zhong.zheng/agentic-runs"
remote_application_path = "/home/zhong.zheng/pty-chi"
upload_application = true
hardware_id = "GH200"
queue = "gpu_gh200"
nodes = 1
walltime_minutes = 30
poll_interval_s = 15
poll_timeout_s = 3600
module_path = "/soft/modulefiles"
modules = ["cuda/12.9.1", "conda/nvidia/suse15.6/2025.01-11"]
conda_env = "ptychopinn_torch_arm"
remote_monitor_script = "/home/zhong.zheng/PtychoPINN/scripts/monitor_gpu_power.py"
device = "cuda"
vendor = "nvidia"
devices = "0"
power_interval_s = 0.2
continue_on_error = true
```

With `upload_application = true`, the executor packages the configured local
application source. Set it to false only when intentionally reusing the checkout
at `remote_application_path`. Passwords and Duo choices must not be added to
the configuration.

The platform profile is fixed configuration, not LLM output. For another GPU,
select its queue, module list, Conda environment, device, and power vendor in
`config.toml` before planning. The generated scripts can use those supplied
values but cannot load modules or activate a different environment themselves.
The approved plan must target exactly this `hardware_id`; use a separate
reviewed execution/profile for another GPU.

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
reviewer: Your Name
```

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

Datasets are not generated locally. The next stage generates an
application-specific dataset script that runs inside the compute allocation.

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

| LLM drafts | Fixed by the system |
| --- | --- |
| Application dataset-generation commands | SSH/SCP and Cobalt submission |
| Application entry point and CLI mapping | Queue/modules/Conda platform profile |
| Per-run timing and result extraction | Required result paths and CSV fields |
| Application-specific validation | Script safety checks and artifact SHA-256 |

Approve the reported review YAML and run:

```bash
./agentic resume
```

The graph now stops at `benchmark_ready`; no GPU job has been submitted yet.
The old `./agentic datasets` command no longer exists: dataset preparation is
part of the reviewed remote-script contract.

## 6. Execute on JLSE

Run this command in a real interactive terminal:

```bash
./agentic benchmark --execute
```

At the JLSE authentication prompts:

1. Enter your login credential at the first prompt.
2. Select the displayed Duo Push option (currently option `1`).
3. Approve the request on your phone.

The executor opens one SSH multiplexed connection and then automatically:

```text
build versioned bundle
→ SCP upload
→ verify bundle SHA-256
→ qsub to gpu_gh200
→ qstat polling
→ SCP results back
→ measurement extraction
```

The bundle contains the approved plan/matrix, both approved scripts, application
source, and a fixed GH200 environment wrapper. The wrapper runs the exact
approved `benchmark_job.sh`, which invokes the approved
`dataset_generation.sh` inside the allocation. Results are downloaded to:

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

## 7. Finish validation and modeling

Continue the same review cycle:

```bash
./agentic status
./agentic review-template
# Edit and save the reported review YAML.
./agentic resume
```

Later gates cover measurement validation, fitted resource models, SystemFlow
mapping, and integration. Repeat until `./agentic status` reports `complete`.

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
