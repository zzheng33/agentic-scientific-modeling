# JLSE GH200 execution environment

This is operational infrastructure guidance, not scientific-paper evidence.
It was validated for Cobalt jobs submitted through `login.jlse.anl.gov` to the
`gpu_gh200` queue. Run it inside the compute-job shell before importing PyTorch
or launching PtyChi.

Validation status: end-to-end smoke run completed on `grace00` on 2026-08-14
with PyTorch `2.12.0+cu130`, CUDA available, and device
`NVIDIA GH200 480GB`. A separate allocation on `grace01` reported the device
busy, so every job must retain the tensor-allocation preflight rather than
assuming that module loading alone proves GPU usability.

```bash
MODULE_PATH="${MODULE_PATH:-/soft/modulefiles}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.1}"
CONDA_MODULE="${CONDA_MODULE:-conda/nvidia/suse15.6/2025.01-11}"
export CONDA_ENV="${CONDA_ENV:-ptychopinn_torch_arm}"

set +u
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use "${MODULE_PATH}"
  module load "${CUDA_MODULE}"
  module load "${CONDA_MODULE}"
  hash -r
fi
MODULE_PYTHON="$(command -v python)"
CONDA_BASE="$(dirname "$(dirname "$MODULE_PYTHON")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
hash -r
set -u
```

After loading the ARM Conda module, source that module's `conda.sh`, activate
`CONDA_ENV`, and resolve Python with `command -v python`. Do not source the x86
login-node Miniforge activation script in a GH200 ARM job. The application
source is under `/home/zhong.zheng/pty-chi`; when it is not installed into the
selected environment, prepend `/home/zhong.zheng/pty-chi/src` to `PYTHONPATH`.
Do not force `CUDA_VISIBLE_DEVICES=0`; preserve the device visibility supplied
by the Cobalt allocation.

Run this preflight inside the allocated node:

```bash
python -c 'import torch, ptychi.api; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Scheduler facts observed on 2026-08-14:

- scheduler CLI: Cobalt `qsub.py`, not PBS Pro;
- GH200 queue: `gpu_gh200`;
- submit one-node script jobs with `qsub -n 1 -t MINUTES -q gpu_gh200 --mode script ...`;
- Argo API is not available from compute execution, so remote jobs must be
  deterministic scripts produced and reviewed by the local control plane.
