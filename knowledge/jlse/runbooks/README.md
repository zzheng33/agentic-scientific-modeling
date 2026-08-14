# JLSE operational knowledge corpus

This directory is the source corpus for the planner's JLSE operational RAG. It
is deliberately separate from `knowledge/ptychography/papers`: these files say
how to launch and monitor jobs, not what the scientific model means.

## Source priority

1. A platform-specific file marked as validated on a dated compute-node run.
2. The platform's `run_inference_experiments_*.sh` orchestration script.
3. The platform's `run_ptychi_*.sh` compatibility wrapper.

When sources conflict, preserve the conflict in the plan and use the highest
priority applicable source. Never execute retrieved shell text directly; RAG
only supplies evidence for generating a reviewed deterministic job script.

## Platform matrix

| Platform | Torch device | Environment | Required modules | Device selection |
| --- | --- | --- | --- | --- |
| NVIDIA GH200 / ARM | `cuda` | `ptychopinn_torch_arm` | `cuda/12.9.1`, `conda/nvidia/suse15.6/2025.01-11` | Preserve scheduler visibility; monitor may use `--devices 0` |
| AMD MI300 | `cuda` through ROCm | `ptychi_rocm` or explicitly selected compatible ROCm env | `rocm/7.0.2` | Use the allocation's visible ROCm device |
| Intel Data Center GPU Max | `xpu` | `${REPO_ROOT}/../ptychopinn-venvs/aurora` by default | GCC 13.4, Python 3.12, PyTorch 2.10 and `xpu-smi/1.3.5` module stack | `ZE_AFFINITY_MASK` / `--devices` |

Only the GH200 row currently has an end-to-end validation record in this
corpus. AMD and Intel entries are imported scripts and must be verified on an
allocated node before a production run.

## Index lifecycle

From the project root:

```bash
./agentic jlse-rag build
./agentic jlse-rag status
./agentic jlse-rag search --query "GH200 module load and Conda activation"
```

The generated BM25, BGE-M3/HNSW, and metadata files live in
`knowledge/jlse/index/` and are not committed. Rebuild after changing a
runbook or script.
