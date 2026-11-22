# Devbox (cloud GPU VM)

Reference hardware on which the pipeline was developed and timed. Nothing here is required by the code; any Linux
host with an NVIDIA GPU and Docker works.

Recorded 2026-11-10 on host `devbox-a100`; development moved to `devbox-t4` on 2026-11-15.

| Item | `devbox-a100` | `devbox-t4` (current) |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | Ubuntu 22.04 |
| CPU / RAM | AMD EPYC 7V13, 24 vCPU, 216 GB | AMD EPYC 7V12, 8 vCPU, 54 GB |
| GPU | NVIDIA A100 80GB PCIe (driver 580.159.03) | Tesla T4 16 GB (driver 580.159.03) |
| Docker | installed; `docker run --gpus all` works | — |
| EGL | `libEGL.so.1`, `libEGL_nvidia`, `libEGL_mesa` present | MuJoCo EGL rendering works |
| Network | Hugging Face reachable | — |
| Local disks | `/` 119 GB (~15 GB free), `/tmp` 63 GB (~28 GB free) | `/` 119 GB (~8 GB free), `/tmp` 346 GB |
| Repo location | network file share | same share |

On the T4, use fp16/fp32 (no bf16 / FlashAttention-2) for the perception models. The robot
part (`retarget_estimated.py`, MuJoCo, EGL rendering) is CPU-bound; 4 clips in parallel fit on
8 vCPUs.

## Storage rule

The repo sits on the network file share. Small per-clip outputs stay in `data/` (git-ignored,
see `data/README.md`). Large, regenerable render intermediates go to
`/cloudfile/nvme_tmp/egodex/<clip>/panels` (`E2H_BULK_DIR`). Weights live under
`data/weights/` or `/mnt/nvme_hot`.

## Python environment

The uv venv is the repo's `.venv` (on the share on `devbox-t4`; a local-disk venv via
`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/e2h UV_LINK_MODE=copy` is faster where disk allows):

```bash
uv sync --frozen --extra video --extra sim --extra hands --extra objects --extra egodex
MUJOCO_GL=egl uv run --frozen --extra video --extra sim --extra hands --extra objects --extra egodex \
  python -m pytest tests -q
```

Tools present: `uv`, `ffmpeg`, `git-lfs`, `hf`, `docker`.

Per-stage Docker images (`nodes/{perception,hands,tactile,retarget}`; `scripts/build_node.sh <node>`) replace the
hand-made venvs with `DOCKER=1 scripts/run_pipeline.sh`; they need Docker with `--gpus all` (the A100 box).

## Switching GPU

Resize the VM in the cloud console (stop, resize, start); no repo change
is needed. Re-run `nvidia-smi` and update this file.
