#!/usr/bin/env bash
# Recreate the ephemeral NVMe hot disk and rebuild the SimFoundry conda envs on it (T6S.2a).
# The disk (/dev/nvme0n1) is wiped on every VM stop/start; envs and weights must live there
# because the CIFS share has ~50 ms small-file latency and the root disk is nearly full.
#
# Usage: scripts/rebuild_hot_disk.sh [envs...]   (default: simfoundry hunyuan any6d da3)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOT=/mnt/nvme_hot
ENVS=("${@:-simfoundry hunyuan any6d da3}")
read -r -a ENVS <<<"${ENVS[*]}"

if ! mountpoint -q "$HOT"; then
  if ! sudo blkid /dev/nvme0n1 >/dev/null 2>&1; then
    sudo mkfs.ext4 -q -L nvme_hot /dev/nvme0n1
  fi
  sudo mkdir -p "$HOT"
  sudo mount -o noatime /dev/nvme0n1 "$HOT"
  sudo chown "$(id -u):$(id -g)" "$HOT"
fi
mkdir -p "$HOT"/{envs/conda_envs,pkgs,weights,tmp,cache/pip,cache/uv,hf}

# Keep every cache off the root disk.
export TMPDIR=$HOT/tmp PIP_CACHE_DIR=$HOT/cache/pip UV_CACHE_DIR=$HOT/cache/uv
export HF_HOME=$HOT/hf HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export CONDA_ENVS_DIRS=$HOT/envs/conda_envs CONDA_PKGS_DIRS=$HOT/pkgs
export TORCH_CUDA_ARCH_LIST=8.0 # A100
export PATH="$ROOT/data/envs/miniforge/bin:$PATH"

ln -sfn "$HOT/envs/conda_envs" "$ROOT/data/envs/conda_envs"
git config --global --add safe.directory '*'

cd "$ROOT/third_party/SimFoundry"
for e in "${ENVS[@]}"; do
  echo "=== building $e ($(date +%T)) ==="
  bash scripts/installation/install_everything.sh --only "$e"
done
echo "=== done ($(date +%T)) ==="
