# Build from the repository root: scripts/build_node.sh tactile
# Run: docker run --gpus all -v $PWD:/app -v $PWD/data:/data -e HF_HOME=/hf -v <hf>:/hf \
#        ego2humanoid/tactile:dev python scripts/estimate_tactile_egotac.py <clip_dir>
# PYTHONPATH is deliberately unset: src/common would shadow EgoTac's namespace package `common`.
# Repo code (incl. third_party/EgoTac) and weights are mounted at runtime, never baked in.
FROM python:3.10-slim-bookworm
RUN pip install --no-cache-dir uv==0.8.22
WORKDIR /opt/node
COPY nodes/tactile/pyproject.toml nodes/tactile/uv.lock nodes/tactile/.python-version ./
# torch 2.5.1 PyPI wheels bundle CUDA 12.4 runtime libs (cu128 wheels need torch>=2.7, which EgoTac does not support).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH HF_HOME=/hf PYTHONUNBUFFERED=1
RUN uv sync --frozen --no-dev && rm -rf /root/.cache
WORKDIR /app
