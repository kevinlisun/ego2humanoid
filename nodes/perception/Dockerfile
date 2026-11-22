# Build from the repository root: scripts/build_node.sh perception
# Run: docker run --gpus all -v $PWD:/app -v $PWD/data:/data -v $HF_HOME:/hf ego2humanoid/perception:dev python scripts/<script>.py ...
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl libgl1 libglib2.0-0 libegl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY nodes/perception/pyproject.toml nodes/perception/uv.lock nodes/perception/.python-version ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=0 PATH=/opt/venv/bin:$PATH
RUN uv sync --frozen --no-dev --no-install-project && uv cache clean
# pymeshlab plugins need libOpenGL; kept after the venv layer to preserve the build cache.
RUN apt-get update && apt-get install -y --no-install-recommends libopengl0 && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/app/src HF_HOME=/hf PYTHONUNBUFFERED=1
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
