# Build from the repository root: scripts/build_node.sh replay
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libegl1 libgl1 libglx-mesa0 libegl-mesa0 libgles2 libglvnd0 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY assets ./assets
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH MUJOCO_GL=egl
RUN uv sync --frozen --no-dev --extra sim --extra video
ENTRYPOINT ["ego2humanoid"]
