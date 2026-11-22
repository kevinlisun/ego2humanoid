# Stages export + retarget. Build from the repo root: scripts/build_node.sh retarget
#   docker run --rm --gpus all -v $PWD:/app -v <resolved data/derived>:/app/data/derived ego2humanoid/retarget:dev \
#       python scripts/retarget_estimated.py <clip>
# Two venvs: /opt/venv (py3.12, root uv.lock, source-built MuJoCo) and /opt/bpy (py3.11, bpy 5.0.1 for rendering).
ARG BASE=ubuntu:22.04

# Overridable with `--build-context mujoco-src=<dir>` when third_party/mujoco is made of symlinks (Docker does not follow them).
FROM scratch AS mujoco-src
COPY third_party/mujoco /

FROM ${BASE} AS mujoco-build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl \
      libgl-dev libegl-dev libglx-dev libopengl-dev \
      libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
COPY --from=mujoco-src / /src/mujoco
WORKDIR /src/mujoco
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/mujoco \
      -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
    && cmake --build /build -j3 && cmake --install /build
RUN uv venv --seed --python 3.12 /opt/buildvenv
ENV VIRTUAL_ENV=/opt/buildvenv PATH=/opt/buildvenv/bin:$PATH
WORKDIR /src/mujoco/python
RUN bash make_sdist.sh \
    && MUJOCO_PATH=/opt/mujoco MUJOCO_PLUGIN_PATH=/src/mujoco/plugin \
       MUJOCO_CMAKE_ARGS="-DGLFW_BUILD_WAYLAND=OFF" \
       pip wheel --no-deps -w /wheels dist/mujoco_core-*.tar.gz

FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git ffmpeg libegl1 libgl1 libglu1-mesa libosmesa6 libglib2.0-0 \
      libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
COPY --from=mujoco-build /wheels /wheels

ENV UV_LINK_MODE=copy UV_CACHE_DIR=/opt/uv-cache UV_PYTHON_INSTALL_DIR=/opt/uv-python
WORKDIR /app
# Main venv: root pyproject/uv.lock (no node-specific lock needed), then swap in the source-built MuJoCo.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --python 3.12 --group dev \
      --extra video --extra sim --extra hands --extra objects --extra egodex \
    && uv pip install --python /opt/venv/bin/python --reinstall --no-deps /wheels/mujoco_core-*.whl \
    && rm -rf /wheels
# Render venv for Blender (bpy needs python 3.11 + numpy 1.x).
COPY nodes/retarget/bpy-requirements.txt /tmp/bpy-requirements.txt
RUN uv venv --python 3.11 /opt/bpy \
    && uv pip install --python /opt/bpy/bin/python -r /tmp/bpy-requirements.txt \
    && rm -rf /opt/uv-cache /tmp/bpy-requirements.txt

ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src MUJOCO_GL=egl HF_HOME=/hf \
    BPY_PY=/opt/bpy/bin/python BPY_PYTHON=/opt/bpy/bin/python
COPY configs ./configs
COPY assets ./assets
CMD ["bash"]
