# Reproducible environment: system libs + MuJoCo built from the pinned submodule (third_party/mujoco),
# Python dependencies managed only by uv (pyproject.toml + uv.lock).
#
#   docker build -t ego2humanoid .
#   docker run --rm --gpus all -v $PWD:/work -w /work ego2humanoid uv run --no-sync pytest tests/unit
#
# After editing third_party/mujoco (e.g. by RSI), rebuild the image; only the mujoco stages re-run.

ARG BASE=ubuntu:22.04

# ---- Stage 1: MuJoCo C library + Python bindings wheel built from source ----
FROM ${BASE} AS mujoco-build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl \
      libgl-dev libegl-dev libglx-dev libopengl-dev \
      libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY third_party/mujoco /src/mujoco
WORKDIR /src/mujoco
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/mujoco \
      -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
    && cmake --build /build -j"$(nproc)" && cmake --install /build

# make_sdist.sh pip-installs pinned build tools into the active venv, so give it a throw-away one.
RUN uv venv --seed --python 3.12 /opt/buildvenv
ENV VIRTUAL_ENV=/opt/buildvenv PATH=/opt/buildvenv/bin:$PATH
WORKDIR /src/mujoco/python
RUN bash make_sdist.sh \
    && MUJOCO_PATH=/opt/mujoco MUJOCO_PLUGIN_PATH=/src/mujoco/plugin \
       MUJOCO_CMAKE_ARGS="-DGLFW_BUILD_WAYLAND=OFF" \
       pip wheel --no-deps -w /wheels dist/mujoco_core-*.tar.gz

# ---- Stage 2: runtime ----
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git libegl1 libgl1 libglu1-mesa libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=mujoco-build /wheels /wheels

ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON=3.12 MUJOCO_GL=egl PATH=/opt/venv/bin:$PATH
WORKDIR /work
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Python deps come from uv.lock; the PyPI mujoco-core is then replaced by the source-built wheel.
RUN uv sync --frozen --group dev --extra video --extra sim --extra hands --extra objects --extra egodex \
    && uv pip install --python /opt/venv/bin/python --reinstall --no-deps /wheels/mujoco_core-*.whl \
    && python -c "import mujoco; print('mujoco', mujoco.__version__)"
CMD ["bash"]
