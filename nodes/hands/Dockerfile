# Hand-pose stage. Build from the repo root:
#   docker build -f nodes/hands/Dockerfile -t ego2humanoid/hands:dev .                 (FAH, default)
#   docker build -f nodes/hands/Dockerfile --target wilor -t ego2humanoid/hands-wilor:dev .  (WiLoR fallback)
# Restricted FAH/WiLoR code, MANO files and weights are NOT baked in; mount them at runtime:
#   docker run --gpus all -v $PWD/data:/data -v <FAH>:/third/FAH:ro -v <MANO>:/mano:ro ... \
#       python scripts/estimate_hands_fah.py <clip_dir> --focal F      (FAH_HOME=/third/FAH, MANO_DIR=/mano)
#   WiLoR: -v <WiLoR>:/third/WiLoR (WILOR_REPO), python scripts/estimate_hands_wilor.py <clip_dir> --focal F
FROM python:3.11-slim-bookworm AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 libglx-mesa0 libegl-mesa0 libgles2 libglvnd0 libosmesa6 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv==0.9.5
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0 UV_NO_CACHE=1 \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 UV_INDEX_STRATEGY=unsafe-best-match \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf PYTHONPATH=/app/src PYTHONUNBUFFERED=1
RUN uv venv /opt/venv --python 3.11
WORKDIR /app

FROM base AS wilor
COPY nodes/hands/requirements-wilor.txt /tmp/req.txt
# chumpy's sdist imports pip at build time, so it is built without isolation
RUN grep -v '^chumpy' /tmp/req.txt > /tmp/req-main.txt \
    && uv pip install -r /tmp/req-main.txt \
    && uv pip install pip setuptools wheel \
    && uv pip install --no-build-isolation chumpy==0.70 \
    && rm -f /tmp/req*.txt
# py3.11/numpy1.26 shims needed to unpickle MANO (chumpy)
COPY nodes/hands/sitecustomize-wilor.py /opt/venv/lib/python3.11/site-packages/sitecustomize.py
COPY src ./src
COPY scripts ./scripts
ENV WILOR_REPO=/third/WiLoR PYOPENGL_PLATFORM=egl

# Last stage = default target
FROM base AS fah
COPY nodes/hands/requirements-fah.txt /tmp/req.txt
RUN uv pip install -r /tmp/req.txt && rm -f /tmp/req.txt
COPY src ./src
COPY scripts ./scripts
# FAH was installed editable in the working venv; the mounted checkout is put on PYTHONPATH instead
ENV FAH_HOME=/third/FAH MANO_DIR=/mano PYTHONPATH=/app/src:/third/FAH
