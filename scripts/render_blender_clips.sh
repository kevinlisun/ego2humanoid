#!/usr/bin/env bash
# Blender (Cycles, GPU) render of the MuJoCo rollouts from the robot head camera (Track BT, BT.17).
#
# Usage: scripts/render_blender_clips.sh <clip_id>...     e.g. scripts/render_blender_clips.sh 0 1 10
# Needs the retarget outputs tagged $TAG (review/sim_state_sharpa$TAG.npz, review/trajectory_sharpa$TAG.npz);
# a clip is waited for up to WAIT_S seconds. Writes review/twin_sim$TAG.mp4.
#
# Environment: TAG [_bt]   DERIVED_DIR [data/derived/egodex]   TASK [basic_pick_place]   FPS [10]
#   BPY_PY [/tmp/bpy_venv/bin/python]   WAIT_S [0]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:-_bt}
DERIVED_DIR=${DERIVED_DIR:-data/derived/egodex}
TASK=${TASK:-basic_pick_place}
FPS=${FPS:-10}
BPY_PY=${BPY_PY:-/tmp/bpy_venv/bin/python}
WAIT_S=${WAIT_S:-0}
UV=(uv run --frozen --extra video --extra sim --extra hands --extra objects --extra egodex python)

for id in "$@"; do
  clip="$DERIVED_DIR/${TASK}_$id"
  waited=0
  until [ -e "$clip/review/trajectory_sharpa$TAG.npz" ] && [ -e "$clip/retarget_rgb_sharpa$TAG.json" ]; do
    [ "$waited" -ge "$WAIT_S" ] && { echo "!! clip $id: no sharpa$TAG outputs"; continue 2; }
    sleep 30
    waited=$((waited + 30))
  done
  echo "== clip $id (Blender)"
  "${UV[@]}" scripts/export_twin_scene.py "$clip" "$TAG" > /dev/null || { echo "!! clip $id export failed"; continue; }
  TMPDIR=/tmp "$BPY_PY" scripts/render_twin_sim.py "$clip" --tag "$TAG" --fps "$FPS" --device gpu \
    | tail -1 || echo "!! clip $id render failed"
done
