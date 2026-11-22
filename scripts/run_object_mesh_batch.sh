#!/usr/bin/env bash
# Per clip: crop the held object (SAM 3) -> Hunyuan3D mesh (FAH venv) -> mesh-object retarget/replay + render.
# Needs objects_sam3.npz and retarget_rgb.json (run run_object_eval_batch.sh first).
# Usage: run_object_mesh_batch.sh <clip_dir>...   (needs HF_TOKEN, HF_HOME; weights in data/weights/hunyuan3d)
set -uo pipefail
for clip in "$@"; do
  prompt=$(uv run python -c "
import json
from stages.reconstruction.objects.prompt import object_prompt
print(object_prompt(json.load(open('$clip/episode.json'))['task_description']))")
  echo "== $clip: '$prompt'"
  /tmp/venvs/fah/bin/python scripts/crop_object_sam3.py "$clip" "$prompt" 2>&1 | tail -1
  /tmp/venvs/fah/bin/python scripts/generate_object_mesh.py "$clip" 2>&1 | tail -1 | cut -c1-200
  RENDER=1 MUJOCO_GL=egl uv run --extra objects python scripts/retarget_estimated.py "$clip" 2>&1 | tail -1 | python3 -c "
import sys,json
d=json.loads(sys.stdin.read()); p=d.get('perceived_object')
print('mesh:', None if p is None else {k:p[k] for k in ('mesh_asset','physics','contact_steps')}, None if p is None else {k:p['sim_events'][k] for k in ('success','lift_height_m','translation_m')})"
  MUJOCO_GL=egl uv run --extra objects python scripts/preview_object_mesh.py "$clip" 2>&1 | tail -1
done
