#!/usr/bin/env bash
# Per clip: SAM 3 + MoGe object perception (FAH venv) -> perceived-object retarget/replay (main venv).
# Usage: run_object_eval_batch.sh <clip_dir>...   (needs HF_TOKEN, HF_HOME on a disk with space)
set -euo pipefail
for clip in "$@"; do
  prompt=$(uv run python -c "
import json,sys
from stages.reconstruction.objects.prompt import object_prompt
print(object_prompt(json.load(open('$clip/episode.json'))['task_description']))")
  echo "== $clip: '$prompt'"
  /tmp/venvs/fah/bin/python scripts/perceive_objects_sam3.py "$clip" "$prompt" 3 0.1 2>&1 | tail -1
  uv run python scripts/retarget_estimated.py "$clip" 2>&1 | tail -1 | python3 -c "
import sys,json
d=json.loads(sys.stdin.read()); p=d.get('perceived_object')
print('perceived:', None if p is None else {k:p[k] for k in ('n_matches','hold_s','observed_lift_m','contact_steps')}, None if p is None else {k:p['sim_events'][k] for k in ('success','lift_height_m','translation_m','failure_reasons')})"
done
