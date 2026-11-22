#!/usr/bin/env bash
# Pure-RGB hand evaluation over fetched EgoDex clips: export -> MoGe-2 focal -> FAH -> metrics.
# Needs: uv env for ego2humanoid (UV_PROJECT_ENVIRONMENT) and FAH venv (FAH_PY, default /tmp/venvs/fah/bin/python).
set -euo pipefail
FAH_PY=${FAH_PY:-/tmp/venvs/fah/bin/python}
export FAH_HOME=${FAH_HOME:-/tmp/fah_home} FAH_SAM3D=${FAH_SAM3D:-/tmp/fah_home/sam-3d-body-dinov3} MANO_DIR=${MANO_DIR:-/tmp/fah_home/mano}
for h5 in $(find data/raw/egodex -name "*.hdf5" | sort); do
  uv run python -c "
import sys
from egodex.clip_reader import load_clip
from egodex.export import export_clip
print(export_clip(load_clip(sys.argv[1]), 'data/derived/egodex'))" "$h5" > /tmp/clipdir.txt
  clip=$(tail -1 /tmp/clipdir.txt)
  [ -f "$clip/retarget_rgb.json" ] && continue
  f=$($FAH_PY scripts/estimate_intrinsics_moge.py "$clip" 2>/dev/null | tail -1)
  $FAH_PY scripts/estimate_gravity_moge.py "$clip" > /dev/null 2>&1
  $FAH_PY scripts/estimate_hands_fah.py "$clip" --focal "$f" >/dev/null 2>&1
  uv run python scripts/eval_hands_vs_egodex.py "$clip" > /dev/null
  cp "$clip/hand_eval.json" "$clip/hand_eval_rgb.json"
  uv run python scripts/retarget_estimated.py "$clip" > /dev/null
  echo "$clip f=$f"
done
