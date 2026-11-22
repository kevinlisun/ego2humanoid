#!/usr/bin/env bash
# One-command real2sim data pipeline: EgoDex clip -> perception -> retarget + MuJoCo rollout -> review video.
#
# Usage: scripts/run_pipeline.sh <clip_id>...        e.g. scripts/run_pipeline.sh 0 1 10 100
# Every stage is skipped when its output already exists (FORCE=1 reruns them all). See run_pipeline.md.
#
# Environment (defaults in brackets):
#   HAND [sharpa]            dex3 | sharpa
#   RENDER [1]               1 renders review/result_grid_<hand>.mp4
#   JOBS [1]                 clips processed in parallel (4 on the T4 box)
#   STAGES [all]             comma list out of: export,intrinsics,gravity,hands,tactile,objects,scene,
#                            scene_mesh,held_mesh,retarget
#   FORCE [0]                1 reruns stages whose outputs exist
#   DRY_RUN [0]              1 prints the commands without running them
#   TASK [basic_pick_place]  EgoDex task folder
#   RAW_DIR [data/raw/egodex/test]    DERIVED_DIR [data/derived/egodex]    LOG_DIR [/tmp/e2h_pipeline]
#   PERC_PY [/tmp/venvs/perc/bin/python]  MoGe, SAM 3 / OWLv2, Hunyuan3D (torch, moge, transformers>=5, hy3dgen)
#   FAH_PY [/tmp/venvs/fah/bin/python]    FAH hand pose (also needs FAH_HOME, FAH_SAM3D, MANO_DIR)
#   WILOR_PY [/tmp/venvs/wilor/bin/python]  WiLoR hand pose, used when FAH_PY does not exist (WILOR_REPO checkout)
#   EGOTAC_PY [unset]        EgoTac venv python; the tactile stage is skipped when unset
#   HF_HOME [unset]          Hugging Face cache holding facebook/sam3 and the MoGe weights; exported to every
#                            stage (Docker: -v $HF_HOME:$HF_HOME -e HF_HOME); SAM3 stages fail fast without it
#   DOCKER [0]               1 runs every stage in its own node image (nodes/<node>, scripts/build_node.sh) via
#                            nodes/bin/<node>-python; sets the *_PY variables and UV below
#   OBJ_DETECTOR [sam3]      sam3 | owl (owl is the public-weights fallback; sam3 falls back to owl on error)
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# The perception venvs do not have the project installed; scripts import it from src/.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

HAND=${HAND:-sharpa}
RENDER=${RENDER:-1}
JOBS=${JOBS:-1}
STAGES=${STAGES:-export,intrinsics,gravity,hands,tactile,objects,scene,scene_mesh,held_mesh,retarget}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
TASK=${TASK:-basic_pick_place}
RAW_DIR=${RAW_DIR:-data/raw/egodex/test}
DERIVED_DIR=${DERIVED_DIR:-data/derived/egodex}
LOG_DIR=${LOG_DIR:-/tmp/e2h_pipeline}
PERC_PY=${PERC_PY:-/tmp/venvs/perc/bin/python}
FAH_PY_USER=${FAH_PY:-}
FAH_PY=${FAH_PY:-/tmp/venvs/fah/bin/python}
WILOR_PY=${WILOR_PY:-/tmp/venvs/wilor/bin/python}
EGOTAC_PY=${EGOTAC_PY:-}
OBJ_DETECTOR=${OBJ_DETECTOR:-sam3}
UV=(uv run --frozen --extra video --extra sim --extra hands --extra objects --extra egodex python)
if [ "${DOCKER:-0}" = 1 ]; then
  PERC_PY=$ROOT/nodes/bin/perception-python
  FAH_PY=${FAH_PY_USER:-$ROOT/nodes/bin/hands-python}
  WILOR_PY=$ROOT/nodes/bin/hands-wilor-python
  EGOTAC_PY=$ROOT/nodes/bin/tactile-python
  UV=("$ROOT/nodes/bin/retarget-python")
fi
SUFFIX=""
[ "$HAND" != dex3 ] && SUFFIX="_$HAND"
[ -n "${HF_HOME:-}" ] && export HF_HOME
export HAND RENDER STAGES FORCE DRY_RUN TASK RAW_DIR DERIVED_DIR LOG_DIR PERC_PY FAH_PY WILOR_PY EGOTAC_PY \
  OBJ_DETECTOR SUFFIX ROOT

if [ $# -eq 0 ]; then
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
mkdir -p "$LOG_DIR"

want() { [[ ",$STAGES," == *",$1,"* ]]; }
# todo <output>: true when the stage must run (output missing or FORCE=1)
todo() { [ "$FORCE" = 1 ] || [ ! -e "$1" ]; }
run() {
  echo "  \$ $*"
  [ "$DRY_RUN" = 1 ] && return 0
  "$@"
}

# sam3_weights: preflight of the SAM3 cache; a clear WEIGHTS_NOT_FOUND error instead of "SAM3 missing" later
sam3_weights() { [ "$DRY_RUN" = 1 ] || python3 -m stages.reconstruction.objects.hf_weights facebook/sam3; }

clip_pipeline() {
  local id=$1 clip="$DERIVED_DIR/${TASK}_$1" h5="$RAW_DIR/$TASK/$1.hdf5" f prompt
  echo "== clip $id -> $clip"

  if want export && todo "$clip/episode.json"; then
    [ -f "$h5" ] || { echo "  missing $h5: run scripts/fetch_egodex.py first"; return 1; }
    run "${UV[@]}" -c "
import sys
from egodex.clip_reader import load_clip
from egodex.export import export_clip
print(export_clip(load_clip(sys.argv[1]), sys.argv[2]))" "$h5" "$DERIVED_DIR" || return 1
  fi
  if want intrinsics && todo "$clip/intrinsics_est.json"; then
    run "$PERC_PY" scripts/estimate_intrinsics_moge.py "$clip" || return 1
  fi
  if want gravity && todo "$clip/gravity_est.json"; then
    run "$PERC_PY" scripts/estimate_gravity_moge.py "$clip" || return 1
  fi
  f=$(python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))['focal_px']))" \
    "$clip/intrinsics_est.json" 2>/dev/null || echo "<focal>")
  if want hands && todo "$clip/hands_fah_f$f.npz"; then
    if [ -x "$FAH_PY" ]; then
      run "$FAH_PY" scripts/estimate_hands_fah.py "$clip" --focal "$f" || return 1
    else
      # FAH needs gated weights: fall back to WiLoR (public weights, same output format)
      run env PYOPENGL_PLATFORM=egl "$WILOR_PY" scripts/estimate_hands_wilor.py "$clip" --focal "$f" || return 1
    fi
  fi
  if want tactile && [ -n "$EGOTAC_PY" ] && todo "$clip/tactile_egotac.npz"; then
    run "$EGOTAC_PY" scripts/estimate_tactile_egotac.py "$clip" || return 1
  fi
  prompt=$(PYTHONPATH=src python3 -c "
import json, sys
from stages.reconstruction.objects.prompt import object_prompt
print(object_prompt(json.load(open(sys.argv[1]))['task_description']))" "$clip/episode.json" 2>/dev/null \
    || echo object)
  if want objects && todo "$clip/objects_sam3.npz"; then
    echo "  object prompt: '$prompt'"
    # without SAM3 weights the sam3 command below fails and falls back to OWLv2
    [ "$OBJ_DETECTOR" = sam3 ] && { sam3_weights || echo "  (falling back to OWLv2)"; }
    if [ "$OBJ_DETECTOR" = sam3 ]; then
      run "$PERC_PY" scripts/perceive_objects_sam3.py "$clip" "$prompt" 3 0.1 \
        || run "$PERC_PY" scripts/perceive_objects_owl.py "$clip" "$prompt" || return 1
    else
      run "$PERC_PY" scripts/perceive_objects_owl.py "$clip" "$prompt" || return 1
    fi
  fi
  if want scene && todo "$clip/scene/inventory.json"; then
    run "$PERC_PY" scripts/perceive_scene_objects.py "$clip" || return 1
  fi
  if want scene_mesh && { [ "$FORCE" = 1 ] || ! ls "$clip"/scene/mesh_*.glb >/dev/null 2>&1; }; then
    run "$PERC_PY" scripts/generate_object_mesh.py "$clip" --scene || return 1
  fi
  # The held-object crop picks its frames from a first retarget pass (perceived_object.views).
  if want held_mesh && todo "$clip/object_mesh_raw.glb"; then
    sam3_weights || return 1
    if todo "$clip/retarget_rgb$SUFFIX.json"; then
      run env RENDER=0 MUJOCO_GL=egl "${UV[@]}" scripts/retarget_estimated.py "$clip" >/dev/null || return 1
    fi
    run "$PERC_PY" scripts/crop_object_sam3.py "$clip" "$prompt" || return 1
    run "$PERC_PY" scripts/generate_object_mesh.py "$clip" || return 1
  fi
  if want retarget; then
    local scene_env=()
    [ -d "data/scenes/egodex_$id" ] && scene_env=(SCENE_DIR="data/scenes/egodex_$id")
    run env PYTHONUNBUFFERED=1 MUJOCO_GL=egl "${scene_env[@]}" "${UV[@]}" scripts/retarget_estimated.py "$clip" \
      || return 1
    [ "$DRY_RUN" = 1 ] || python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
e = (d.get('perceived_object') or {}).get('sim_events') or d.get('proxy_events') or {}
print('  result:', {k: e.get(k) for k in ('success', 'failure_reasons', 'place_error_m')})" \
      "$clip/retarget_rgb$SUFFIX.json"
  fi
  echo "== clip $id done"
}
export -f want todo run sam3_weights clip_pipeline
export UV_STR="${UV[*]}"

status=0
if [ "$JOBS" -le 1 ]; then
  for id in "$@"; do
    clip_pipeline "$id" 2>&1 | tee "$LOG_DIR/${HAND}_$id.log"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "!! clip $id failed (log $LOG_DIR/${HAND}_$id.log)"; status=1; }
  done
else
  printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c '
    UV=($UV_STR); clip_pipeline {} > "$LOG_DIR/${HAND}_{}.log" 2>&1 \
      && echo "clip {} ok" || { echo "!! clip {} failed (log $LOG_DIR/${HAND}_{}.log)"; exit 1; }' \
    || status=1
fi
if [ "$DRY_RUN" != 1 ] \
  && uv run --frozen python scripts/summarize_batch.py "$DERIVED_DIR" "$LOG_DIR/batch_results.md" >/dev/null 2>&1; then
  echo "summary: $LOG_DIR/batch_results.md"
fi
exit $status
