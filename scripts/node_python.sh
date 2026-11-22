#!/usr/bin/env bash
# Run `python <args>` inside a node image. Invoked through the nodes/bin/<node>-python symlinks, which
# scripts/run_pipeline.sh uses as PERC_PY / FAH_PY / WILOR_PY / EGOTAC_PY / BPY when DOCKER=1.
# Repo, data and weights are mounted at their host paths, so relative and absolute paths both resolve.
#   IMAGE_TAG [dev]   HF_HOME [/mnt/tmp/hfcache]   FAH_SRC [<repo>/third_party/FAH]   WILOR_SRC [<repo>/third_party/WiLoR]
#   EGOTAC_SRC [<repo>/third_party/EgoTac]   MANO_SRC [<repo>/assets/mano/clean for hands]
#   FAH_WEIGHTS_SRC [<repo>/data/weights/fah, hands only; mounted at /fah_weights]   DOCKER_ARGS [extra docker run flags]
set -euo pipefail
node=$(basename "$0"); node=${node%-python}
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
tag=${IMAGE_TAG:-dev}
args=(run --rm --gpus all -u "$(id -u):$(id -g)" -w "$ROOT" -e HOME=/tmp -e USER=e2h -e LOGNAME=e2h -e "PYTHONPATH=$ROOT/src"
  -e "HF_HOME=${HF_HOME:-/mnt/tmp/hfcache}" -v "$ROOT:$ROOT" -v "${HF_HOME:-/mnt/tmp/hfcache}:${HF_HOME:-/mnt/tmp/hfcache}")
for d in "$ROOT"/data/*; do [ -L "$d" ] && args+=(-v "$(readlink -f "$d"):$(readlink -f "$d")"); done
# Forward the pipeline's tuning variables (retarget_estimated.py reads many by name).
while IFS= read -r e; do args+=(-e "$e"); done < <(compgen -e | grep -E '^(E2H_|SIM_|SCENE|HAND$|RENDER|MUJOCO_GL$|PYOPENGL_PLATFORM$|PYTHONUNBUFFERED$|OBJ_|HY3DGEN|FAH_|WILOR_)')
case "$node" in
  hands) image=hands
    args+=(-e "PYTHONPATH=$ROOT/src:/third/FAH" -e FAH_WEIGHTS=/fah_weights -v "${FAH_WEIGHTS_SRC:-$ROOT/data/weights/fah}:/fah_weights:ro" -v "${FAH_SRC:-$ROOT/third_party/FAH}:/third/FAH:ro")
    [ -z "${MANO_SRC:-}" ] && [ -d "$ROOT/assets/mano/clean" ] && MANO_SRC=$ROOT/assets/mano/clean ;;
  hands-wilor) image=hands-wilor
    args+=(-v "${WILOR_SRC:-$ROOT/third_party/WiLoR}:/third/WiLoR:ro") ;;
  retarget) image=retarget
    for d in "$ROOT"/third_party/*; do [ -L "$d" ] && args+=(-v "$(readlink -f "$d"):$(readlink -f "$d"):ro"); done ;;
  tactile) image=tactile
    # src/common would shadow EgoTac's own `common` package, so PYTHONPATH must stay empty
    args+=(-e PYTHONPATH= -v "${EGOTAC_SRC:-$ROOT/third_party/EgoTac}:$ROOT/third_party/EgoTac:ro") ;;
  *) image=$node ;;
esac
[ -n "${MANO_SRC:-}" ] && args+=(-v "$MANO_SRC:/mano:ro")
# shellcheck disable=SC2086
exec docker "${args[@]}" ${DOCKER_ARGS:-} --entrypoint python "ego2humanoid/$image:$tag" "$@"
