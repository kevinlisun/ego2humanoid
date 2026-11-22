#!/usr/bin/env bash
# One-shot setup for the `basic_pick_place_0` tutorial: submodules, EgoDex clips, robot models, weights, node images.
# Idempotent: finished steps are skipped. Usage: scripts/setup_usecase0.sh [--no-build]
# Manual prerequisites (cannot be automated): accept the facebook/sam3 licence and `hf auth login`;
# put MANO_RIGHT.pkl (https://mano.is.tue.mpg.de) in third_party/WiLoR/mano_data/.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export HF_HOME=${HF_HOME:-/mnt/tmp/hfcache}
mkdir -p "$HF_HOME"

echo "== submodules"
git submodule update --init --depth 1 third_party/EgoTac third_party/WiLoR

echo "== EgoDex clips (CC-BY-NC-ND, never redistribute)"
[ -f data/raw/egodex/test/basic_pick_place/0.mp4 ] ||
  uv run --frozen --extra egodex python scripts/fetch_egodex.py basic_pick_place -n 12

echo "== robot models"
[ -d assets/robots/g1_dex3/meshes ] || bash scripts/fetch_g1_assets.sh
[ -d assets/robots/sharpa_wave/meshes ] || bash scripts/fetch_sharpa_assets.sh

echo "== weights"
H3D=data/weights/hunyuan3d/Hunyuan3D-2mini
[ -f $H3D/hunyuan3d-dit-v2-mini/model.fp16.safetensors ] ||
  hf download tencent/Hunyuan3D-2mini --local-dir $H3D \
    --include "config.json" "hunyuan3d-dit-v2-mini/*" "hunyuan3d-vae-v2-mini/*"
[ -f data/weights/egotac/egotac_base_h4.pt ] ||
  hf download wkzhang/EgoTac egotac_base_h4.pt --local-dir data/weights/egotac
hf download Ruicheng/moge-2-vitl-normal > /dev/null
[ -d "$HF_HOME/hub/models--facebook--sam3" ] || hf download facebook/sam3 > /dev/null 2>&1 || echo "!! facebook/sam3 is gated: accept the licence and run 'hf auth login'"
for f in detector.pt wilor_final.ckpt; do
  [ -s third_party/WiLoR/pretrained_models/$f ] ||
    curl -fL -o third_party/WiLoR/pretrained_models/$f \
      https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/$f
done
[ -f third_party/WiLoR/mano_data/MANO_RIGHT.pkl ] ||
  echo "!! missing third_party/WiLoR/mano_data/MANO_RIGHT.pkl (register at https://mano.is.tue.mpg.de)"

if [ "${1:-}" != --no-build ]; then
  echo "== node images"
  for n in perception hands tactile retarget; do
    docker image inspect "ego2humanoid/$n:dev" > /dev/null 2>&1 || scripts/build_node.sh "$n"
  done
  docker image inspect ego2humanoid/hands-wilor:dev > /dev/null 2>&1 ||
    docker build -f nodes/hands/Dockerfile --target wilor -t ego2humanoid/hands-wilor:dev .
fi
echo "setup done; run: DOCKER=1 FAH_PY=/nonexistent HF_HOME=$HF_HOME scripts/run_pipeline.sh 0"
