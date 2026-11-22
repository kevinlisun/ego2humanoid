"""One resumable SimFoundry pass on a clip (Copilot relay for text VLM calls, local FLUX for removal).

Usage: python scripts/run_simfoundry_clip.py <scene_name> <video.mp4> [--stages 5,6] [--no-stream]
Exit code 75 means relay requests await a Copilot answer (see data/scenes/<scene>/relay/outbox).
"""

import argparse
import os
import sys
from pathlib import Path

from stages.reconstruction.scene.simfoundry_runner import STAGES, run_round

ROOT = Path(__file__).resolve().parents[1]
HOT = Path("/mnt/nvme_hot")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_name")
    ap.add_argument("video", type=Path)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token"))
    os.environ["PATH"] = f"{ROOT / 'data/envs/miniforge/bin'}{os.pathsep}{os.environ['PATH']}"
    os.environ["CONDA_PKGS_DIRS"] = str(HOT / "pkgs")
    os.environ["TMPDIR"] = str(HOT / "tmp")
    # The root disk is full: keep every download cache (rembg, pooch, torch hub) on the NVMe.
    os.environ["U2NET_HOME"] = str(HOT / "cache/u2net")
    os.environ["XDG_CACHE_HOME"] = str(HOT / "cache/xdg")
    os.environ["TORCH_HOME"] = str(HOT / "cache/torch")
    os.environ["HY3DGEN_MODELS"] = str(HOT / "cache/hy3dgen")
    code, pending = run_round(
        ROOT / "third_party/SimFoundry",
        args.scene_name,
        ROOT / "data/scenes",
        args.video.resolve(),
        conda_env_dir=HOT / "envs/conda_envs",
        hf_home=HOT / "hf",
        stages=tuple(args.stages.split(",")),
        stream=not args.no_stream,
    )
    print(f"exit={code} pending_relay_requests={pending}")
    return code


if __name__ == "__main__":
    sys.exit(main())
