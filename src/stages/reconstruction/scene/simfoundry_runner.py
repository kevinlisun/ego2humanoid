"""Run SimFoundry pipeline A on one clip with VLM requests routed through the Copilot relay."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from stages.reconstruction.scene.vlm_relay import PENDING_RELAY_EXIT_CODE, relay_status

HOOK_DIR = Path(__file__).parent / "relay_hook"
# Stage 9 (articulation) is unnecessary for rigid objects; 13-14 need USD/OmniGibson.
STAGES = ("1b", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12")
# The relay is text-only: skip the Gemini image upsample and remove objects with the local
# FLUX.1-Kontext-dev model (SimFoundry's `flux` removal backend) instead of `gemini-3-pro-image`.
DEFAULT_OVERRIDES = (
    "s5_scene.use_upsampled_source_image=false",
    "s5_scene.removal_model=flux",
    "s6_upsample.model=flux",
)


def build_command(
    simfoundry_dir: Path,
    scene_name: str,
    root_dir: Path,
    video: Path,
    stages: tuple[str, ...] = STAGES,
    extra_overrides: tuple[str, ...] = DEFAULT_OVERRIDES,
    stream: bool = True,
) -> list[str]:
    cmd = [
        "bash",
        str(simfoundry_dir / "scripts/pipeline/A_reconstruction/run.sh"),
        "--scene-name",
        scene_name,
        "--root-dir",
        str(root_dir),
        "--video-fpath",
        str(video),
        "--include",
        ",".join(stages),
        "--cache-mode",
        "--skip-successful",
        "--stream" if stream else "--no-stream",
    ]
    return cmd + (["--", *extra_overrides] if extra_overrides else [])


def run_round(
    simfoundry_dir: Path,
    scene_name: str,
    root_dir: Path,
    video: Path,
    conda_env_dir: Path | None = None,
    hf_home: Path | None = None,
    **kw,
) -> tuple[int, int]:
    """One resumable pass. Returns (exit_code, pending_request_count); exit 75 means answers are needed."""
    relay_dir = root_dir / scene_name / "relay"
    env = dict(os.environ)
    env["E2H_RELAY_DIR"] = str(relay_dir)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HOOK_DIR), env.get("PYTHONPATH")]))
    env["CACHE_MODE"] = "1"
    if conda_env_dir is not None:
        env["CONDA_ENVS_DIRS"] = str(conda_env_dir)
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    proc = subprocess.run(build_command(simfoundry_dir, scene_name, root_dir, video, **kw), env=env)
    pending = len(relay_status(relay_dir).pending)
    code = PENDING_RELAY_EXIT_CODE if pending and proc.returncode != 0 else proc.returncode
    return code, pending
