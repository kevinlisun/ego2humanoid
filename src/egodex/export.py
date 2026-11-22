"""Export an EgoDex clip to the pipeline layout: ``video.mp4`` + ``episode.json`` + ``gt/``.

The pipeline input is RGB only. Ground truth (intrinsics, camera poses, ARKit hand landmarks) is written
under ``gt/`` for evaluation and must never be read by pipeline stages.
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np

from contracts import ArtifactRef, EpisodeInput, VideoSpec
from egodex.clip_reader import EgoDexClip


def export_clip(clip: EgoDexClip, out_dir: str | Path) -> Path:
    out = Path(out_dir) / clip.clip_id
    (out / "gt").mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    shutil.copyfile(clip.video_path, video)
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    episode = EpisodeInput(
        episode_id=clip.clip_id,
        video=VideoSpec(video=ArtifactRef(uri="video.mp4", sha256=sha), fps_nominal=30.0),
        task_description=str(
            clip.attrs.get("description", EpisodeInput.model_fields["task_description"].default)
        ),
    )
    (out / "episode.json").write_text(episode.model_dump_json(indent=2))
    np.savez_compressed(
        out / "gt" / "egodex_gt.npz",
        timestamps_s=clip.timestamps_s,
        intrinsics=clip.intrinsics,
        T_world_camera=clip.T_world_camera,
        joints_world=clip.joints_world,
        joints_camera=clip.joints_camera(),
        confidence=clip.confidence,
    )
    return out
