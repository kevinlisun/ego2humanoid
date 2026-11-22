"""EgoDex clip reader: paired ``N.hdf5`` (ARKit poses) + ``N.mp4`` (1080p, 30 Hz).

Poses are SE3 in the ARKit origin frame. Hand joints are exported in a 21-point MANO-like order
(wrist, then thumb/index/middle/ring/little with 4 joints each, base to tip); EgoDex's extra
metacarpal joints are dropped. Ground-truth hands are ARKit estimates and serve as an evaluation
reference only (see ``evaluation/hand_metrics.py``), never as a pipeline input.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 30.0
FINGERS = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
HAND_JOINTS_21 = ("Hand",) + tuple(
    f"{f}{j}" for f in FINGERS for j in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
)


@dataclass
class EgoDexClip:
    clip_id: str
    video_path: Path
    timestamps_s: np.ndarray  # (N,)
    intrinsics: np.ndarray  # (3, 3)
    T_world_camera: np.ndarray  # (N, 4, 4)
    joints_world: np.ndarray  # (N, 2, 21, 3) left, right
    T_world_wrist: np.ndarray  # (N, 2, 4, 4)
    confidence: np.ndarray  # (N, 2) wrist confidence, 1.0 if the file has none
    attrs: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps_s)

    def joints_camera(self) -> np.ndarray:
        """Hand joints in the camera frame (OpenCV convention: +x right, +y down, +z forward)."""
        inv = np.linalg.inv(self.T_world_camera)
        ones = np.ones((*self.joints_world.shape[:-1], 1))
        h = np.concatenate([self.joints_world, ones], axis=-1)
        return np.einsum("nij,nhkj->nhki", inv, h)[..., :3]

    def active_hand(self) -> str | None:
        """Hand named in the ``environment`` metadata (``hand:right``), if any."""
        for part in str(self.attrs.get("environment", "")).split(","):
            key, _, val = part.strip().partition(":")
            if key == "hand" and val in ("left", "right"):
                return val
        return None


def load_clip(hdf5_path: str | Path) -> EgoDexClip:
    import h5py

    p = Path(hdf5_path)
    with h5py.File(p) as f:
        tf = f["transforms"]
        cam = tf["camera"][:].astype(np.float64)
        n = len(cam)
        joints, wrists, conf = [], [], []
        for side in ("left", "right"):
            joints.append(np.stack([tf[side + j][:, :3, 3] for j in HAND_JOINTS_21], axis=1))
            wrists.append(tf[side + "Hand"][:].astype(np.float64))
            has_conf = "confidences" in f and side + "Hand" in f["confidences"]
            conf.append(f["confidences"][side + "Hand"][:] if has_conf else np.ones(n))
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        return EgoDexClip(
            clip_id=f"{p.parent.name}_{p.stem}",
            video_path=p.with_suffix(".mp4"),
            timestamps_s=np.arange(n) / FPS,
            intrinsics=f["camera/intrinsic"][:].astype(np.float64),
            T_world_camera=cam,
            joints_world=np.stack(joints, axis=1).astype(np.float64),
            T_world_wrist=np.stack(wrists, axis=1),
            confidence=np.stack(conf, axis=1).astype(np.float64),
            attrs=attrs,
        )


def list_clips(root: str | Path) -> list[Path]:
    """All ``*.hdf5`` clips that have a paired mp4, sorted."""
    return sorted(p for p in Path(root).rglob("*.hdf5") if p.with_suffix(".mp4").exists())
