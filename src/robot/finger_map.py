"""Map human five-finger curl (0 open .. 1 closed) onto the robot hand joints of one side.

Dex3 (3 fingers, 7 joints) collapses ring/pinky onto the middle finger; Sharpa Wave (5 fingers,
22 joints) drives one robot finger per human finger.
"""

from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "finger_mapping.yaml"
HUMAN_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_CANON = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def load_finger_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def sharpa_joint_names(side: str, cfg: dict | None = None) -> list[str]:
    """Hand joint names of one side in the model's actuator order."""
    cfg = (cfg or load_finger_config())["sharpa"]
    return [f"{side}_{finger}_{suf}" for finger in HUMAN_FINGERS for suf in cfg["joint_suffixes"][finger]]


def map_curls_to_sharpa(curls: np.ndarray, side: str = "right", cfg: dict | None = None) -> np.ndarray:
    """``curls``: (..., 5) in ``HUMAN_FINGERS`` order -> (..., 22) targets in the side's actuator order."""
    full = cfg or load_finger_config()
    cfg = full["sharpa"]
    curls = np.clip(np.asarray(curls, dtype=float), 0.0, 1.0)
    if curls.shape[-1] != len(HUMAN_FINGERS):
        raise ValueError("curls must have a trailing dimension of 5")
    out = []
    for i, finger in enumerate(HUMAN_FINGERS):
        key = "thumb" if finger == "thumb" else "finger"
        for suf in cfg["joint_suffixes"][finger]:
            lo, hi = cfg["open"][key][suf], cfg["closed"][key][suf]
            out.append(lo + curls[..., i] * (hi - lo))
    return np.stack(out, axis=-1)


def map_curls_to_dex3(curls: np.ndarray, side: str = "right", cfg: dict | None = None) -> np.ndarray:
    """``curls``: (..., 5) in ``HUMAN_FINGERS`` order -> (..., 7) targets in the side's actuator order."""
    cfg = cfg or load_finger_config()
    curls = np.clip(np.asarray(curls, dtype=float), 0.0, 1.0)
    if curls.shape[-1] != len(HUMAN_FINGERS):
        raise ValueError("curls must have a trailing dimension of 5")
    robot_curl = {
        name: np.mean([curls[..., HUMAN_FINGERS.index(f)] for f in srcs], axis=0)
        for name, srcs in cfg["finger_sources"].items()
    }
    lo, hi = np.asarray(cfg["open"]), np.asarray(cfg["closed"])
    per_joint = {}
    for k, joint in enumerate(_CANON):
        c = robot_curl[joint.split("_")[0]]
        per_joint[joint] = lo[k] + c * (hi[k] - lo[k])
    return np.stack([per_joint[j] for j in cfg["joint_order"][side]], axis=-1)
