"""Legacy `E2H_*` environment variables of `scripts/retarget_estimated.py` mapped onto `SimParams` paths."""

import logging
from collections.abc import Mapping

log = logging.getLogger(__name__)

# env var -> (SimParams path, parser)
_BOOL_NOT_ZERO = lambda v: v != "0"  # noqa: E731  (the old switches were "on unless 0")
LEGACY_ENV = {
    "E2H_ADJUST_STANCE": ("trajectory.adjust_stance", _BOOL_NOT_ZERO),
    "E2H_SCENE_OBJECTS": ("trajectory.scene_objects", _BOOL_NOT_ZERO),
    "E2H_PLACE_DWELL": ("trajectory.place_dwell_frames", int),
    "E2H_PLACE_LEAD": ("trajectory.place_lead_frames", int),
    "E2H_RELEASE_OPEN": ("trajectory.release_open_frames", int),
    "E2H_RETREAT_M": ("trajectory.retreat_height_m", float),
    "E2H_MINTHICK": ("trajectory.min_object_thickness_m", float),
    "E2H_TABLE_MARGIN": ("trajectory.table_margin_m", float),
    "E2H_TABLE_NEAR_EDGE": ("trajectory.table_near_edge_m", float),
    "E2H_SQUEEZE": ("trajectory.squeeze_m", float),
    "E2H_GRIP_N": ("trajectory.grip_min_n", float),
    "E2H_CARRY_ACCEL": ("trajectory.carry_accel_m_s2", float),
}


def legacy_overlay(env: Mapping[str, str]) -> dict:
    """Nested overlay for the legacy variables set in `env`; each hit is logged as deprecated."""
    out: dict = {}
    for name, (path, parse) in LEGACY_ENV.items():
        if name not in env:
            continue
        log.warning("deprecated env %s -> %s", name, path)
        node = out
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = parse(env[name])
    return out
