"""L.17: turn a retargeted EgoDex clip into a loco-manipulation case (scenario + object-relative task).

The retarget stage solved the arm/finger joints for a fixed pelvis at the world origin (z =
``FIXED_PELVIS_Z``, facing +x, table in front). Its wrist poses become object-relative keyframes, the clip
scene is shifted so the table sits ``TABLE_X_M`` from the robot start, and the robot walks to a stance from
which they are reachable (stances around the retarget's own are offered to the planner as candidates).
Retargeted fingers drive the hand stages.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from contracts.locomanip import ManipulationTask, TaskKeyframe
from robot.layout import RobotLayout
from stages.locomotion.plan.candidates import Stance
from stages.locomotion.plan.ik import StanceIK
from stages.locomotion.tabletop import Scenario
from stages.reconstruction.objects.mug_mesh import procedural_mug
from stages.reconstruction.objects.prompt import object_prompt
from stages.retargeting.sharpa_pinch import finger_pose

DERIVED = Path("data/derived/egodex")
TABLE_X_M = 3.0
PRE_GRASP_LEAD_S = (0.6, 0.2)  # human approach samples before the grasp onset
PRE_GRASP_MAX_M = 0.10  # keep the approach direction, but not the human's full stand-off distance
LIFT_MAX_M = 0.12  # a human lifts far above the table; the robot only needs to clear it to count as lifted
SQUEEZE_LAG_S = 0.4  # finger pose this long after the onset is the "squeezed" stage
# Stances around the retarget's own: offsets back from the table, sideways, yaw, pelvis height, waist pitch.
NATIVE_DX_M, NATIVE_DY_M, NATIVE_YAW_RAD = (
    np.arange(-0.30, 0.01, 0.05),
    (-0.1, -0.05, 0, 0.05, 0.1),
    (-0.2, 0, 0.2),
)
NATIVE_HEIGHTS_M, NATIVE_WAISTS_RAD = (0.74, 0.7, 0.65), (0.0, 0.15, 0.3, 0.45)


class ClipPoses:
    """Finger stages from the retargeted fingers of the active hand (``carry`` stays the generic fist)."""

    def __init__(self, stages: dict[str, dict[str, float]], side: str):
        self.stages, self.side = stages, side

    def pose(self, stage: str, side: str = "right") -> dict[str, float]:
        if stage == "carry":
            return finger_pose("closed", side)
        return dict(self.stages[stage])


@dataclass
class ClipCase:
    clip: str
    side: str
    scenario: Scenario
    task: ManipulationTask
    poses: ClipPoses
    native_stances: list[Stance]


def _wrist_fk(side: str, q_rows: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sik = StanceIK(side)
    fm = sik.model
    idx = [
        names.index(n) if (n := mujoco.mj_id2name(fm, mujoco.mjtObj.mjOBJ_ACTUATOR, i)) in names else -1
        for i in range(fm.nu)
    ]
    P, R = [], []
    for q in q_rows:
        p, r = sik.ik.wrist_pose(np.array([q[i] if i >= 0 else 0.0 for i in idx]))
        P.append(p)
        R.append(r)
    return np.array(P), np.array(R)


MUG_WORDS = ("mug", "cup")
MUG_DUP_RADIUS_M = 0.10  # detections this close to the mug are the same mug (cup base, rim)


def _real2sim_scene(clip_dir: Path | None, shift_x: float, held_phrase: str = "") -> dict:
    """Perceived scenery saved by the retarget (``scene_sim.pkl``), moved into the loco frame."""
    f = None if clip_dir is None else clip_dir / "scene_sim.pkl"
    if f is None or not f.exists():
        return {}
    with open(f, "rb") as fh:
        z = pickle.load(fh)
    base = np.asarray(z["base"], float)
    if not (np.allclose(base[:3, :3], np.eye(3), atol=1e-3) and np.allclose(base[:2, 3], 0, atol=1e-3)):
        print(f"warning: retarget base is not the identity, scenery may be misplaced:\n{base}")
    move = lambda o: replace(o, pos_m=(o.pos_m[0] + shift_x, o.pos_m[1], o.pos_m[2]))  # noqa: E731
    decor, held = [move(o) for o in z["extra"]], z["pbox"]
    out = {"table_rgba": [0.88, 0.88, 0.85, 1.0]}
    if held_phrase.split()[-1:] and held_phrase.split()[-1] in MUG_WORDS:
        # The held mug is also detected as scenery (cup, its base); keep only the simulated one, drawn clean.
        mugs = [o for o in decor if any(w in o.name for w in MUG_WORDS)]
        decor = [
            o
            for o in decor
            if o not in mugs
            and all(np.linalg.norm(np.subtract(o.pos_m[:2], m.pos_m[:2])) > MUG_DUP_RADIUS_M for m in mugs)
        ]
        height = 0.09
        verts, faces = procedural_mug(height_m=height)
        verts[:, 2] += height / 2 - float(held.half_size_m[2])  # the base rests on the table
        out["object_visual"] = replace(
            held, name="mug_visual", mesh_vertices=verts, mesh_faces=faces, mesh_uv=None, mesh_texture=None,
            collision_pieces=(), rgba=(0.95, 0.95, 0.92, 1.0),
        )  # fmt: skip
    out["decor"] = tuple(decor)
    return out


def _scenario(j: dict, clip_dir: Path | None = None) -> Scenario:
    po = j["perceived_object"]
    base = Scenario.load().cfg
    tc = np.array(po["table_center_xy_m"])
    hx, hy = po["table_half_size_xy_m"]
    obj_xy, goal_xy = np.array(po["start_m"][:2]), np.array(po["object_final_xy_m"])
    cfg = {
        **base,
        "name": f"clip_{j.get('clip', 'x')}",
        "table": {
            "center_xy": [TABLE_X_M, float(tc[1])],
            "half_size": [hx, hy, 0.02],
            "top_z": po["table_top_z_m"],
        },
        "object": {
            **base["object"],
            "name": "object",
            "half_size": list(po["half_size_m"]),
            "xy_on_table": (obj_xy - tc).tolist(),
        },
        "goal_zone": {**base["goal_zone"], "xy_on_table": (goal_xy - tc).tolist()},
    }
    r2s = _real2sim_scene(clip_dir, TABLE_X_M - float(tc[0]), j.get("held_phrase", ""))
    if r2s:
        cfg["table"]["rgba"] = r2s["table_rgba"]
        cfg["goal_zone"]["rgba"] = [0.1, 0.8, 0.2, 0.12]
        cfg["decor"] = r2s["decor"]
        if "object_visual" in r2s:
            cfg["object_visual"] = r2s["object_visual"]
    return Scenario(cfg)


def load_clip(clip: int | str, derived: Path = DERIVED) -> ClipCase:
    cid = f"basic_pick_place_{clip}" if str(clip).isdigit() else str(clip)
    d = derived / cid
    j = json.loads((d / "retarget_rgb_sharpa.json").read_text())
    j["clip"] = cid
    ep = d / "episode.json"
    j["held_phrase"] = object_prompt(json.loads(ep.read_text())["task_description"]) if ep.exists() else ""
    z = np.load(d / "retarget_rgb_sharpa.npz")
    ts, q = z["timestamps_s"], z["q_target_rad"]
    side, po = j["side"], j["perceived_object"]
    sc = _scenario(j, d)
    obj0 = np.array([*po["start_m"][:2], po["table_top_z_m"] + po["half_size_m"][2]])
    names = list(RobotLayout.from_model(sc.build("sharpa")[0]).actuator_names)
    P, R = _wrist_fk(side, q, names)  # clip frame, pelvis at the origin

    # Video-only clips have no tactile timing; fall back to the hold window.
    tt = po.get("tactile_timing") or {}
    t_g = float(tt.get("grasp_onset_s", po["hold_s"][0]))
    t_rel = float(po["release_t_s"])
    at = lambda t: int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))  # noqa: E731
    goal_obj = np.array([*po["object_final_xy_m"], obj0[2]])
    held = (ts >= t_g) & (ts <= t_rel)
    t_lift = float(ts[np.flatnonzero(held)[np.argmax(P[held, 2])]]) if held.any() else t_g + 0.5

    def kf(phase, t):
        """Clip wrist pose at ``t`` expressed in the object frame (object at start before the place)."""
        T = np.eye(4)
        T[:3, :3] = R[at(t)]
        T[:3, 3] = P[at(t)] - (goal_obj if phase in ("place", "release") else obj0)
        return TaskKeyframe(
            phase=phase, time_s=t, T_object_hand=T.tolist(), hand=side, contact=phase == "grasp"
        )

    grasp_kf, lift_kf = kf("grasp", t_g), kf("lift", t_lift)
    kfs = []
    for lead in PRE_GRASP_LEAD_S:
        pre = kf("pre_grasp", t_g - lead)
        T_p = np.array(pre.T_object_hand)
        delta = T_p[:3, 3] - np.array(grasp_kf.T_object_hand)[:3, 3]
        n = float(np.linalg.norm(delta))
        if n > PRE_GRASP_MAX_M:
            T_p[:3, 3] -= delta * (1 - PRE_GRASP_MAX_M / n)
            pre.T_object_hand = T_p.tolist()
        kfs.append(pre)
    T_l = np.array(lift_kf.T_object_hand)
    T_g = np.array(grasp_kf.T_object_hand)
    T_l[:3, 3] = T_g[:3, 3] + [0.0, 0.0, min(max(T_l[2, 3] - T_g[2, 3], 0.05), LIFT_MAX_M)]
    lift_kf.T_object_hand = T_l.tolist()
    kfs += [grasp_kf, lift_kf, kf("place", t_rel - 0.1), kf("release", t_rel + 0.4)]
    layout_names = [n for n in names if n.startswith(f"{side}_") and n in finger_pose("closed", side)]
    row = lambda t: {n: float(q[at(t), names.index(n)]) for n in layout_names}  # noqa: E731
    poses = ClipPoses(
        {"open": row(t_g - PRE_GRASP_LEAD_S[0]), "touch": row(t_g), "squeeze": row(t_g + SQUEEZE_LAG_S)}, side
    )
    task = ManipulationTask(
        scenario=sc.cfg["name"],
        object_name="object",
        goal_xy_on_table=tuple(sc.cfg["goal_zone"]["xy_on_table"]),
        keyframes=kfs,
        source="rgb_estimated",
    )
    S = TABLE_X_M - po["table_center_xy_m"][0]
    native = [
        Stance(S + dx, dy, yaw, h, w)
        for dx in NATIVE_DX_M
        for dy in NATIVE_DY_M
        for yaw in NATIVE_YAW_RAD
        for h in NATIVE_HEIGHTS_M
        for w in NATIVE_WAISTS_RAD
    ]
    return ClipCase(cid, side, sc, task, poses, native)
