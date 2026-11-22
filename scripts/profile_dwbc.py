"""L.3: measure the Decoupled WBC capability envelope on our G1 asset -> configs/dwbc_profile.json."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from robot.asset import MJCF_PATH
from robot.dwbc import DwbcRobot

OUT = Path("configs/dwbc_profile.json")


def new_robot():
    m = mujoco.MjSpec.from_file(str(MJCF_PATH)).compile()
    d = mujoco.MjData(m)
    r = DwbcRobot(m, d)
    r.reset_pose(pelvis_z=0.79)
    return m, d, r


def run(r, d, m, secs, vel=(0, 0, 0), height=None, rpy=None):
    for _ in range(int(secs / m.opt.timestep)):
        r.cmd.vel[:] = vel
        if height is not None:
            r.cmd.height_m = height
        if rpy is not None:
            r.cmd.rpy_rad[:] = rpy
        r.step()
        if d.qpos[2] < 0.4:
            return False
    return True


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def walk_case(vel, seeds=3, secs=6.0):
    res = {"falls": 0, "speed": [], "yaw_rate": [], "drift": []}
    for s in range(seeds):
        m, d, r = new_robot()
        d.qvel[:2] = np.random.default_rng(s).normal(0, 0.02, 2)
        run(r, d, m, 2.0)
        p0, y0 = d.qpos[:2].copy(), yaw_of(d)
        if not run(r, d, m, secs, vel):
            res["falls"] += 1
            continue
        dp, dy = d.qpos[:2] - p0, yaw_of(d) - y0
        dy = (dy + np.pi) % (2 * np.pi) - np.pi
        c, sn = np.cos(y0), np.sin(y0)
        fwd, lat = c * dp[0] + sn * dp[1], -sn * dp[0] + c * dp[1]
        res["speed"].append(float(fwd / secs))
        res["drift"].append(float(lat / secs))
        res["yaw_rate"].append(float(dy / secs))
    out = {k: (round(float(np.mean(v)), 3) if v else None) for k, v in res.items() if isinstance(v, list)}
    return {"vel": list(vel), "falls": res["falls"], "lateral_key": "drift = lateral m/s", **out}


def stop_case(vel=(0.4, 0, 0)):
    m, d, r = new_robot()
    run(r, d, m, 2.0)
    run(r, d, m, 4.0, vel)
    p = d.qpos[:2].copy()
    t_settle, dist = None, 0.0
    for i in range(int(4 / m.opt.timestep)):
        r.cmd.vel[:] = 0
        r.step()
        if i % 25 == 0 and t_settle is None and np.linalg.norm(d.qvel[:2]) < 0.05:
            t_settle = i * m.opt.timestep
    dist = float(np.linalg.norm(d.qpos[:2] - p))
    return {"stop_distance_m": round(dist, 3), "settle_s": t_settle}


def posture_case(height, rpy):
    m, d, r = new_robot()
    ok = run(r, d, m, 3.0, height=height, rpy=rpy)
    if ok:
        run(r, d, m, 1.0, height=height, rpy=rpy)
    ok = ok and d.qpos[2] > 0.4
    return {"height": height, "rpy": list(rpy), "stable": bool(ok), "pelvis_z": round(float(d.qpos[2]), 3)}


def main():
    prof = {"controller": "decoupled_wbc", "asset": str(MJCF_PATH.name), "walk": [], "posture": []}
    for vel in [
        (0.2, 0, 0),
        (0.4, 0, 0),
        (0.6, 0, 0),
        (-0.3, 0, 0),
        (0, 0.2, 0),
        (0, 0, 0.3),
        (0, 0, -0.3),
        (0.3, 0, 0.3),
    ]:
        prof["walk"].append(walk_case(vel))
        print(prof["walk"][-1], flush=True)
    prof["stop"] = stop_case()
    print(prof["stop"], flush=True)
    for h in (0.5, 0.6, 0.7, 0.74, 0.8):
        for rpy in ((0, 0, 0), (0, 0.3, 0), (0, 0.5, 0), (0, -0.3, 0), (0.2, 0, 0), (0, 0, 0.5)):
            prof["posture"].append(posture_case(h, rpy))
            print(prof["posture"][-1], flush=True)
    OUT.write_text(json.dumps(prof, indent=1))


if __name__ == "__main__":
    sys.exit(main())
