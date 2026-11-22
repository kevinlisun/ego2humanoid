"""Per-frame contact and motion probes attached to `run_rollout` through its `on_record` hook (Track RSI)."""

import mujoco
import numpy as np

_FORCE = np.zeros(6)


class ContactProbe:
    """Records penetration, contact force and object speed at every recorded frame (50 Hz)."""

    def __init__(self, model: mujoco.MjModel, object_names: list[str]):
        self.names = list(object_names)
        n_obj = len(self.names)
        obj_body = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): i for i, n in enumerate(self.names)
        }
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        world = 0
        non_robot = set(obj_body) | {table, world}
        self._body_of_geom = np.asarray(model.geom_bodyid)
        self._obj_body, self._table, self._non_robot = obj_body, table, non_robot
        self._n_obj = n_obj
        self._rows: dict[str, list] = {
            k: [] for k in ("table", "robot_obj", "obj_support", "force", "speed", "ang", "finite")
        }
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
            for a in range(model.nu)
        ]
        self.joint_group = np.array([1 if "_hand_" in (n or "") else 0 for n in names], np.int8)

    def _class(self, geom: int) -> tuple[str, int]:
        body = int(self._body_of_geom[geom])
        if body in self._obj_body:
            return "obj", self._obj_body[body]
        if body == self._table:
            return "table", -1
        if body == 0:
            return "world", -1
        return "robot", -1

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        n = self._n_obj
        pen_table = 0.0
        pen_ro, pen_os, force = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(data.ncon):
            c = data.contact[i]
            (k1, o1), (k2, o2) = self._class(int(c.geom1)), self._class(int(c.geom2))
            pen = max(-float(c.dist), 0.0)
            kinds = {k1, k2}
            if kinds == {"robot", "table"}:
                pen_table = max(pen_table, pen)
            elif "robot" in kinds and "obj" in kinds:
                j = o1 if k1 == "obj" else o2
                pen_ro[j] = max(pen_ro[j], pen)
                mujoco.mj_contactForce(model, data, i, _FORCE)
                force[j] += float(_FORCE[0])
            elif "obj" in kinds and kinds != {"obj", "robot"}:
                for k, o in ((k1, o1), (k2, o2)):
                    if k == "obj":
                        pen_os[o] = max(pen_os[o], pen)
        speed, ang = np.zeros(n), np.zeros(n)
        for body, j in self._obj_body.items():
            v = data.cvel[body]
            ang[j], speed[j] = float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))
        r = self._rows
        r["table"].append(pen_table)
        r["robot_obj"].append(pen_ro)
        r["obj_support"].append(pen_os)
        r["force"].append(force)
        r["speed"].append(speed)
        r["ang"].append(ang)
        r["finite"].append(bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()))

    def arrays(self) -> dict[str, np.ndarray]:
        r = self._rows
        n = self._n_obj
        shape = (len(r["table"]), n)
        return {
            "pen_robot_table_m": np.asarray(r["table"], float),
            "pen_robot_object_m": np.asarray(r["robot_obj"], float).reshape(shape),
            "pen_object_support_m": np.asarray(r["obj_support"], float).reshape(shape),
            "force_robot_object_n": np.asarray(r["force"], float).reshape(shape),
            "obj_speed_m_s": np.asarray(r["speed"], float).reshape(shape),
            "obj_ang_speed_rad_s": np.asarray(r["ang"], float).reshape(shape),
            "finite": np.asarray(r["finite"], bool),
            "joint_group": self.joint_group,
        }
