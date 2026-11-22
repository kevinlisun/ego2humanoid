"""In-process local runner; stops at the first failing stage and never fabricates downstream outputs."""

from pathlib import Path

from orchestration.dag import Dag


def run_dag(dag: Dag, root: str | Path) -> dict[str, Path]:
    root = Path(root)
    by_name = {s.name: s for s in dag.stages}
    outputs: dict[str, Path] = {}
    for name in dag.order():
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        stage = by_name[name]
        stage.fn(out, {d: outputs[d] for d in stage.deps})
        outputs[name] = out
    return outputs


def synthetic_replay_dag(episode_id: str = "ep_synthetic") -> Dag:
    """Scripted G1+Dex3 pick-place -> physics replay -> QA artefacts (no real data needed)."""
    import numpy as np

    from common.io import save_npz
    from stages.manipulation.engine import replay_episode
    from stages.manipulation.mjcf_scene import BoxObject, SceneConfig
    from stages.manipulation.scripted import scripted_pick_place

    cube = np.array([0.38, -0.12, 0.825])
    base = np.eye(4)
    base[2, 3] = 0.793
    cfg = SceneConfig(
        T_sim_robot_base=base,
        table_top_z_m=0.80,
        objects=(BoxObject("cube", (0.025,) * 3, tuple(cube), mass_kg=0.1),),
    )

    def retarget(out: Path, _: dict[str, Path]) -> None:
        t, q = scripted_pick_place(cfg, cube)
        save_npz(out / "robot_traj.npz", {"timestamps_s": t, "q_target_rad": q})

    def replay(out: Path, ups: dict[str, Path]) -> None:
        from common.io import load_npz

        traj = load_npz(ups["retarget"] / "robot_traj.npz")
        res = replay_episode(cfg, traj["timestamps_s"], traj["q_target_rad"], out, episode_id)
        if not res.report.task_success:
            raise RuntimeError(f"replay rejected: {res.report.error_codes}")

    dag = Dag()
    dag.add("retarget", retarget)
    dag.add("replay", replay, ("retarget",))
    return dag
