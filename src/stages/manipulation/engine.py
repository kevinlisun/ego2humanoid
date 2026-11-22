"""Seeded multi-rollout replay: runs the trajectory, QA-checks each seed, writes the SimRunBundle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from common.hashing import sha256_file
from common.io import dump_json, write_bundle
from contracts.enums import ReplayValidity, RolloutMode, StageStatus
from contracts.qa import QAReport, SimRunBundle
from robot.asset import MJCF_PATH, probe_robot
from stages.manipulation.checks import CheckResult, check_rollout
from stages.manipulation.mjcf_scene import SceneConfig
from stages.manipulation.render import VideoRecorder, select_gl_backend
from stages.manipulation.rollout import Perturbation, RolloutResult, run_rollout


@dataclass
class ReplayOutcome:
    bundle: SimRunBundle
    report: QAReport
    results: list[tuple[RolloutResult, CheckResult]]


def replay_episode(
    cfg: SceneConfig,
    timestamps_s: np.ndarray,
    q_target_rad: np.ndarray,
    out_dir: str | Path,
    episode_id: str,
    n_seeds: int = 5,
    render: bool = True,
    perturbation: Perturbation | None = None,
    input_source: Literal["rgb_estimated", "oracle_gt", "synthetic"] = "synthetic",
) -> ReplayOutcome:
    out_dir = Path(out_dir)
    probe = probe_robot()
    limits = np.stack([probe.joint_lower_rad, probe.joint_upper_rad], axis=1)
    results = []
    for seed in range(n_seeds):
        r = run_rollout(cfg, timestamps_s, q_target_rad, seed=seed, perturbation=perturbation)
        c = check_rollout(r.arrays, limits)
        c.passed = c.passed and r.contact_steps_with_object > 0 and not r.state_injection_after_reset
        if r.contact_steps_with_object == 0:
            c.error_codes.append("NO_ROBOT_OBJECT_CONTACT")
        if r.state_injection_after_reset:
            c.error_codes.append("STATE_INJECTION")
        results.append((r, c))

    succeeded = sum(c.passed for _, c in results)
    best_r, best_c = next(((r, c) for r, c in results if c.passed), results[0])
    video = out_dir / "render.mp4"
    if render:
        select_gl_backend()
        rec = VideoRecorder(video)
        run_rollout(
            cfg, timestamps_s, q_target_rad, seed=best_r.seed, perturbation=perturbation, on_record=rec
        )
        rec.close()
    validated = succeeded >= 3 and n_seeds >= 5
    status = StageStatus.PASS if validated else StageStatus.FAIL
    codes = [] if validated else sorted({e for _, c in results for e in c.error_codes} or {"REPLAY_FAILED"})
    bundle = SimRunBundle(
        episode_id=episode_id,
        input_source=input_source,
        manifest_uri=str(out_dir / "manifest.json"),
        data_uri=str(out_dir / "actual_states.npz"),
        frame_count=len(best_r.arrays["timestamps_s"]),
        status=status,
        error_codes=codes,
        video_mp4=str(video) if render else "",
        actual_npz=str(out_dir / "actual_states.npz"),
        robot_asset_sha256=sha256_file(MJCF_PATH),
        replay_validity=ReplayValidity.PHYSICS_VALIDATED if validated else ReplayValidity.PHYSICS_FAILED,
        rollout_mode=RolloutMode.ACTION_CONTROLLED,
        physics_step_count=best_r.physics_step_count,
        object_driven_by_robot_contacts=best_r.contact_steps_with_object > 0,
        state_injection_after_reset=any(r.state_injection_after_reset for r, _ in results),
        rollouts_attempted=n_seeds,
        rollouts_succeeded=succeeded,
        task_success=validated,
    )
    write_bundle(out_dir, bundle, best_r.arrays, "actual_states.npz")
    report = QAReport(
        episode_id=episode_id,
        input_source=input_source,
        status=status,
        replay_validity=bundle.replay_validity,
        accepted_for_dataset=validated and input_source == "rgb_estimated",
        task_success=validated,
        metrics={
            **best_c.metrics,
            "rollouts_attempted": n_seeds,
            "rollouts_succeeded": succeeded,
        },
        error_codes=codes,
        evidence_uris=[str(video)] if render else [],
    )
    dump_json(out_dir / "qa_report.json", report.model_dump(mode="json"))
    return ReplayOutcome(bundle, report, results)
