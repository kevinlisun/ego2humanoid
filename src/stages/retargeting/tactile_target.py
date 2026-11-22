"""Human tactile (per MANO region) -> Dex3 tactile target, and sim-vs-target comparison.

Arrays are (N, 2, R): frames, hand (left, right), region in ``region_names``.
"""

from __future__ import annotations

import numpy as np

DEX3_FORCE_MAX_N = 10.0
CONTACT_PROB_THRESHOLD = 0.5


def build_tactile_target(
    region_names: list[str] | tuple[str, ...],
    contact_prob: np.ndarray,
    human_force: np.ndarray | None = None,
    human_force_max_n: float | None = None,
    force_max_n: float = DEX3_FORCE_MAX_N,
    prob_threshold: float = CONTACT_PROB_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Threshold ``contact_prob`` and rescale human forces into ``[0, force_max_n]`` newtons.

    Human force is divided by ``human_force_max_n`` (default: its own maximum) then scaled and clipped.
    Without ``human_force`` the force target is ``contact_prob * force_max_n`` on contact frames.
    """
    prob = np.asarray(contact_prob, float)
    if prob.ndim != 3 or prob.shape[-1] != len(region_names):
        raise ValueError("contact_prob must have shape (N, 2, R) matching region_names")
    contact = prob >= prob_threshold
    if human_force is None:
        scaled = prob
    else:
        f = np.clip(np.asarray(human_force, float), 0.0, None)
        if f.shape != prob.shape:
            raise ValueError("human_force must match contact_prob shape")
        ref = human_force_max_n if human_force_max_n is not None else float(f.max(initial=0.0))
        scaled = f / ref if ref > 0 else np.zeros_like(f)
    force = np.clip(scaled * force_max_n, 0.0, force_max_n) * contact
    return {
        "contact_target": contact,
        "force_target_n": force,
        "region_names": np.array(list(region_names)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_tactile(sim: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per-region contact precision/recall/F1 and force correlation (sim vs target), both hands pooled.

    Regions are matched by name; regions missing in ``sim`` or flagged ``has_region=False`` are skipped.
    Frames are truncated to the shorter sequence. Undefined values are NaN.
    """
    n = min(len(sim["contact"]), len(target["contact_target"]))
    sim_names = [str(x) for x in sim["region_names"]]
    has = np.asarray(sim.get("has_region", np.ones(len(sim_names), bool)), bool)
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(str(x) for x in target["region_names"]):
        if name not in sim_names or not has[sim_names.index(name)]:
            continue
        i = sim_names.index(name)
        s_c = np.asarray(sim["contact"])[:n, :, i].ravel().astype(bool)
        t_c = np.asarray(target["contact_target"])[:n, :, j].ravel().astype(bool)
        tp = int((s_c & t_c).sum())
        prec = tp / s_c.sum() if s_c.any() else float("nan")
        rec = tp / t_c.sum() if t_c.any() else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else float("nan")
        corr = _corr(
            np.asarray(sim["normal_force_n"], float)[:n, :, i].ravel(),
            np.asarray(target["force_target_n"], float)[:n, :, j].ravel(),
        )
        out[name] = {"precision": prec, "recall": rec, "f1": f1, "force_corr": corr}
    return out
