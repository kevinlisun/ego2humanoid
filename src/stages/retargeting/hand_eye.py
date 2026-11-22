"""Hand-eye coordination: close the loop on the robot's own pinch point.

Open-loop retargeting replays the human wrist pose and lets the fingers follow a commanded gap. The
robot hand is not human-sized, so its pinch point ends up somewhere else than the human's, and the
error is *time-varying*: measured on clip 1, the fingers finished closing 1.1 s before the robot's
pinch point reached the object, and the hand then swept past it with the grip already shut. The
object never moved although the simulator reported 561 contact steps.

``retarget_hand`` already nudges the wrist once per frame so that the robot pinch point lands on the
human one, but that is a single Gauss-Newton step taken before the arm moves, so a few centimetres of
residual survive. This module measures the pinch point of the *commanded* pose by forward kinematics
and feeds the residual back into the wrist target until the robot tracks the intended pinch
trajectory, which is itself expressed relative to the object (see ``grasp_target``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from robot.layout import RobotLayout
from robot.pinch import PinchSolver, SharpaPinchSolver
from stages.manipulation.mjcf_scene import SceneConfig, build_scene
from stages.retargeting.kinematic import SIDE_SLICE

PINCH_TOL_M = 0.005
MAX_ITERS = 4
MAX_OFFSET_M = 0.30  # beyond this the wrist leaves the arm's workspace and the IK error explodes
GAIN = 1.0
APPROACH_FRAMES = 30  # the reach towards the object that is reshaped (1 s at 30 Hz)
SETTLE_FRAMES = 9  # the pinch hangs over the object and comes straight down for 0.3 s before closing
APPROACH_HEIGHT_M = 0.03  # above the grasp point when the vertical descent starts
APPROACH_BLEND_FRAMES = 8
HOVER_RADIUS_M = 0.06  # an open hand this close to the grasp point (horizontally) hovers over it ...
HOVER_FADE_M = 0.04  # ... fading back to the human height over this much further out
MAX_DWELL_FRAMES = 90  # how far back a human dwell at the object is still lifted (3 s)


def pinch_track(q_rad: np.ndarray, side: str, hand: str, T_sim_robot_base: np.ndarray) -> np.ndarray:
    """(N, DoF) commanded poses -> (N, 3) midpoint of the two pinching fingertips, in the sim frame."""
    from robot.kinematics import ArmIK

    model = build_scene(SceneConfig(T_sim_robot_base=T_sim_robot_base, hand=hand))
    ik = ArmIK(model, side)
    solver = (
        SharpaPinchSolver(ik, side, RobotLayout.from_model(model).finger_idx(side))
        if hand == "sharpa"
        else PinchSolver(ik, side, SIDE_SLICE[side])
    )
    out = np.empty((len(q_rad), 3))
    for i, q in enumerate(q_rad):
        a, b = solver.tips(q)
        out[i] = 0.5 * (a + b)
    return out


def grasp_window(n: int, hold_idx: np.ndarray, ramp_frames: int = 31) -> np.ndarray:
    """(n,) weight that is 1 on ``hold_idx`` and fades out over ``ramp_frames`` on either side.

    The correction has to vanish away from the grasp, otherwise the resting poses at the start and
    end of the clip are dragged off the table with it.
    """
    w = np.zeros(n)
    w[hold_idx] = 1.0
    ramp = np.hanning(ramp_frames)
    return np.maximum(np.clip(np.convolve(w, ramp / ramp.sum(), mode="same"), 0.0, 1.0), w)


def phase_windows(n: int, holds: Sequence[np.ndarray], ramp_frames: int = 31) -> np.ndarray:
    """(P, n) per-phase weights, normalised so they never sum above 1.

    A pick-and-place is accurate about a *different* object in each phase: the grasped one while the
    fingers close, the receptacle while they open. Each phase gets its own ``grasp_window``; where
    two windows overlap -- a short carry, or two interactions close in time -- the weights are scaled
    down together so the hand interpolates from one object frame to the other instead of receiving
    the sum of both corrections, which would send it between the two.
    """
    w = (
        np.stack([grasp_window(n, np.asarray(h, int), ramp_frames) for h in holds])
        if holds
        else np.zeros((0, n))
    )
    total = w.sum(axis=0)
    over = total > 1.0
    w[:, over] /= total[over]
    return w


def blend_corrections(windows: np.ndarray, aligns: Sequence[np.ndarray]) -> np.ndarray:
    """(n, 3) trajectory correction: each phase's rigid offset faded in over its own window."""
    if len(windows) == 0:
        return np.zeros((windows.shape[1] if windows.ndim == 2 else 0, 3))
    return windows.T @ np.asarray(aligns, float)


def align_pinch(
    retarget: Callable[[np.ndarray], object],
    measure: Callable[[object], np.ndarray],
    target_pinch_m: np.ndarray,
    weight: np.ndarray,
    max_iters: int = MAX_ITERS,
    tol_m: float = PINCH_TOL_M,
) -> tuple[object, np.ndarray, float]:
    """Iterate ``retarget(offset_m)`` until the robot pinch point tracks ``target_pinch_m``.

    ``retarget`` takes an (N, 3) displacement added to the active hand's landmarks and returns a
    retarget result; ``measure`` reads the achieved (N, 3) pinch track off that result, normally
    ``pinch_track``. Returns the best iterate, its offset and the weighted RMS residual in metres.
    Only the fully weighted grasp frames are scored. The ramp frames carry a deliberately partial
    correction, so including them would inflate the residual and stop the loop early; away from the
    window the hand is not interacting with anything and must keep following the human exactly.
    """
    n = len(target_pinch_m)
    offset = np.zeros((n, 3))
    best: tuple[object, np.ndarray, float] | None = None
    mask = weight >= 1.0 - 1e-9
    if not mask.any():
        mask = weight > 0
    for _ in range(max_iters):
        result = retarget(offset)
        pinch = measure(result)
        err = target_pinch_m - pinch
        rms = float(np.sqrt((weight[mask, None] * err[mask] ** 2).sum() / max(mask.sum(), 1)))
        if best is None or rms < best[2]:
            best = (result, offset.copy(), rms)
        if rms < tol_m:
            break
        step = GAIN * weight[:, None] * err
        offset = np.clip(offset + step, -MAX_OFFSET_M, MAX_OFFSET_M)
    assert best is not None
    return best


def settle_approach(
    target_pinch_m: np.ndarray,
    close_start: int,
    approach_frames: int = APPROACH_FRAMES,
    settle_frames: int = SETTLE_FRAMES,
    height_m: float = APPROACH_HEIGHT_M,
    blend_frames: int = APPROACH_BLEND_FRAMES,
    close_frames: int = 0,
    hover_z_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reach the grasp point from above and arrive early, so the fingers close around the object.

    The human hand sweeps in sideways and closes the instant it arrives. The robot arm lags its command
    by about 0.2 s, so copying that puts the descending fingers several centimetres off to the side
    when they come down, one lands on top of the object and the hand stalls above it (clip 0 Sharpa:
    5 cm off in y at the close, -10 N on the object's top face, no lift). Over the ``approach_frames``
    before ``close_start`` the human approach is replayed faster, ending ``height_m`` above the grasp
    point ``settle_frames`` early; the pinch then comes straight down onto it. Frame count and grasp
    timing are unchanged.

    The human hand may still be sinking while its fingers close (clip 1: the pinch reaches the
    object's mid-height 0.4 s after ``close_start``, the Dex3 thumb shut on the top edge 15 mm higher).
    The grasp point is therefore the lowest one of the ``close_frames`` after ``close_start``, and the
    pinch waits there while the fingers close.

    The human often rests the fingers on the object well before the pick (clip 104: 1.3 s, sliding
    it into position). The open robot hand copying that pushes the 30 g block 4.5 cm off the grasp
    point, the grip lands off-centre and the block is set down on the bowl's rim. Until the descent,
    an open pinch within ``HOVER_RADIUS_M`` of the grasp point is therefore kept at least
    ``height_m`` above it and above ``hover_z_m`` (the object's top plus a clearance), reaching back
    over such a dwell up to ``MAX_DWELL_FRAMES``.

    Returns the reshaped (N, 3) target and an (N,) weight that is 1 over the reshaped frames, ramping
    in over the first ``blend_frames`` so the correction joins the raw human motion smoothly.
    """
    target = np.asarray(target_pinch_m, float).copy()
    weight = np.zeros(len(target))
    a, b = max(close_start - approach_frames, 0), close_start - settle_frames
    if b - a < 2 or close_start >= len(target):
        return target, weight
    src = target.copy()
    low = close_start + int(np.argmin(src[close_start : close_start + close_frames + 1, 2]))
    target[close_start:low] = src[low]
    src[close_start] = src[low]
    j = np.arange(a, b + 1)
    s = (j - a) / (b - a)
    u = a + s * (close_start - a)  # the same path, played over a shorter interval
    for d in range(3):
        target[j, d] = np.interp(u, np.arange(len(src)), src[:, d])
    target[j, 2] += height_m * s * s * (3 - 2 * s)
    grasp = src[close_start]
    rise = height_m if hover_z_m is None else max(height_m, hover_z_m - grasp[2])
    r = np.linalg.norm(target[:, :2] - grasp[:2], axis=1)
    h0 = a
    while h0 > max(close_start - MAX_DWELL_FRAMES, 0) and r[h0 - 1] < HOVER_RADIUS_M + HOVER_FADE_M:
        h0 -= 1
    i = np.arange(h0, b + 1)
    near = np.clip((HOVER_RADIUS_M + HOVER_FADE_M - r[i]) / HOVER_FADE_M, 0.0, 1.0)
    target[i, 2] = np.maximum(target[i, 2], grasp[2] + rise * near)
    k = np.arange(b + 1, close_start)
    target[k] = grasp
    target[k, 2] += rise * (close_start - k) / settle_frames
    start = a if h0 == a else max(h0 - blend_frames, 0)
    weight[start:close_start] = 1.0
    n = max(min(blend_frames, b - start), 0)
    weight[start : start + n] = np.linspace(0.0, 1.0, blend_frames + 2)[1:-1][:n]
    return target, weight


def _smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def limit_carry_acceleration(
    target_pinch_m: np.ndarray,
    start: int,
    stop: int,
    dt_s: float,
    max_accel_m_s2: float = 3.0,
    taper_frames: int = 6,
    max_sigma_frames: float = 8.0,
) -> tuple[np.ndarray, float]:
    """Smooth the held part ``[start, stop)`` of the pinch path until its acceleration is bounded.

    The grip is a soft spring: a jerk of the wrist while carrying (clip 114 Sharpa: 30 m/s^2 at one
    frame of the human path) loads the fingertip contacts sideways faster than friction holds, and
    the stored energy throws the object out of the hand. The path is Gaussian-smoothed with a growing
    width, faded in and out over ``taper_frames`` so it still meets the grasp and release points
    exactly, until the peak acceleration inside the span is at most ``max_accel_m_s2`` or the width
    reaches ``max_sigma_frames``. Returns the path and the width used (0 when untouched).
    """
    from scipy.ndimage import gaussian_filter1d

    path = np.asarray(target_pinch_m, float)
    start, stop = max(int(start), 1), min(int(stop), len(path) - 1)
    if stop - start < 2 * taper_frames + 3:
        return path.copy(), 0.0

    def peak(p: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(p[start - 1 : stop + 1], 2, axis=0), axis=1).max()) / dt_s**2

    if peak(path) <= max_accel_m_s2:
        return path.copy(), 0.0
    w = np.zeros(len(path))
    w[start:stop] = 1.0
    ramp = _smoothstep(np.arange(1, taper_frames + 1) / (taper_frames + 1))
    w[start : start + taper_frames] = ramp
    w[stop - taper_frames : stop] = ramp[::-1]
    out, sigma = path.copy(), 0.0
    for sigma in np.arange(1.0, max_sigma_frames + 1e-9, 1.0):
        smooth = gaussian_filter1d(path, sigma, axis=0, mode="nearest")
        out = path + w[:, None] * (smooth - path)
        if peak(out) <= max_accel_m_s2:
            break
    return out, float(sigma)


def release_motion(
    target_pinch_m: np.ndarray,
    release_i: int,
    dwell_frames: int,
    retreat_frames: int,
    retreat_height_m: float,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Let go cleanly: hold still while the fingers open, lift straight up, then rejoin the human.

    The human hand moves away the instant it lets go and the robot copied that with the fingers
    still half closed: the object was dragged along, or stayed stuck to a pad (clips 0, 1, 10). After
    the last closed frame ``release_i`` the pinch is held at the release point for ``dwell_frames``
    (the fingers open meanwhile, see the gap schedule), rises ``retreat_height_m`` straight up over
    ``retreat_frames`` and blends back onto the given path over ``blend_frames``.

    Returns the reshaped (N, 3) path, an (N,) tracking weight (1 while reshaped, fading out over the
    blend) and the frame after the last reshaped one; the fingers stay open until then.
    """
    target = np.asarray(target_pinch_m, float).copy()
    n = len(target)
    weight = np.zeros(n)
    if release_i + 1 >= n:
        return target, weight, n
    src = target.copy()
    hold = src[release_i].copy()
    a = release_i + 1
    b = min(a + dwell_frames, n)
    target[a:b] = hold
    weight[a:b] = 1.0
    c = min(b + retreat_frames, n)
    k = np.arange(b, c)
    top = hold + np.array([0.0, 0.0, retreat_height_m])
    target[k] = hold
    target[k, 2] += retreat_height_m * _smoothstep((k - b + 1) / max(retreat_frames, 1))
    weight[k] = 1.0
    d = min(c + blend_frames, n)
    k = np.arange(c, d)
    s = _smoothstep((k - c + 1) / (blend_frames + 1))[:, None]
    target[k] = (1 - s) * top + s * src[k]
    weight[k] = 1.0 - s[:, 0]
    return target, weight, d
