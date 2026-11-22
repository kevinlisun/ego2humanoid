"""Side-by-side result video: source RGB with the estimated hands (+ tactile) | MuJoCo replay."""

from pathlib import Path

import av
import cv2
import numpy as np

from stages.qa.hand_overlay import draw_frame_overlay, region_force_series


def _force_scale(force: np.ndarray) -> np.ndarray:
    """Per-hand colour-map top: the 90th percentile of that hand's non-zero forces, so the passive hand
    leaning on the table does not wash out the grasp forces of the active one."""
    out = []
    for h in range(force.shape[1]):
        pos = force[:, h][force[:, h] > 0]
        out.append(max(float(np.percentile(pos, 90)) if pos.size else 0.0, 1e-6))
    return np.asarray(out)


def overlay_frames(
    clip: Path,
    n_out: int,
    joints2d: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    active_side: str,
    t0_s: float,
    fps: float = 30.0,
    min_score: float = 0.7,
    tactile_regions_npz: Path | None = None,
    tactile_label: str = "tactile",
    sim_fps: float = 50.0,
) -> list[np.ndarray]:
    """Source RGB frames with both estimated hands drawn on them, resampled to the sim clock.

    ``joints2d`` (N, 2, 21, 2) px, ``valid`` (N, 2), ``score`` (N, 2) are on the source-video clock
    starting at ``t0_s``; the returned ``n_out`` RGB frames are on the sim clock. When
    ``tactile_regions_npz`` is given, each landmark is painted with the contact force of the MANO
    region it belongs to.
    """
    src = cv2.VideoCapture(str(clip / "video.mp4"))
    # Human tactile is sampled on the source-video clock, simulated tactile on the sim clock.
    on_sim_clock = tactile_regions_npz is not None and tactile_regions_npz.name.startswith("sim_tactile")
    n_tac = n_out if on_sim_clock else len(joints2d)
    force = None if tactile_regions_npz is None else region_force_series(tactile_regions_npz, n_tac)
    scale_n = None if force is None else _force_scale(force)  # fixed scale, comparable across frames
    frames: list[np.ndarray] = []
    # Seek once and decode forward: a seek per output frame re-decoded from the last keyframe each time.
    start = int(round(t0_s * fps))
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    at, raw = -1, None  # source index of ``raw``, the last decoded frame
    for i in range(n_out):
        k = min(int(i / sim_fps * fps), len(joints2d) - 1)
        ok = True
        while ok and at < k:
            ok = src.grab()
            at += 1
            if ok and at == k:
                ok, raw = src.retrieve()
        if not ok:
            break
        frame = raw
        frame = draw_frame_overlay(
            frame,
            joints2d[k],
            valid[k],
            score[k],
            active_side,
            min_score,
            None if force is None else force[i if on_sim_clock else k],
            scale_n,
            tactile_label,
        )
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    src.release()
    return frames


def compose(
    clip: Path,
    sim_mp4: Path,
    joints2d: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    active_side: str,
    t0_s: float,
    success: bool,
    fps: float = 30.0,
    suffix: str = "",
    min_score: float = 0.7,
    tactile_regions_npz: Path | None = None,
    tactile_label: str = "tactile",
    out_path: Path | None = None,
    left_frames: list[np.ndarray] | None = None,
):
    """Left panel: both estimated hands (skeleton, confidence, tactile force); right panel: the sim.

    ``left_frames``: the ``overlay_frames`` of these arguments when the caller already has them; they
    are used only if there is one per sim frame, otherwise they are drawn again.
    """
    sim = [f.to_ndarray(format="rgb24") for f in av.open(str(sim_mp4)).decode(video=0)]
    out_path = out_path or clip / "review" / f"side_by_side{suffix}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = sim[0].shape[:2]
    if left_frames is None or len(left_frames) != len(sim):
        left_frames = overlay_frames(
            clip,
            len(sim),
            joints2d,
            valid,
            score,
            active_side,
            t0_s,
            fps,
            min_score,
            tactile_regions_npz,
            tactile_label,
        )
    with av.open(str(out_path), "w") as out:
        stream = out.add_stream("libx264", rate=50)
        stream.width, stream.height, stream.pix_fmt = 2 * w, h, "yuv420p"
        for fs, left in zip(sim, left_frames, strict=False):
            fh, fw = left.shape[:2]
            left = cv2.resize(left, (w, int(fh * w / fw)))
            canvas = np.zeros((h, w, 3), np.uint8)
            canvas[: left.shape[0]] = left[:h]
            label = f"sim pick-and-place: {'SUCCESS' if success else 'FAIL'}"
            cv2.putText(fs, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(
                canvas, "RGB + estimated hands", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            for pkt in stream.encode(av.VideoFrame.from_ndarray(np.hstack([canvas, fs]), format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return out_path
