"""Build a metric twin scene in the *video camera frame* from the perception outputs (Track BT, BT.10).

    python scripts/build_twin_scene_cam.py <clip_dir>        # numpy + pillow + av only

The sim frame is robot-centric (placement offsets, table height tied to the grip), so it cannot be compared
with the video. Here everything lives in the MoGe camera frame of the real video, scaled by the pipeline's
``depth_scale`` so that objects and the estimated hands share one metric scale. Writes
``review/twin_scene_cam/scene_cam.json`` (+ ``hands.npz``).
"""

import json
import sys
from pathlib import Path

import av
import numpy as np

DOWNSAMPLE_COLOUR_PCT = (3, 97)


def _frame(video: Path, index: int) -> np.ndarray:
    with av.open(str(video)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == index:
                return f.to_ndarray(format="rgb24")
    raise IndexError(index)


def table_basis(n: np.ndarray) -> np.ndarray:
    """Rows e1, e2, up with up = table normal (pointing at the camera) and e1 near the camera x axis."""
    up = n / np.linalg.norm(n)
    e1 = np.array([1.0, 0, 0]) - up[0] * up
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])


def footprint(abh: np.ndarray) -> dict:
    """Oriented footprint of table-frame points: centre (a, b), yaw, extents (long, short) and top height."""
    lo_p, hi_p = DOWNSAMPLE_COLOUR_PCT
    ab = abh[:, :2]
    mu = np.median(ab, 0)
    w, v = np.linalg.eigh(np.cov((ab - mu).T))
    axes = v[:, ::-1]  # principal axis first
    loc = (ab - mu) @ axes
    lo, hi = np.percentile(loc, [lo_p, hi_p], axis=0)
    centre = mu + axes @ ((lo + hi) / 2)
    return {
        "centre_ab": centre.tolist(),
        "yaw": float(np.arctan2(axes[1, 0], axes[0, 0])),
        "size_ab": (hi - lo).tolist(),
        "top_h": float(np.percentile(abh[:, 2], hi_p)),
        "base_h": float(max(np.percentile(abh[:, 2], lo_p), 0.0)),
    }


def ransac_plane(pts: np.ndarray, iters: int = 400, tol: float = 0.03, seed: int = 0):
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - s[0]) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = (n, float(n @ s[0]), inl), int(inl.sum())
    return best


def main() -> None:
    clip = Path(sys.argv[1])
    out = clip / "review" / "twin_scene_cam"
    out.mkdir(parents=True, exist_ok=True)
    inv = json.loads((clip / "scene" / "inventory.json").read_text())
    pts = np.load(clip / "scene" / "points.npz")
    intr = json.loads((clip / "intrinsics_est.json").read_text())
    rj = json.loads((clip / "retarget_rgb_sharpa_bt.json").read_text())
    s = float(rj["perceived_object"]["depth_scale"])
    f, (W, H) = float(intr["focal_px"]), (intr["width"], intr["height"])
    n_cam = np.array(inv["table_normal_cam"])
    B = table_basis(n_cam)
    plane_off = float(inv["table_offset_cam"]) * s

    def to_table(p: np.ndarray) -> np.ndarray:
        q = p * s
        return np.stack([q @ B[0], q @ B[1], q @ B[2] - plane_off], axis=-1)

    objects = []
    for o in inv["objects"]:
        p = pts[o["points"]].reshape(-1, 3)
        p = p[np.isfinite(p).all(1)]
        objects.append(
            {"name": o["name"], "task": bool(o["task_phrase"]), **footprint(to_table(p)), "mesh_id": o["id"]}
        )

    # Pen: SAM3 candidate 0 (task object) before the grasp, hand-attached while held, then where it was released.
    sam = np.load(clip / "objects_sam3.npz", allow_pickle=True)
    ok = np.isfinite(sam["centroid_cam"][:, 0, 0]) & (sam["score"][:, 0] > 0.3)
    pen_cam = np.median(sam["centroid_cam"][ok][:6, 0], 0)
    pen_t = to_table(pen_cam[None])[0]
    radius = 0.006
    # The mask centroid sits inside the thin object: rest the pen on the support under it.
    pen_t[2] = radius
    for o in objects:
        if np.linalg.norm(pen_t[:2] - np.array(o["centre_ab"])) < 0.6 * max(o["size_ab"]):
            pen_t[2] = max(pen_t[2], o["top_h"] + radius)
    pen = {
        "start_abh": pen_t.tolist(),
        "length_m": 0.14,
        "radius_m": radius,
        "hold_s": rj["perceived_object"]["hold_s"] if "hold_s" in rj["perceived_object"] else rj["hold_s"],
    }

    # Walls: two dominant planes among depth points that are neither table nor objects.
    d = pts["depth_small"]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    fs = f * w / W
    cloud = np.stack([(u + 0.5 - w / 2) * d / fs, (v + 0.5 - h / 2) * d / fs, d], -1).reshape(-1, 3) * s
    height = cloud @ B[2] - plane_off
    rest = cloud[(height > 0.05) & np.isfinite(cloud).all(1) & (cloud[:, 2] < 3.0)]
    walls = []
    for k in range(2):
        n, off, inl = ransac_plane(rest, seed=k)
        if n @ rest[inl].mean(0) > 0:
            n, off = -n, -off  # normals face the camera
        walls.append(
            {
                "normal": n.tolist(),
                "offset": off,
                "centre": rest[inl].mean(0).tolist(),
                "inliers": int(inl.sum()),
            }
        )
        rest = rest[~inl]

    # Colours sampled from the real first frame.
    img = _frame(clip / "video.mp4", 0)
    vv, uu = np.mgrid[0:h, 0:w]
    px = (
        np.clip(((uu + 0.5) * W / w).astype(int), 0, W - 1),
        np.clip(((vv + 0.5) * H / h).astype(int), 0, H - 1),
    )
    colour = img[px[1], px[0]].reshape(-1, 3) / 255.0
    cloud_all = cloud.reshape(-1, 3)
    ok_all = np.isfinite(cloud_all).all(1)
    table_mask = ok_all & (np.abs(cloud_all @ B[2] - plane_off) < 0.015)
    table_rgb = np.median(colour[table_mask], 0)
    for wl in walls:
        m = ok_all & (np.abs((cloud_all - np.array(wl["centre"])) @ np.array(wl["normal"])) < 0.03)
        wl["rgb"] = np.median(colour[m], 0).tolist() if m.sum() else [0.8, 0.75, 0.65]

    hands = np.load(clip / "hands_fah_f874.npz")
    # EgoDex camera poses are used only to replay the head motion in this visual comparison (frame ``key`` = world).
    Tw = np.load(clip / "gt" / "egodex_gt.npz")["T_world_camera"]
    cam_T = np.linalg.inv(Tw[inv["frame"]]) @ Tw
    np.savez_compressed(out / "hands.npz", joints3d=hands["joints3d"], valid=hands["valid"], cam_T=cam_T)
    meta = {
        "focal_px": f,
        "image_wh": [W, H],
        "scale": s,
        "basis": B.tolist(),
        "plane_off": plane_off,
        "table_rgb": table_rgb.tolist(),
        "objects": objects,
        "pen": pen,
        "walls": walls,
        "fps": 30.0,
    }
    (out / "scene_cam.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {len(objects)} objects, {len(walls)} walls, depth_scale {s:.3f}")
    for o in objects:
        print(
            f"  {o['name']:22s} ab={np.round(o['centre_ab'], 3)} size={np.round(o['size_ab'], 3)} top={o['top_h']:.3f}"
        )


if __name__ == "__main__":
    main()
