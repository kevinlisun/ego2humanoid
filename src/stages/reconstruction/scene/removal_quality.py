"""Object-removal quality gate for SimFoundry stage 5 (T6S.2e).

Per iteration: pixels outside the (dilated) mask must stay unchanged, and pixels inside
the mask must change. A residual score measures how much of the masked area still looks
like the removed object (colour distance to the pre-removal masked mean).
"""

from pathlib import Path

import cv2
import numpy as np


def _gray_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def removal_metrics(
    before: np.ndarray, after: np.ndarray, mask: np.ndarray, dilate_px: int = 15
) -> dict[str, float]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    near = cv2.dilate(mask.astype(np.uint8), k) > 0
    outside = ~near
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).mean(-1)
    changed_inside = float((diff[mask] > 12).mean()) if mask.any() else 0.0
    outside_change = float(diff[outside].mean()) if outside.any() else 0.0
    obj_color = before[mask].mean(0) if mask.any() else np.zeros(3)
    dist = np.linalg.norm(after.astype(np.float32) - obj_color, axis=-1)
    residual = float((dist[mask] < 40).mean()) if mask.any() else 0.0
    return {
        "mask_area_frac": float(mask.mean()),
        "changed_inside_frac": changed_inside,
        "outside_mean_abs_diff": outside_change,
        "residual_object_color_frac": residual,
    }


def passes(m: dict[str, float]) -> bool:
    return (
        m["changed_inside_frac"] >= 0.8
        and m["outside_mean_abs_diff"] <= 3.0
        and m["residual_object_color_frac"] <= 0.25
    )


def build_report(s5: Path) -> dict:
    items = {}
    for pre in sorted((s5 / "pre_object_removal").glob("iter_[0-9].png")):
        name = pre.stem
        post, mask_f = s5 / "post_object_removal" / pre.name, s5 / "removal_mask" / pre.name
        if not (post.exists() and mask_f.exists()):
            continue
        # pre_object_removal holds the green-painted prompt image, so compare against the clean
        # input of this iteration: the original frame, or the previous iteration's output.
        k = int(name.split("_")[1])
        clean = s5 / "post_object_removal" / f"iter_{k - 1}.png" if k else s5 / "source_padded_resized.png"
        b, a = cv2.imread(str(clean)), cv2.imread(str(post))
        if b.shape != a.shape:
            a = cv2.resize(a, (b.shape[1], b.shape[0]))
        m = removal_metrics(b, a, _gray_mask(mask_f, b.shape[:2]))
        items[name] = {**m, "pass": passes(m)}
    return {"iterations": items, "all_pass": bool(items) and all(v["pass"] for v in items.values())}
