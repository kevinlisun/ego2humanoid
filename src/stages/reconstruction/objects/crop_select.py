"""Choose the cleanest object mask among candidate views for the held-object crop."""

import cv2
import numpy as np

MAX_FRAME_FRACTION = 0.1  # a mask covering more of the frame is a segmentation failure
MIN_COMPONENT_FRACTION = 0.05  # components smaller than this share of the mask are speckle


def is_fragmented(mask: np.ndarray) -> bool:
    """True when the mask has more than one significant connected component (occluded object)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas >= MIN_COMPONENT_FRACTION * areas.sum()).sum()) > 1 if n > 2 else False


def select_crop_mask(masks: list[np.ndarray], inventory_area_px: int | None = None) -> int | None:
    """Index of the largest whole mask, or ``None`` when every candidate is rejected.

    Rejected: empty, larger than ``MAX_FRAME_FRACTION`` of the frame, fragmented, or (when the
    clean inventory view gives ``inventory_area_px``) below half of that area.
    """
    best, best_area = None, 0
    for i, m in enumerate(masks):
        area = int(m.sum())
        if area == 0 or area > MAX_FRAME_FRACTION * m.size or is_fragmented(m):
            continue
        if inventory_area_px is not None and area < 0.5 * inventory_area_px:
            continue
        if area > best_area:
            best, best_area = i, area
    return best
