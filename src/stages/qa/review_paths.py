"""Where review artefacts live, and how stale ones are removed before a re-render.

Small deliverables (the grid and per-panel videos shown to the owner) stay next to the clip in
``<clip>/review``. The large intermediates (cached RGB-D captures, per-panel videos) go to a bulk
volume, ``/cloudfile/nvme_tmp/egodex/<clip>/panels`` by default, so ``data/`` stays small. Override
the root with ``E2H_BULK_DIR``; if it cannot be created the panels fall back to ``<clip>/review/panels``.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BULK_ROOT = "/cloudfile/nvme_tmp/egodex"

_REVIEW_STEMS = ("sim_mujoco", "trajectory", "ego_plate_composite", "result_grid", "side_by_side")
_PANEL_NAMES = ("bg_rgbd.npz", "ego_d435.npz", "tl.mp4", "tr.mp4", "br.mp4")


def panels_dir(clip: Path) -> Path:
    """Directory for the large per-hand panel captures of ``clip`` (created if needed)."""
    root = Path(os.environ.get("E2H_BULK_DIR", DEFAULT_BULK_ROOT))
    target = root / Path(clip).resolve().name / "panels"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback = Path(clip) / "review" / "panels"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def existing_panels_dir(clip: Path) -> Path:
    """Where panels of ``clip`` are cached now: the bulk volume, else the legacy local folder."""
    root = Path(os.environ.get("E2H_BULK_DIR", DEFAULT_BULK_ROOT))
    bulk = root / Path(clip).resolve().name / "panels"
    return bulk if bulk.exists() else Path(clip) / "review" / "panels"


def clean_review(clip: Path, suffix: str, extra_files: tuple[Path, ...] = ()) -> list[Path]:
    """Delete every artefact a render of ``clip`` for ``suffix`` is about to rewrite.

    Only this hand's files are touched (the other hand's grid and panels survive). Returns the
    removed paths.
    """
    clip = Path(clip)
    names = [f"{stem}{suffix}.mp4" for stem in _REVIEW_STEMS if stem != "trajectory"]
    names.append(f"trajectory{suffix}.npz")
    victims = [clip / "review" / n for n in names]
    for folder in {panels_dir(clip), clip / "review" / "panels"}:
        for n in _PANEL_NAMES:
            stem, ext = n.rsplit(".", 1)
            victims.append(folder / f"{stem}{suffix}.{ext}")
    victims.extend(extra_files)
    removed = []
    for path in victims:
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed
