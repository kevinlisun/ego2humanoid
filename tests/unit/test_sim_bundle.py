"""Bundle round-trip on the real clip-104 bundle when present, else skipped."""

from pathlib import Path

import numpy as np
import pytest

from stages.manipulation.bundle import load_bundle

CLIP = Path("data/derived/egodex/basic_pick_place_104/real2sim_sharpa")


@pytest.mark.skipif(not (CLIP / "sim_bundle.json").exists(), reason="no bundle")
def test_load_bundle_roundtrip():
    b = load_bundle(CLIP)
    assert b.case.q_target_rad.shape[0] == b.case.timestamps_s.shape[0]
    assert np.isfinite(b.case.q_target_rad).all()
