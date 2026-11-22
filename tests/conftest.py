import os
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def pytest_collection_modifyitems(config, items):
    has_data = any(p for p in DATA_DIR.glob("*") if p.name != "README.md")
    skip_data = pytest.mark.skip(reason="real data under data/ is missing")
    for item in items:
        if ("realdata" in item.keywords or "system" in item.keywords) and not has_data:
            item.add_marker(skip_data)
