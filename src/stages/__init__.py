"""Headless rendering: MuJoCo reads MUJOCO_GL on first import, so default it here."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
