# `nodes/retarget/` — export, retargeting, simulation and rendering image

Image for stages `export` and `retarget` (`scripts/retarget_estimated.py`), incl. optional Blender [[Blender, 2024](../../docs/REFERENCES.md#blender)] render.

- `/opt/venv` (py3.12): root `pyproject.toml`/`uv.lock` (extras video sim hands objects egodex + dev), MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)] rebuilt from `third_party/mujoco`. No node-specific lock: deps equal the root lock.
- `/opt/bpy` (py3.11): bpy 5.0.1, numpy 1.26.4, trimesh (`bpy-requirements.txt`, from the working /tmp/bpy_venv). Exposed as `BPY_PY` (and `BPY_PYTHON`, which `retarget_estimated.py` actually reads).
- EGL headless (`MUJOCO_GL=egl`); no weights/data baked in. Mount repo at `/app` (optional) and data at `/app/data` (resolve the `data/derived` symlink); `HF_HOME=/hf`.

```
scripts/build_node.sh retarget
docker run --rm --gpus all -v $PWD:/app -v $(readlink -f data/derived):/app/data/derived \
  -e RENDER=0 ego2humanoid/retarget:dev python scripts/retarget_estimated.py <clip>
```

Notes: in worktrees where `third_party/mujoco` is a tree of symlinks, build with
`docker build --build-context mujoco-src=$(readlink -f <real mujoco dir>) -f nodes/retarget/Dockerfile -t ego2humanoid/retarget:dev .`.
`tests/`, `scripts/`, `third_party/` are not baked in: mount them (and symlink targets) at run time.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-blender"></span>[Blender, 2024](../../docs/REFERENCES.md#blender). *Blender — a 3D modelling and rendering package.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
