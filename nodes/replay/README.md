# `nodes/replay/` — contract-path physics-replay image

`Dockerfile` of the headless physics-replay image for the contract path (`ego2humanoid synthetic-run`). It runs the synthetic scripted pick-place through MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)] and QA; it is not the
real-clip `retarget` stage (that is `nodes/retarget`).

* Base `python:3.12-slim` plus `ffmpeg` and the EGL/GL libraries; `MUJOCO_GL=egl`.
* Copies `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `configs/` and `assets/` (the robot meshes must be fetched
  before the build, see `assets/robots/README.md`).
* `uv sync --frozen --no-dev --extra sim --extra video` into `/opt/venv`.
* `ENTRYPOINT ["ego2humanoid"]`.

```bash
scripts/build_node.sh replay               # tag ego2humanoid/replay:dev
scripts/smoke_node.sh replay               # synthetic-run; checks replay/render.mp4 and replay/qa_report.json
```

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
