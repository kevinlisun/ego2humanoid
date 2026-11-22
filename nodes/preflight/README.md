# `nodes/preflight/` — contract-path preflight image

`Dockerfile` of the `ego2humanoid` command-line image used by the contract path (`python -m cli.main preflight`). It validates an episode JSON and an RGB video; it is not part of the real-clip pipeline.

* Base `python:3.12-slim` plus `ffmpeg` and `uv`; copies `pyproject.toml`, `uv.lock`, `README.md` and `src/`.
* `uv sync --frozen --no-dev` into `/opt/venv`; no `sim`/`video` extras, no GPU, no weights.
* `ENTRYPOINT ["ego2humanoid"]`.

```bash
scripts/build_node.sh preflight            # from the repository root; tag ego2humanoid/preflight:dev
scripts/smoke_node.sh preflight            # runs the image with --help
```
