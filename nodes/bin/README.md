# `nodes/bin/` — per-node Python launchers

Five symlinks, all to `../../scripts/node_python.sh`:

`perception-python`, `hands-python`, `hands-wilor-python`, `tactile-python`, `retarget-python`.

`node_python.sh` takes the node name from the name it was invoked as (`<node>-python`) and runs
`docker run --rm --gpus all ... --entrypoint python ego2humanoid/<node>:$IMAGE_TAG "$@"`. So
`nodes/bin/retarget-python scripts/retarget_estimated.py <clip>` behaves like `python scripts/retarget_estimated.py <clip>`
inside the retarget image.

With `DOCKER=1`, `scripts/run_pipeline.sh` sets `PERC_PY`, `FAH_PY` (unless you set it), `WILOR_PY`, `EGOTAC_PY` and the
retarget runner to these wrappers.

Behaviour of the wrapper:

* The repo, `$HF_HOME` (default `/mnt/tmp/hfcache`) and the targets of symlinks under `data/` are mounted at their
  host paths; `PYTHONPATH` is `<repo>/src`.
* Variables starting with `E2H_`, `SIM_`, `SCENE`, `OBJ_`, `HY3DGEN`, `FAH_`, `WILOR_`, `RENDER` and `HAND`, `MUJOCO_GL`,
  `PYOPENGL_PLATFORM`, `PYTHONUNBUFFERED` are forwarded.
* `hands`: mounts FAH code (`FAH_SRC`), weights (`FAH_WEIGHTS_SRC` at `/fah_weights`) and MANO (`MANO_SRC`).
  `hands-wilor`: mounts `WILOR_SRC`. `tactile`: mounts `EGOTAC_SRC` with an empty `PYTHONPATH`.
  `retarget`: mounts the symlink targets of `third_party/*` read-only.
* `IMAGE_TAG` (default `dev`) and `DOCKER_ARGS` customise the image tag and extra `docker run` flags.

Build the images first with `scripts/build_node.sh <node>`.
