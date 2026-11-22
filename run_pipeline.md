# Run the data pipeline in one command

`scripts/run_pipeline.sh` runs the whole real2sim chain for one or more EgoDex clips:

```text
EgoDex hdf5 + mp4
  -> export          episode.json, video.mp4, gt/            (uv venv)
  -> intrinsics      intrinsics_est.json   (MoGe-2 focal)    (perception venv)
  -> gravity         gravity_est.json      (MoGe-2 normals)  (perception venv)
  -> hands           hands_fah_f<focal>.npz (FAH)            (FAH venv)
  -> tactile         tactile_egotac.npz    (optional)        (EgoTac venv)
  -> objects         objects_sam3.npz      (SAM 3, OWLv2 fallback)
  -> scene           scene/inventory.json, points.npz, crop_*.png
  -> scene_mesh      scene/mesh_*.glb      (Hunyuan3D-2mini)
  -> held_mesh       object_mesh_raw.glb   (first retarget pass -> SAM 3 crop -> Hunyuan3D)
  -> retarget        retarget_rgb_<hand>.{json,npz}, review/result_grid_<hand>.mp4  (uv venv, MuJoCo)
```

The models are MoGe-2 [[Wang et al., 2025b](docs/REFERENCES.md#wang2025moge2)], FAH [[Cao, 2026](docs/REFERENCES.md#cao2026fah)] (or WiLoR [[Potamias et al., 2025](docs/REFERENCES.md#potamias2025wilor)]), EgoTac [[Zhang et al., 2026](docs/REFERENCES.md#zhang2026egotac)],
SAM 3 [[Carion et al., 2026](docs/REFERENCES.md#carion2026sam3)] (or OWLv2 [[Minderer et al., 2023](docs/REFERENCES.md#minderer2023owlv2)]), Hunyuan3D-2mini [[Hunyuan3D Team, 2025](docs/REFERENCES.md#hunyuan2025hunyuan3d2)] and MuJoCo
[[Todorov et al., 2012](docs/REFERENCES.md#todorov2012mujoco)]; the design is described in `docs/TECH_DESIGN.md`.

Each stage is skipped when its output already exists, so the same command resumes an interrupted run or
re-runs only the robot part on clips that are already perceived.

## Quick start

```bash
# all 10 clips, Sharpa, rendered, 4 in parallel (the robot part is CPU-bound; 4 fit on the T4 box)
JOBS=4 scripts/run_pipeline.sh 0 1 10 100 101 104 108 114 116 117

# one clip, robot part only (perception outputs already present)
STAGES=retarget scripts/run_pipeline.sh 0

# print the commands without running them
DRY_RUN=1 scripts/run_pipeline.sh 0
```

Progress bar for a running batch: `LOG_DIR=<dir> watch -n 30 scripts/pipeline_progress.sh 0 1 10 100`.

The per-clip log is `$LOG_DIR/<hand>_<clip>.log` (default `/tmp/e2h_pipeline`). Each clip ends with a
`result:` line (success, failure reasons, place error). The batch ends with the metric table
`$LOG_DIR/batch_results.md` from `scripts/summarize_batch.py`.

Outputs:

* Per-clip outputs: `data/derived/egodex/basic_pick_place_<clip>/` (`data/README.md`).
* Review video: `review/result_grid_<hand>.mp4` (2x2: RGB with hands, third-person robot 16:9, ego Blender 16:9, trajectories).
* Bulky render panels: `/cloudfile/nvme_tmp/egodex/<clip>/panels` (`E2H_BULK_DIR`).

Every rendered run deletes that hand's previous review files first.

## Your own video (no EgoDex hdf5)

Only `video.mp4` and `episode.json` are needed; the `gt/` folder is evaluation-only. With `TASK=loco_manip`
and id `allinone` the clip dir is `data/derived/egodex/loco_manip_allinone/`:

```bash
# put video.mp4 and episode.json (task_description drives the object prompt) in the clip dir, then:
TASK=loco_manip STAGES=intrinsics,gravity,objects,scene OBJ_DETECTOR=owl scripts/run_pipeline.sh allinone
```

The `hands` stage needs the FAH venv, the FAH weights (public, `noahcao/FAH`), MANO (`assets/mano`) and the
gated SAM 3D Body `model_config.yaml` (needs an `HF_TOKEN` with the SAM licence accepted). `held_mesh` needs
SAM 3 (also gated). Both are missing on `devbox-t4` as of 2026-11-17, so `retarget` and loco-manipulation
cannot run on a new video yet.

## Options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `HAND` | `sharpa` | `sharpa` or `dex3` |
| `RENDER` | `1` | `1` renders the 2x2 review video; `0` is about twice as fast (use for sweeps) |
| `JOBS` | `1` | clips processed in parallel |
| `STAGES` | all | comma list, e.g. `scene,scene_mesh,retarget` |
| `FORCE` | `0` | `1` reruns stages whose outputs exist |
| `DRY_RUN` | `0` | `1` only prints the commands |
| `PERC_PY` | `/tmp/venvs/perc/bin/python` | MoGe, SAM 3 / OWLv2, Hunyuan3D venv |
| `HF_HOME` | unset | Hugging Face cache holding `facebook/sam3` and the MoGe weights. Exported to every stage; the SAM3 stages fail fast with `WEIGHTS_NOT_FOUND: ... set HF_HOME` when it is wrong (`objects` falls back to OWLv2, `held_mesh` stops). In a container mount it and pass it: `-v $HF_HOME:$HF_HOME -e HF_HOME` |
| `FAH_PY` | `/tmp/venvs/fah/bin/python` | FAH venv (also needs `FAH_HOME`, `FAH_SAM3D`, `MANO_DIR`) |
| `WILOR_PY` | `/tmp/venvs/wilor/bin/python` | WiLoR venv (public weights); used for `hands` when `FAH_PY` does not exist |
| `EGOTAC_PY` | unset | EgoTac venv; the tactile stage is skipped when unset |
| `DOCKER` | `0` | `1` runs each stage in its node image (`nodes/<node>/`, built with `scripts/build_node.sh <node>`) through `nodes/bin/<node>-python` (`scripts/node_python.sh`); sets the `*_PY` variables and the retarget runner. `E2H_*`, `SCENE*`, `HAND`, `RENDER`, `OBJ_*` variables are forwarded into the container |
| `FAH_SRC`, `FAH_WEIGHTS_SRC`, `MANO_SRC` | `<repo>/third_party/FAH`, `<repo>/data/weights/fah`, `<repo>/assets/mano/clean` | `DOCKER=1`, `hands` stage only: FAH code, FAH weights (`hf download noahcao/FAH --exclude "qc/*"`) and MANO, mounted into the container. The gated SAM 3D Body `model_config.yaml` still needs `HF_TOKEN`. To use WiLoR instead, set `FAH_PY=/nonexistent` |
| `OBJ_DETECTOR` | `sam3` | `sam3` (falls back to `owl` on error) or `owl` |
| `RAW_DIR`, `DERIVED_DIR`, `LOG_DIR`, `TASK` | `data/raw/egodex/test`, `data/derived/egodex`, `/tmp/e2h_pipeline`, `basic_pick_place` | paths |

The pipeline also honours every knob of `scripts/retarget_estimated.py`, because the environment is
passed through. Examples:

* `E2H_RETARGET_CACHE=<dir>` reuses the placement retarget across sweeps. Delete the cache after
  changing the placement or IK code.
* `E2H_SCENE_OBJECTS=0` turns off the tabletop scene objects.
* `E2H_DUMP=<file.pkl>` dumps the trajectory for offline debugging. Use it with a single clip only.
* Physics and plan constants are `SimParams` leaves (`configs/sim_params/default.yaml`, `E2H_SIM_PARAMS`); the place-aware placement retry uses `E2H_PLACE_GOAL` internally (TECH_DESIGN §4.7), do not set it by hand.

`SCENE_DIR=data/scenes/egodex_<clip>` is set automatically when a SimFoundry scene exists (clip 104 only). The clip's
`scene/inventory.json` objects are simulated either way; `retarget_estimated.py` ignores `SCENE_DIR` when an inventory exists.

## Prerequisites

With `DOCKER=1` the venvs below are replaced by the per-stage node images (build each with `scripts/build_node.sh <node>`: `perception`, `hands`, `tactile`, `retarget`); only Docker with `--gpus`, the repo and `data/` are needed.

1. **Repo venv**: `uv sync --frozen --extra video --extra sim --extra hands --extra objects --extra egodex`
   (`docs/DEVBOX.md`).
2. **Raw clips**, once:
   `uv run --frozen --extra egodex python scripts/fetch_egodex.py basic_pick_place -n 12`
   writes `data/raw/egodex/test/basic_pick_place/<id>.{hdf5,mp4}`. The data is CC-BY-NC-ND, so never
   commit or redistribute it.
3. **Perception venv** (`PERC_PY`): torch, `moge`, `transformers>=5`, `hy3dgen`.
   * Hunyuan3D weights go in `data/weights/hunyuan3d/Hunyuan3D-2mini`.
   * SAM 3 (`facebook/sam3`) is gated on the Hub. Without an accepted licence and an `HF_TOKEN`, the
     `objects` stage falls back to OWLv2 + SAM. The held-object crop (`held_mesh`) needs SAM 3.
4. **FAH venv** (`FAH_PY`), needed only for the `hands` stage (TECH_DESIGN §7, TASKS T1.8); WiLoR (`WILOR_PY`) is the public-weights fallback.
5. **EgoTac venv** (`EGOTAC_PY`), optional. Weights: `data/weights/egotac/egotac_base_h4.pt`.

State on `devbox-t4` (2026-11-17):

* `/tmp/venvs/perc` works. `export`, `intrinsics`, `gravity`, `objects` (via OWLv2) and `scene` were
  re-run from the raw clip 0 into a scratch dir and reproduced the stored outputs: focal 782.2 vs
  782.0 px, the same gravity vector, the same 13-object inventory.
* The FAH and EgoTac venvs are not installed on this box. All 10 clips already have their hand and
  tactile files, so the default command skips those stages.

## Timing (T4 box, Sharpa, rendered)

* Perception: a few minutes per clip on the GPU.
* Robot part: about 5 to 27 min per clip, depending on clip length and the number of
  placement-search retargets (clip 117 is the slowest). About half of that is rendering.
* Every run logs `[time] <stage>` lines: `grep '^\[time\]' $LOG_DIR/*.log`.

### Locomanipulation on a video-only clip

After the retarget stage, `uv run --extra sim --extra video --extra hands --extra objects --extra egodex python scripts/loco_manip_episode.py --clip <clip_dir_name> --jobs 1`
writes `loco_manip_result.json` and `review/loco_manip.mp4` into the clip dir. Clips without tactile timing use the retarget's hold window.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](docs/REFERENCES.md).

- <span id="cite-cao2026fah"></span>[Cao, 2026](docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](docs/REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-hunyuan2025hunyuan3d2"></span>[Hunyuan3D Team, 2025](docs/REFERENCES.md#hunyuan2025hunyuan3d2). *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation.*
- <span id="cite-minderer2023owlv2"></span>[Minderer et al., 2023](docs/REFERENCES.md#minderer2023owlv2). *Scaling Open-Vocabulary Object Detection (OWLv2).*
- <span id="cite-potamias2025wilor"></span>[Potamias et al., 2025](docs/REFERENCES.md#potamias2025wilor). *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
