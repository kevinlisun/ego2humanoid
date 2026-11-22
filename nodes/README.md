# `nodes/` — one Docker image per pipeline stage

One Docker image definition per pipeline stage group, so each model stack keeps its own Python and CUDA pins.
Build from the repository root with `scripts/build_node.sh <node> [tag]` (default tag `ego2humanoid/<node>:dev`).
`DOCKER=1 scripts/run_pipeline.sh ...` runs every stage in its image through `nodes/bin/`.

| Node | Used for | Details |
|---|---|---|
| `perception/` | `intrinsics`, `gravity`, `objects`, `scene`, `scene_mesh`, `held_mesh` (MoGe-2 [[Wang et al., 2025b](../docs/REFERENCES.md#wang2025moge2)], SAM 3 [[Carion et al., 2026](../docs/REFERENCES.md#carion2026sam3)] / OWLv2 [[Minderer et al., 2023](../docs/REFERENCES.md#minderer2023owlv2)], Hunyuan3D-2 [[Hunyuan3D Team, 2025](../docs/REFERENCES.md#hunyuan2025hunyuan3d2)]) | `perception/README.md`; has its own `pyproject.toml` and `uv.lock` |
| `hands/` | `hands`: FAH [[Cao, 2026](../docs/REFERENCES.md#cao2026fah)] (`requirements-fah.txt`) and the WiLoR [[Potamias et al., 2025](../docs/REFERENCES.md#potamias2025wilor)] fallback (`--target wilor`, `requirements-wilor.txt`) | `hands/README.md`; WiLoR image: `docker build -f nodes/hands/Dockerfile --target wilor -t ego2humanoid/hands-wilor:dev .` |
| `tactile/` | `tactile`: EgoTac [[Zhang et al., 2026](../docs/REFERENCES.md#zhang2026egotac)] | `tactile/README.md` |
| `retarget/` | `export` and `retarget` (`scripts/retarget_estimated.py`, MuJoCo [[Todorov et al., 2012](../docs/REFERENCES.md#todorov2012mujoco)], optional Blender [[Blender, 2024](../docs/REFERENCES.md#blender)] render) | `retarget/README.md` |
| `preflight/` | CLI image (`ego2humanoid`) for the stage-CLI contract path, `python:3.12-slim` + ffmpeg | `Dockerfile` only; smoke test `scripts/smoke_node.sh preflight` |
| `replay/` | Same CLI with the `sim` and `video` extras, EGL headless MuJoCo, `configs/` and `assets/` baked in | `Dockerfile` only; smoke test `scripts/smoke_node.sh replay` |
| `bin/` | `<node>-python` wrappers (symlinks to `scripts/node_python.sh`) | `bin/README.md` |

Weights, data and third-party checkouts are not baked in; they are mounted at run time (`scripts/node_python.sh`).
`preflight` and `replay` belong to the synthetic contract path (`src/cli/main.py`); the real-clip pipeline uses the
other four nodes.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../docs/REFERENCES.md).

- <span id="cite-blender"></span>[Blender, 2024](../docs/REFERENCES.md#blender). *Blender — a 3D modelling and rendering package.*
- <span id="cite-cao2026fah"></span>[Cao, 2026](../docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](../docs/REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-hunyuan2025hunyuan3d2"></span>[Hunyuan3D Team, 2025](../docs/REFERENCES.md#hunyuan2025hunyuan3d2). *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation.*
- <span id="cite-minderer2023owlv2"></span>[Minderer et al., 2023](../docs/REFERENCES.md#minderer2023owlv2). *Scaling Open-Vocabulary Object Detection (OWLv2).*
- <span id="cite-potamias2025wilor"></span>[Potamias et al., 2025](../docs/REFERENCES.md#potamias2025wilor). *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](../docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
