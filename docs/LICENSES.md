# License register

The Ego2Humanoid code itself is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). The table below lists third-party components, which keep their own licences.

Download and use were reviewed and approved on 2026-11-10 (open question Q6 in `TASKS.md`) for **non-commercial research** only. Gated assets require each user's own account approval; nothing here is committed to git except pinned submodule SHAs.

| Component | Use | License | Status |
|---|---|---|---|
| FAH code [[Cao, 2026](REFERENCES.md#cao2026fah)] | hand estimation | Apache-2.0 | submodule pinned |
| EgoTac code [[Zhang et al., 2026](REFERENCES.md#zhang2026egotac)] | tactile prediction | no licence file in the repo (research use only until clarified) | submodule pinned |
| EgoTac weights | tactile prediction | Apache-2.0 (HF model card, not gated) | `data/weights/egotac` (not committed) |
| FAH weights | hand estimation | `third_party/FAH/WEIGHTS_LICENSE` (non-commercial) | not downloaded |
| WiLoR [[Potamias et al., 2025](REFERENCES.md#potamias2025wilor)] (code + weights) | hand estimation (default backend) | CC-BY-NC-ND 4.0 | submodule pinned (`third_party/WiLoR`); weights downloaded to `third_party/WiLoR/pretrained_models/` (not committed) |
| MoGe-2 [[Wang et al., 2025b](REFERENCES.md#wang2025moge2)] | focal, gravity, point maps | MIT (code and weights) | weights from Hugging Face into `$HF_HOME` |
| OWLv2 [[Minderer et al., 2023](REFERENCES.md#minderer2023owlv2)] + SAM [[Kirillov et al., 2023](REFERENCES.md#kirillov2023sam)] | object detection fallback | Apache-2.0 | weights from Hugging Face into `$HF_HOME` |
| Hunyuan3D-2mini [[Hunyuan3D Team, 2025](REFERENCES.md#hunyuan2025hunyuan3d2)] | object mesh generation | Tencent Hunyuan 3D 2.0 Community License; **the licence does not apply in the EU, the UK or South Korea** | `data/weights/hunyuan3d` (not committed); users in excluded territories must use another image-to-3D model or obtain a licence from Tencent |
| CoACD [[Wei et al., 2022](REFERENCES.md#wei2022coacd)] | convex decomposition of collision meshes | MIT | PyPI dependency (`objects` extra) |
| Blender [[Blender, 2024](REFERENCES.md#blender)] | review rendering (`bpy`) | GPL-3.0-or-later (tool, not linked into this code) | installed in the `retarget` node image |
| SPIDER [[Pan et al., 2025](REFERENCES.md#pan2025spider)] | retargeting | CC-BY-NC 4.0 | submodule pinned |
| SimFoundry [[NVIDIA GEAR, 2026](REFERENCES.md#simfoundry2026)] | scene reconstruction | Apache-2.0 (+ third-party notices) | submodule pinned; no external API calls; relay not approved for frames |
| GR00T-WholeBodyControl (Decoupled WBC) [[NVIDIA et al., 2025](REFERENCES.md#nvidia2025gr00tn1)] | track L whole-body controller (legs + waist ONNX policies) | code Apache-2.0; weights NVIDIA Open Model License | submodule pinned (`third_party/GR00T-WholeBodyControl`); the two small ONNX policies are read from the submodule (`src/robot/dwbc.py`); SONIC/GEAR checkpoints (HF `nvidia/GEAR-SONIC`) are not used |
| Unitree unitree_ros G1+Dex3 MJCF | robot model | BSD-3-Clause | fetched by `scripts/fetch_g1_assets.sh` |
| MuJoCo [[Todorov et al., 2012](REFERENCES.md#todorov2012mujoco)] (source build for `nodes/retarget`) | physics engine | Apache-2.0 | submodule pinned (`third_party/mujoco`) |
| MuJoCo Menagerie [[Zakka et al., 2022](REFERENCES.md#zakka2022menagerie)] `sharpa_wave` | five-finger tactile hand model | Apache-2.0 | pinned to `mujoco_menagerie@370c4396`; MJCF tracked, meshes fetched by `scripts/fetch_sharpa_assets.sh` |
| MANO [[Romero et al., 2017](REFERENCES.md#romero2017mano)] | hand model | MPI registration required | user action: https://mano.is.tue.mpg.de |
| HOT3D [[Banerjee et al., 2025](REFERENCES.md#banerjee2025hot3d)] | dataset | dropped (2026-11-10) | not used |
| EgoDex [[Hoque et al., 2025](REFERENCES.md#hoque2025egodex)] | dataset | CC-BY-NC-ND, public download | https://github.com/apple/ml-egodex ; research only, never redistribute data or derivatives; stays under gitignored `data/` |
| SAM 3 [[Carion et al., 2026](REFERENCES.md#carion2026sam3)] / SAM 3D Body [[Yang et al., 2026](REFERENCES.md#yang2026sam3dbody)] / DINOv3 [[Siméoni et al., 2025](REFERENCES.md#simeoni2025dinov3)] | FAH dependencies | gated on Hugging Face / Meta terms | user action: request access, see links in `third_party/FAH/README.md` |
| FLUX.1-Kontext-dev [[Black Forest Labs, 2025](REFERENCES.md#batifol2025flux)] | SimFoundry only | non-commercial, gated | user action, optional |

Rule: never commit weights, MANO files, or HOT3D data (`HARNESS.md` §5).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](REFERENCES.md).

- <span id="cite-banerjee2025hot3d"></span>[Banerjee et al., 2025](REFERENCES.md#banerjee2025hot3d). *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos.*
- <span id="cite-batifol2025flux"></span>[Black Forest Labs, 2025](REFERENCES.md#batifol2025flux). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing.*
- <span id="cite-blender"></span>[Blender, 2024](REFERENCES.md#blender). *Blender — a 3D modelling and rendering package.*
- <span id="cite-cao2026fah"></span>[Cao, 2026](REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-hoque2025egodex"></span>[Hoque et al., 2025](REFERENCES.md#hoque2025egodex). *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.*
- <span id="cite-hunyuan2025hunyuan3d2"></span>[Hunyuan3D Team, 2025](REFERENCES.md#hunyuan2025hunyuan3d2). *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation.*
- <span id="cite-kirillov2023sam"></span>[Kirillov et al., 2023](REFERENCES.md#kirillov2023sam). *Segment Anything.*
- <span id="cite-minderer2023owlv2"></span>[Minderer et al., 2023](REFERENCES.md#minderer2023owlv2). *Scaling Open-Vocabulary Object Detection (OWLv2).*
- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
- <span id="cite-nvidia2025gr00tn1"></span>[NVIDIA et al., 2025](REFERENCES.md#nvidia2025gr00tn1). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*
- <span id="cite-pan2025spider"></span>[Pan et al., 2025](REFERENCES.md#pan2025spider). *SPIDER: Scalable Physics-Informed Dexterous Retargeting.*
- <span id="cite-potamias2025wilor"></span>[Potamias et al., 2025](REFERENCES.md#potamias2025wilor). *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.*
- <span id="cite-romero2017mano"></span>[Romero et al., 2017](REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
- <span id="cite-simeoni2025dinov3"></span>[Siméoni et al., 2025](REFERENCES.md#simeoni2025dinov3). *DINOv3.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-wei2022coacd"></span>[Wei et al., 2022](REFERENCES.md#wei2022coacd). *Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search (CoACD).*
- <span id="cite-yang2026sam3dbody"></span>[Yang et al., 2026](REFERENCES.md#yang2026sam3dbody). *SAM 3D Body: Robust Full-Body Human Mesh Recovery.*
- <span id="cite-zakka2022menagerie"></span>[Zakka et al., 2022](REFERENCES.md#zakka2022menagerie). *MuJoCo Menagerie.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
