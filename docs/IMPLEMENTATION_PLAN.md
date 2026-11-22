# Implementation Status — Ego2Humanoid

Task-level tracking lives in `TASKS.md`; the design as built is `docs/TECH_DESIGN.md`. The
original v0.3 milestone plan (M0–M7) is archived in `docs/archive/IMPLEMENTATION_PLAN_v0.3.md`.
This file records where each milestone actually ended up.

| Milestone (v0.3 plan) | Status | What was built instead / evidence |
|---|---|---|
| M0 Foundations | ✅ partly | uv project, flat `src/` layout (`common`, `contracts`, `egodex`, `robot`, `rsi`, `skills`, `evaluation`, `orchestration`, `cli`, `synthetic`, `stages`), contracts v1.2, NPZ validators, synthetic generators, local runner. Node Dockerfiles for `preflight`/`replay` (contract path) and per-stage images `perception`, `hands`, `tactile`, `retarget` (`DOCKER=1 scripts/run_pipeline.sh`). |
| M1 HOT3D [[Banerjee et al., 2025](REFERENCES.md#banerjee2025hot3d)] + oracle | ⛔ replaced | EgoDex reader, selector and export (`src/egodex`); no oracle mode. |
| M2 Robot + MuJoCo | ✅ | G1+Dex3 and G1+Sharpa Wave MJCF, fixed pelvis, PD replay with the injection guard, QA events, EGL rendering; scripted pick-place fixture (smoke test). |
| M3 Align + retarget | ✅ replaced | Own kinematic retargeting and hand-eye correction instead of SPIDER [[Pan et al., 2025](REFERENCES.md#pan2025spider)] (`stages/retargeting`); placement (`stages/locomotion/base_placement.py`), camera pitch and table fit (`stages/reconstruction/align`). |
| M4 MVP-A oracle E2E | ⛔ skipped | RGB-only from the start. |
| M5 MVP-B RGB E2E | ✅ | MoGe-2 [[Wang et al., 2025b](REFERENCES.md#wang2025moge2)], FAH [[Cao, 2026](REFERENCES.md#cao2026fah)], EgoTac [[Zhang et al., 2026](REFERENCES.md#zhang2026egotac)], SAM 3 [[Carion et al., 2026](REFERENCES.md#carion2026sam3)] / OWLv2 [[Minderer et al., 2023](REFERENCES.md#minderer2023owlv2)], Hunyuan3D [[Hunyuan3D Team, 2025](REFERENCES.md#hunyuan2025hunyuan3d2)], optional SimFoundry [[NVIDIA GEAR, 2026](REFERENCES.md#simfoundry2026)]; batch of 10 EgoDex clips × 2 hands (`docs/notes/batch_results.md`). |
| M6 Cloud pipeline | ✖ dropped | Per-stage Docker node images exist (`nodes/`); the cloud pipeline was removed. |
| M7 Post-MVP | ⏸ | OmniGibson [[Li et al., 2022](REFERENCES.md#li2022behavior1k)] dropped. Bimanual: the passive hand is retargeted but does not manipulate. Done beyond the plan: RSI loop (`src/rsi`, `src/skills`), Blender digital twin, approach locomotion (`stages/locomotion`). Open defects: bugfix backlog BF.1–BF.11 in `TASKS.md`. |

## Fallbacks that were taken

| Trigger | Fallback in use |
|---|---|
| SAM 3 weights gated | `perceive_objects_owl.py` (OWLv2 + SAM) writes the same `objects_sam3.npz` |
| No reliable monocular 6D object pose | Centroid/extent tracks plus a generated mesh; the object start xy is refined by physics |
| A SimFoundry scene for every clip is too costly | Scene optional (only clip 104); a synthetic robot-fitted table everywhere |
| The Gaussian-splat [[Kerbl et al., 2023](REFERENCES.md#kerbl20233dgs)] background needs a CUDA env | The ego panel renders the simulated scene from the D435; the splat path is kept but unused |

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](REFERENCES.md).

- <span id="cite-banerjee2025hot3d"></span>[Banerjee et al., 2025](REFERENCES.md#banerjee2025hot3d). *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos.*
- <span id="cite-cao2026fah"></span>[Cao, 2026](REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-hunyuan2025hunyuan3d2"></span>[Hunyuan3D Team, 2025](REFERENCES.md#hunyuan2025hunyuan3d2). *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation.*
- <span id="cite-kerbl20233dgs"></span>[Kerbl et al., 2023](REFERENCES.md#kerbl20233dgs). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
- <span id="cite-li2022behavior1k"></span>[Li et al., 2022](REFERENCES.md#li2022behavior1k). *BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities (OmniGibson).*
- <span id="cite-minderer2023owlv2"></span>[Minderer et al., 2023](REFERENCES.md#minderer2023owlv2). *Scaling Open-Vocabulary Object Detection (OWLv2).*
- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
- <span id="cite-pan2025spider"></span>[Pan et al., 2025](REFERENCES.md#pan2025spider). *SPIDER: Scalable Physics-Informed Dexterous Retargeting.*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
