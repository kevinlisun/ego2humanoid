# `nodes/perception/` — geometry, object and mesh perception image

Stages: intrinsics, gravity (MoGe [[Wang et al., 2025b](../../docs/REFERENCES.md#wang2025moge2)]), objects (SAM3 [[Carion et al., 2026](../../docs/REFERENCES.md#carion2026sam3)] / OWLv2 [[Minderer et al., 2023](../../docs/REFERENCES.md#minderer2023owlv2)]), scene, scene_mesh, held_mesh (Hunyuan3D-2 [[Hunyuan3D Team, 2025](../../docs/REFERENCES.md#hunyuan2025hunyuan3d2)]).

Build (repo root): `scripts/build_node.sh perception`

Run (weights and data are mounted, never baked in):

    docker run --rm --gpus all -v $PWD/data:/data -v $HF_HOME:/hf \
      ego2humanoid/perception:dev python scripts/estimate_intrinsics_moge.py ...

Notes:
- Python 3.11, torch 2.9.1+cu128 (pinned wheel URLs; `no-sources` because MoGe ships cu130 uv sources).
- MoGe @ 74fbce0, utils3d-moge @ 62f09d5, Hunyuan3D-2 @ f8db630 (pure Python shapegen, no custom CUDA ext built).
- Hunyuan3D weights under `data/weights/hunyuan3d`; SAM3 / OWLv2 / MoGe via `HF_HOME=/hf`.
- Mount the repo at `/app` for live code (`PYTHONPATH=/app/src`); otherwise the COPY'd snapshot is used.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-carion2026sam3"></span>[Carion et al., 2026](../../docs/REFERENCES.md#carion2026sam3). *SAM 3: Segment Anything with Concepts.*
- <span id="cite-hunyuan2025hunyuan3d2"></span>[Hunyuan3D Team, 2025](../../docs/REFERENCES.md#hunyuan2025hunyuan3d2). *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation.*
- <span id="cite-minderer2023owlv2"></span>[Minderer et al., 2023](../../docs/REFERENCES.md#minderer2023owlv2). *Scaling Open-Vocabulary Object Detection (OWLv2).*
- <span id="cite-wang2025moge2"></span>[Wang et al., 2025b](../../docs/REFERENCES.md#wang2025moge2). *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.*
