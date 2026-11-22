# `nodes/tactile/` — vision-to-tactile image (EgoTac)

EgoTac [[Zhang et al., 2026](../../docs/REFERENCES.md#zhang2026egotac)] tactile stage (`scripts/estimate_tactile_egotac.py`). Python 3.10, torch 2.5.1 (PyPI wheels bundle CUDA 12.4;
cu128 needs torch>=2.7, unsupported by EgoTac). Inference subset of `third_party/EgoTac/environment.yaml`.

    scripts/build_node.sh tactile
    docker run --rm --gpus all -v $PWD:/app -v $PWD/data:/data -v <hf-cache>:/hf \
      ego2humanoid/tactile:dev python scripts/estimate_tactile_egotac.py <clip_dir>

Code, `third_party/EgoTac` and `data/weights/egotac/egotac_base_h4.pt` are mounted, never baked in.
If `third_party/EgoTac` files are symlinks, also mount their real target over `/app/third_party/EgoTac`.
`PYTHONPATH` is intentionally unset: `src/common` would shadow EgoTac's namespace package `common`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-zhang2026egotac"></span>[Zhang et al., 2026](../../docs/REFERENCES.md#zhang2026egotac). *EgoTac: In-the-wild Tactile Prediction from Egocentric Vision.*
