# `nodes/hands/` — hand-pose estimation image (FAH, WiLoR)

Hand-pose stage (`scripts/estimate_hands_fah.py`, WiLoR [[Potamias et al., 2025](../../docs/REFERENCES.md#potamias2025wilor)] fallback `scripts/estimate_hands_wilor.py`).
Pins reproduce the working venvs (`requirements-fah.txt`: torch 2.11+cu128, numpy 2; `requirements-wilor.txt`:
torch 2.9.1+cu128, numpy 1.26). No uv.lock: the two stacks conflict, so each target has its own pinned requirements.

    docker build -f nodes/hands/Dockerfile -t ego2humanoid/hands:dev .                        # FAH (default)
    docker build -f nodes/hands/Dockerfile --target wilor -t ego2humanoid/hands-wilor:dev .   # WiLoR

Restricted FAH [[Cao, 2026](../../docs/REFERENCES.md#cao2026fah)]/WiLoR code, MANO [[Romero et al., 2017](../../docs/REFERENCES.md#romero2017mano)] files and weights are not baked in; mount them:

    docker run --rm --gpus all -v $PWD/data:/data -v /path/FAH:/third/FAH:ro -v /path/mano:/mano:ro \
        -v /path/hf:/hf ego2humanoid/hands:dev python scripts/estimate_hands_fah.py <clip_dir> --focal F
    docker run --rm --gpus all -v $PWD/data:/data -v /path/WiLoR:/third/WiLoR:ro \
        ego2humanoid/hands-wilor:dev python scripts/estimate_hands_wilor.py <clip_dir> --focal F

Env: `FAH_HOME=/third/FAH`, `MANO_DIR=/mano`, `WILOR_REPO=/third/WiLoR`; `/third/FAH` is on `PYTHONPATH`
(editable install replaced by the mount). In `run_pipeline.sh` use `FAH_PY`/`WILOR_PY` wrappers around `docker run`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-cao2026fah"></span>[Cao, 2026](../../docs/REFERENCES.md#cao2026fah). *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space (FAH).*
- <span id="cite-potamias2025wilor"></span>[Potamias et al., 2025](../../docs/REFERENCES.md#potamias2025wilor). *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.*
- <span id="cite-romero2017mano"></span>[Romero et al., 2017](../../docs/REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
