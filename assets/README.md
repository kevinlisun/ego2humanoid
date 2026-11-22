# `assets/` — static robot models and hand-model files

Static robot models used by the MuJoCo [[Todorov et al., 2012](../docs/REFERENCES.md#todorov2012mujoco)] simulation and the retargeting. Only the robot directory exists here.

| Path | Content |
|---|---|
| `robots/` | Unitree G1 [[Unitree Robotics](../docs/REFERENCES.md#unitree-g1)] with Dex3 hands and the Sharpa Wave [[Sharpa](../docs/REFERENCES.md#sharpa-wave)] hand (see `robots/README.md`) |

Mesh files are not committed (`.gitignore`: `assets/robots/*/meshes/`); fetch them with
`scripts/fetch_g1_assets.sh` and `scripts/fetch_sharpa_assets.sh`. The hand estimator's cleaned MANO [[Romero et al., 2017](../docs/REFERENCES.md#romero2017mano)] model
(`assets/mano/clean`, default `MANO_SRC` of the `hands` node) is a separate manual download and is not part of this
directory in git (MANO licence: <https://mano.is.tue.mpg.de>). Licences are listed in `docs/LICENSES.md`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../docs/REFERENCES.md).

- <span id="cite-romero2017mano"></span>[Romero et al., 2017](../docs/REFERENCES.md#romero2017mano). *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).*
- <span id="cite-sharpa-wave"></span>[Sharpa](../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-unitree-g1"></span>[Unitree Robotics](../docs/REFERENCES.md#unitree-g1). *G1 humanoid robot and Dex3-1 dexterous hand.*
