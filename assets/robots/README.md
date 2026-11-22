# `assets/robots/` — pinned robot models (MJCF)

Pinned robot models. The MJCF files are tracked in git; meshes are downloaded by script and git-ignored.

| Directory | Robot | Source (pinned commit in the fetch script) | Licence | Fetch |
|---|---|---|---|---|
| `g1_dex3/` | Unitree G1 [[Unitree Robotics](../../docs/REFERENCES.md#unitree-g1)], 29 DoF body with Dex3-1 hands (rev 1.0) | `unitreerobotics/unitree_ros`, `robots/g1_description` | BSD-3-Clause (`g1_dex3/LICENSE.unitree_ros`) | `scripts/fetch_g1_assets.sh` |
| `sharpa_wave/` | Sharpa Wave [[Sharpa](../../docs/REFERENCES.md#sharpa-wave)] five-finger tactile hand (`left_hand.xml`, `right_hand.xml`) | `google-deepmind/mujoco_menagerie`, `sharpa_wave` | Apache-2.0 (`sharpa_wave/LICENSE`) | `scripts/fetch_sharpa_assets.sh` |

`sharpa_wave/` keeps the upstream `README.md` (it requires MuJoCo [[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco)] 3.1.6 or later). Both fetch scripts are
idempotent, check the MJCF against upstream and print the pinned SHA.

The simulation (`HAND=dex3` or `HAND=sharpa`) attaches the chosen hand to the G1 arms. Hand choice:
`docs/notes/hand_selection.md`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-sharpa-wave"></span>[Sharpa](../../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-unitree-g1"></span>[Unitree Robotics](../../docs/REFERENCES.md#unitree-g1). *G1 humanoid robot and Dex3-1 dexterous hand.*
