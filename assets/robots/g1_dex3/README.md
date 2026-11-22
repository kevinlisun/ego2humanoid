# `assets/robots/g1_dex3/` — Unitree G1 with Dex3-1 hands

Unitree G1 [[Unitree Robotics](../../../docs/REFERENCES.md#unitree-g1)] (29-DoF body) with Dex3-1 hands, rev 1.0, from `unitreerobotics/unitree_ros`
(`robots/g1_description`, pinned SHA in `scripts/fetch_g1_assets.sh`).

| File | Tracked | Content |
|---|---|---|
| `g1_29dof_with_hand_rev_1_0.xml` | yes | The MJCF |
| `robot_joint_order.json` | yes | Authoritative joint order: `robot_id` `unitree_g1_dex3_v1`, 43 `joint_names` with `lower_rad` / `upper_rad` limits, the `mjcf` file name and its `mjcf_sha256`; `floating_base_joint` is removed (fixed-pelvis robot) |
| `LICENSE.unitree_ros` | yes | Upstream BSD-3-Clause licence |
| `meshes/` | no (git-ignored) | STL meshes (about 31 MB), written by `scripts/fetch_g1_assets.sh` |

Fetch: `bash scripts/fetch_g1_assets.sh`. Without the meshes MuJoCo [[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco)] fails with
`Error opening file 'meshes/...STL'`.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../../docs/REFERENCES.md).

- <span id="cite-todorov2012mujoco"></span>[Todorov et al., 2012](../../../docs/REFERENCES.md#todorov2012mujoco). *MuJoCo: A Physics Engine for Model-Based Control.*
- <span id="cite-unitree-g1"></span>[Unitree Robotics](../../../docs/REFERENCES.md#unitree-g1). *G1 humanoid robot and Dex3-1 dexterous hand.*
