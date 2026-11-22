# Track RSI — regression baseline (RSI.1 / RSI.4 gate)

Sharpa, `RENDER=0`, retarget stage only (perception artefacts unchanged), clips 0, 10, 101, 104, 108, 114, 116, 117
(the owner's "10001" was read as 101). Baseline = `develop` @ 84315ff; new = `feature/rsi-real2sim` @ 0566953
(physics rollouts built through `SceneConfig.from_params(SimParams())`, `sim_params_hash 18812717e39b7dc9`).

| clip | all `retarget_rgb_sharpa.npz` arrays (incl. `q_target`) | json scalar diffs |
|---|---|---|
| 0, 10, 104, 108, 114, 116, 117 | bit-identical (max abs diff 0) | `perceived_object.mesh_texture_iou` only (±0.001) |
| 101 | bit-identical | none |

`mesh_texture_iou` is a perception-side render score that jitters between runs; it does not feed the sim.
QA verdicts (`NO_LIFT`/`NO_TRANSLATION`, …) are unchanged for every clip, i.e. the develop QA state is
reproduced exactly, not improved: defaults == today.
Wall time per clip under shared-CPU load: 0.35–2.1 h (develop and RSI branch within 5 %).
