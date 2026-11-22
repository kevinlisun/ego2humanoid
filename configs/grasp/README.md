# `configs/grasp/` — stored Sharpa Wave pinch poses

| File | Purpose |
|---|---|
| `sharpa_pinch_v1.json` | Thumb-index pad pinch joint sets of the Sharpa Wave [[Sharpa](../../docs/REFERENCES.md#sharpa-wave)] hand (right-hand joint names; mirrored by prefix) |

## `sharpa_pinch_v1.json` keys

| Key | Unit | Meaning | Read by |
|---|---|---|---|
| `object_width_m` | m | Object width between the pads the poses were solved for (`0.03`) | `PinchPoses.object_width_m` |
| `pinch_point_wrist_m` | m | Pinch point in the wrist-yaw-link frame (fingers +x, palm normal +y, thumb side +z); mirrors `PINCH_POINT_WRIST` | written only; no runtime reader found |
| `stages.<touch\|squeeze\|open>.joints` | rad | 9 joints: `right_thumb_{CMC_FE,CMC_AA,MCP_FE,MCP_AA,IP}`, `right_index_{MCP_FE,MCP_AA,PIP,DIP}` | stored solver output (see gotcha) |
| `stages.*.gap_m` | m | Achieved thumb–index fingertip gap (touch 0.0355, squeeze 0.0248, open 0.1181) | stored output |
| `stages.*.tip_err_m` | m | Residual fingertip-site error of the least-squares solve | stored output |

## Code

- Writer: `scripts/solve_sharpa_pinch.py [--width 0.03]` builds the Sharpa scenario, calls
  `stages.retargeting.sharpa_pinch.solve_pinch(model, width, side="right")` and overwrites this file.
  Targets: open gap `OPEN_GAP_M=0.14`, touch = `width + 0.005`, squeeze = `width - 0.006` (m).
- Loader: `PinchPoses.load(path=POSES_PATH)`; `pose(stage, side="right")` returns `carry` (fist) or a blend of the
  `finger_mapping.yaml` open→closed poses using `WRAP_BLEND = {open: 0.0, touch: 0.7, squeeze: 1.0}`.
- Consumer: `stages/locomotion/episode.py` (`PinchPoses.load()` for `hand == "sharpa"`; finger script `touch`,`squeeze` at grasp, `open` at place).

## Gotchas

- `PinchPoses.pose` currently builds poses from `configs/finger_mapping.yaml` and `WRAP_BLEND`; of this file only
  `object_width_m` is read by code. Re-solving only refreshes the stored joint sets.
- Requires `mujoco` and the Sharpa assets (`scripts/fetch_sharpa_assets.sh`).
- Tests: `tests/unit/test_grip.py` (uses `PinchPoses`).
- Parent: [`../README.md`](../README.md).

## Related docs

- [`docs/TECH_DESIGN.md`](../../docs/TECH_DESIGN.md) — retargeting and grasp design.
- Finger open/closed mapping used for the blends: `../finger_mapping.yaml` ([`../README.md`](../README.md)).
- Scenario used by the solver: [`../scenarios/README.md`](../scenarios/README.md).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-sharpa-wave"></span>[Sharpa](../../docs/REFERENCES.md#sharpa-wave). *Wave tactile dexterous hand.*
