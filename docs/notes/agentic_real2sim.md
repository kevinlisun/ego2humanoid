# Agentic real2sim — Mode A playbook (Track RSI)

A Copilot CLI session tunes one clip's `SimParams` through `scripts/real2sim_eval.py`. Design:
`docs/notes/track_r_agentic_real2sim.md` §10. "Matches the video" means matches monocular estimates
(`correspondence_to_real=UNVERIFIED_MONOCULAR`).

## Setup

```bash
# in the retarget node image (nodes/retarget, MuJoCo 4.0.0 from third_party/mujoco; run via
# scripts/node_python.sh / nodes/bin/retarget-python) or the root Dockerfile image, HAND=sharpa
python scripts/real2sim_eval.py init data/derived/egodex/basic_pick_place_104   # full run, iter_000
python scripts/real2sim_eval.py show data/derived/egodex/basic_pick_place_104   # feedback.md
python scripts/real2sim_eval.py schema --affects sim                            # tunable leaves
```

## Loop

1. Read `feedback.md`: timeline (time-stamped events), residuals vs. the video, `Hints (affects)`.
2. Pick **one hypothesis per iteration** (1–3 keys). Prefer `sim`-only keys (rollout-only, seconds)
   over `plan` keys (full re-plan, 20–70 min).
3. `eval <clip> --set key=value … --rationale "<why>"`; use `--freeze-plan` to get the physics answer
   for a plan-changing key without re-planning.
4. `history <clip>`; keep the best parent (`--parent N`) and revert when an iteration is worse.
5. Stop when the run succeeds and `match_score ≤ 0.35`, after 5 iterations without improvement, or at
   20 iterations / the owner's budget.
6. `robust <clip> --iter N`, then `accept <clip> --iter N` (≥ 3/5 perturbation passes required).
   Add a Session Log line to `TASKS.md`.

## Rules

* Never edit code, QA thresholds, feedback weights or perception outputs inside the loop. A needed
  structural change (new collision piece, new contact type) is reported as a finding.
* No state injection after reset; parameters act at scene build only (HARNESS §7).
* Accepted params are per clip and hand; promoting them to `configs/sim_params/default.yaml` is an
  owner decision (Q17).

## Reading the feedback

| Event | First keys to try |
|---|---|
| `AIRBORNE_RELEASE` | `trajectory.place_dwell_frames`↑, `trajectory.bare_place_lift_m`↓, `trajectory.release_open_frames` |
| `SLIP` / `DROPPED_IN_CARRY` | `trajectory.grip_min_n`↑, `contact.min_slide_friction`↑, `objects.held.friction`↑, `trajectory.carry_accel_m_s2`↓ |
| `PENETRATION` | `contact.*` solver keys, `pd.groups.hand.kp` |
| `TRACKING_ERROR` | `pd.groups.*.kp/kd` |
| `GRASP_PENETRATION` (finger-object >2 mm in approach/close/hold) | `trajectory.squeeze_m`↓, `open_margin_m`↑, `close_ramp_frames`↑, `pd.groups.hand.kp`↓, `contact.solref`↓ |
| `START_EMBEDDED` (object starts >5 mm inside a support) | `objects.<name>.pos_offset_m` / `size_scale`, `robot_collision` |
| `PLACE_BURIED` (>30% of the held mesh under the receptacle mesh; residual `place_buried_frac`) | `trajectory.place_clearance_m`↑ (full re-plan), `held_clearance_m`↑, `objects.receptacle.pos_offset_m` |
| `PLACE_OUTSIDE_RECEPTACLE` / `RIM_BLOCKED` (`skills.receptacle_check`; residual `place_outside_m`) | `trajectory.receptacle_opening_margin_m`↑ (full re-plan), `place_clearance_m`↑, `place_xy_lead_frames`, `objects.receptacle.size_scale` |
| `RETRY_WORSE` / `RETRY_BUDGET_EXHAUSTED` (bundle meta `place_retry`) | `trajectory.place_retry_budget_s` |
| `FINAL_TILTED` (>15 deg) | `trajectory.release_open_frames`↑, `bare_place_lift_m`↓, `place_dwell_frames`↑, `retreat_height_m`↑ |
| `FINAL_NOT_AT_REST` (>5 mm in last 0.5 s) | `trajectory.retreat_frames`↑, `contact.solref`, `contact.noslip_iterations`↑ |
| `FINAL_HEIGHT` (>1 cm, bare table) | `trajectory.bare_place_lift_m`↓, `place_dwell_frames`↑ |
| `SCENE_ROLE_SWAP` (grip geometry overrode the task's source name; `skills/scene_layout.py`) | `scene.source_from` = `name`, `scene.max_registration_m`↑ |
| `SCENE_RECEPTACLE_REBASED` (receptacle moved >10 cm onto the release point) | `scene.source_from`, `objects.receptacle.pos_offset_m` |
| `SCENE_GRIP_GAP` (grip >8 cm from the source, or inventory dropped) | `scene.max_registration_m`↑, `scene.source_from` |

The `final_tilt` / `final_drift` / `final_height` residuals (`skills/final_pose.py`) check the object's end state after the place and enter `match_score`.

The hints block in `feedback.md` is the authoritative, ranked version of this table.

## Pilot results (RSI.17, RSI.18)

**Mode A vs Mode B, clip 1** (Sharpa, plan frozen, default params as iteration 0, one agent for both).
Both modes tried the same three `sim`-only hypotheses (`noslip_iterations` 15; `pd.groups.hand.kp` 12).

| | Mode A (`real2sim_eval.py`) | Mode B (`real2sim_loop.py`, relay) |
|---|---|---|
| iterations | 3 | 3 (+ AGENT_STOP) |
| best match_score | 0.40 (iter 0) | 0.40 (iter 0) |
| wall time per iteration | 6–8 s rollout | 6 s rollout + relay round trip (here ~1–2 min because a person-in-the-loop answers) |
| outcome | no sim-only knob helps | same; agent stopped with the structural cause |

`carry_path` (23.5 cm, saturated) and `grasp_dt` (0.52 s) dominate the score; both are plan-side
(planned carry 32.0 cm against the demo's 30.1 cm, grasp onset timing), so physics knobs cannot move
them. The loop mechanics (request, answer validation, no-repeat rule, state resume, stop) work end to
end; the useful behaviour of either mode is to recognise this and stop, not to burn iterations.

**Robustness of the default parameters on the dev clips** (`robust`: 25 start offsets + 5 perturbed
seeds, pass = seeds ≥ 3/5; the grid fraction is informational because the open-loop plan is not
re-aimed per start):

| clip | OK | match_score | grid | seeds | passed |
|---|---|---|---|---|---|
| 0 | yes | 0.43 | 0.08 | 5/5 | **yes** (`accept` writes `accepted.yaml`) |
| 1 | yes | 0.40 | 0.08 | 1/5 | no |
| 10 | yes | 0.59 | 0.16 | 0/5 | no |
| 100 | yes | 0.69 | 0.12 | 0/5 | no |
| 101 | no | 1.40 | 0.00 | 0/5 | no |
| 104 (iter 4) | yes | 1.00 | 0.04 | 1/5 | no |
| 108 | yes | 0.67 | 0.04 | 1/5 | no |
| 116 | no | 1.20 | 0.00 | 0/5 | no |

Only clip 0 is robust with defaults. A single successful rollout at the best start is not evidence of
a usable plan: a ±5 mm start and ±20 % friction/mass flip most clips. `accept` refuses the others, as
intended. Full and batch runs read `real2sim_<hand>/accepted.yaml` automatically (below `$E2H_SIM_PARAMS`).
