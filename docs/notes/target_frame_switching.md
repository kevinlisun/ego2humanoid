# Switching the hand-eye reference frame during a task

**Status:** implemented (K.1-K.7). "The problem" below is the state before K.4. Where the code lives now:
`resolve_place_target` in `src/stages/retargeting/script_helpers.py`; `PlaceTarget`, `rebase`, `container_pieces` in
`src/stages/reconstruction/objects/place_target.py`; `grasp_window`/`align_pinch`/phase blending in
`src/stages/retargeting/hand_eye.py`; `EventThresholds`/`detect_pick_place` in `src/stages/manipulation/events.py`.
The receptacle and the other tabletop objects now come from the clip's `scene/inventory.json` (also when `SCENE_DIR` is set;
`SCENE_DIR/objects.json` is the legacy SimFoundry source for clip 104). When the first placement misses the receptacle,
`retarget_estimated.py` re-runs the search up to three passes with the place goal (`E2H_PLACE_GOAL`).

## The problem

`H.10` made the retargeted wrist trajectory *object-relative*: the commanded pinch point is corrected
so that, at the grasp instant, the robot's fingertips meet the grasped object's centre instead of
wherever the human's happened to be. The correction is a single rigid 3-D offset, faded in and out
over the frames in which the fingers are closed.

That is only half of a pick-and-place. Clip 104's task description is

> pick up the white block from the table with the left hand and **place it on the blue bowl**.

so there are *two* objects the hand has to be accurate about, at two different times:

| phase | what the hand must coincide with | current behaviour |
|---|---|---|
| reach / grasp | the **source** object (white block) | corrected (H.10) |
| carry | nothing | follows the human |
| place / release | the **target** object (teal bowl) | **uncorrected** — follows the human |

Two consequences, both measured:

1. (Before K.4) The receptacle was **not in the physics scene at all**. `scripts/retarget_estimated.py` builds the
   rollout with `objects=(perceived_box,)` plus the table; `load_scene_objects` (now in `stages/reconstruction/scene/export_mjcf.py`) was imported only
   inside the `RENDER=1` branch. The robot therefore drops the block onto the bare table where the
   bowl visually is, and the composite video shows a block passing through a bowl that exists only
   as pixels.
2. The success criterion is wrong for this family of tasks. `EventThresholds` calls a rollout
   `placed` when `abs(final_height_above_rest_m) <= 0.02`, i.e. the object must come back to the
   height it started at. Placing an object *into* or *onto* a receptacle ends it one receptacle
   height higher, so a perfect placement is scored as a failure. This is a large part of the current
   `NOT_PLACED` cluster.

## The general scheme: phase-indexed object frames

Generalise "the trajectory is relative to *the* object" to "**the trajectory is relative to a
different object in each interaction phase**".

### 1. Interaction phases

A phase is a triple

```
(frames, reference_object, target_point_in_object_frame)
```

For a pick-and-place the schedule is derived from quantities the pipeline already computes
(`hold_k`, `close_start`, `release_i` in `retarget_estimated.py`):

| phase | frames | reference | target point |
|---|---|---|---|
| `grasp` | `close_start … close_start + hold_lead` | source object | its graspable centre (H.10's `anchor`) |
| `place` | `release_i − place_lead … release_i` | target object | its **drop point** (below) |

The schedule is data, not code: a task with three interactions (open a drawer, put the block in,
close the drawer) is the same list with three entries. Nothing below depends on there being exactly
two phases.

### 2. Blending the corrections

Each phase `p` contributes a rigid offset `align_p = target_point_p − human_pinch_at_phase_p`, and a
weight `w_p(t)` that is 1 on its frames and fades to 0 over a Hanning ramp (today's `grasp_window`).
The commanded pinch becomes

```
target_pinch(t) = human_pinch(t) + Σ_p w_p(t) · align_p
```

with the weights **normalised where windows overlap** (`Σ_p w_p ≤ 1`), so a long carry between two
close-together phases interpolates smoothly from the source frame to the target frame instead of
summing two corrections. Between the phases the hand follows the human exactly, which is what keeps
the carry trajectory human-like; at each end it is pinned to the object that matters there.

`align_pinch` is unchanged — it already takes an arbitrary `target_pinch` track and a weight. The
weight passed to it becomes `Σ_p w_p`, so the residual is still scored only where a correction is
actually demanded.

### 3. Resolving the target object

Two sources, in priority order, behind one interface
`resolve_place_target(clip, task_description, scene_dir) -> PlaceTarget | None`:

* **S-A, reconstructed scene available.** `data/scenes/<scene>/objects.json` already lists named,
  sim-frame-posed meshes (`teal_plastic_bowl`, `orange_smiley_ball`, …). Parse the receptacle phrase
  out of the task description (`place it on the blue bowl` → `blue bowl`) and match it against the
  object names by token overlap, with a small colour synonym table (`blue ≈ teal`). This gives an
  exact mesh and pose — no new perception.
* **S-B, no scene.** Prompt SAM 3 with the receptacle phrase on the same frames already used for the
  source object, lift the candidates into the sim frame with the existing `hand_depth_scale`
  machinery, and keep the candidate that is (a) **stationary** over the clip (the receptacle is not
  carried), (b) nearest to the human pinch point *at release*, (c) not the source object. This
  mirrors `select_held_object`, which keeps the candidate that *moves with* the pinch; the target is
  selected by the opposite test.

Both return the same record:

```python
@dataclass
class PlaceTarget:
    name: str
    pos_m: np.ndarray  # sim-frame centre
    half_size_m: np.ndarray
    drop_point_m: np.ndarray  # where the held object's centre should end up
    mesh: BoxObject | None  # None for S-B, a mesh for S-A
```

### 4. The drop point

The only receptacle-specific geometry. For a container the natural target is the **centre of the
opening, lowered by the held object's half height plus a clearance**:

```
drop_point = [cx, cy, top_z − rim_depth + held_half_z + clearance]
```

`top_z` comes from the mesh bounding box; `rim_depth` is estimated from the mesh itself by comparing
the mesh's own volume with its convex hull (a bowl is hollow, a plate is not): for a hull-filling
object the drop point sits *on top* (`rim_depth = 0`), for a hollow one it sits inside. This keeps a
single formula for "on the plate", "in the bowl" and "in the drawer".

### 5. Putting the receptacle in the physics scene

`run_rollout` must contain it, otherwise nothing can be scored. The receptacle is added as a
**static** body (`mass` large or welded) so the arm cannot knock it across the table, using the
mesh from S-A or a box from S-B. This also fixes the composite video, where the block currently
passes through the bowl.

The `OFFSETS_M` search that refines the source object's resting xy stays as-is; the receptacle pose
is *not* searched, because the reconstructed scene pose is already metric.

### 6. Release gating

Today the fingers open at `release_i = hold_k[-1]`, a perception-derived frame. With a target frame
available the release becomes a condition on the *relative* state, which is what makes it general:
open when the held object is above the drop point (xy within the receptacle's footprint) **and**
descending or within `place_tol_m` of the drop point. Falling back to `release_i` when no target is
resolved keeps every current clip working.

### 7. Scoring

`detect_pick_place` gains an optional `place_target_pos_m`. When present:

* `placed` = the held object's final centre is within `place_radius_m` of the drop point in xy
  **and** within `place_height_tol_m` of it in z — i.e. it is resting on/in the receptacle,
* instead of the current "returned to its original height" test, which is only right for
  table → table transfers.

When absent the existing rule is kept verbatim, so clips without a receptacle are unaffected.

## Why this is the right generalisation

* It is **compositional**: a phase list of length 1 reproduces today's behaviour exactly, a length-3
  list expresses a drawer task, and nothing in `align_pinch` changes.
* It separates *what the hand must touch* (perception, phase schedule) from *how the arm is driven*
  (the existing closed-loop hand-eye solver).
* Both target sources produce the same record, so the simulation and the scoring never learn whether
  the receptacle came from a reconstruction or from a mask.
* It fixes a scoring bug that is currently hiding real successes, rather than tuning thresholds.

## What the implementation had to add beyond the design

Three things only showed up once the receptacle was actually simulated on clip 104:

1. **Frame disagreement.** `objects.json` poses and the retargeted trajectory agree only up to the
   scene alignment. The reconstructed bowl centre sat 0.3 m from where the human releases — off the
   near edge of the synthetic table — and the block was thrown onto the floor. `place_target.rebase`
   keeps the mesh as the source of *shape* (and therefore of the drop height) and uses the release
   point as the source of *position*, clamped inside the table rim.
2. **Hollow geometry is not hollow collision.** The reconstructed bowl is an open shell, so it is
   not watertight, so CoACD falls back to a single convex hull: a solid dome. The block landed on
   it and slid off (0.82 m below the table). `container_pieces` synthesises a floor plus four walls
   from the same bbox and inner-floor height, leaving the visual mesh untouched.
3. **A static body is not a prop.** `rollout` looked up a `<name>_free` joint for every object and
   randomised every object's xy. Both now skip static bodies, and contact statistics exclude the
   receptacle so that brushing the bowl is not counted as grasping.

Measured on clip 104 (Sharpa): `placed` false → true, final height −0.82 m → +0.055 m above rest,
place error 0.90 m → 0.063 m. Batch over 9 clips × 2 hands: 4/18, unchanged elsewhere.

## Place correction diagnostics and back-off (RSI follow-up)

`retarget_estimated.py` prints `place diag:` and stores `place_diag` in the retarget report: the
requested place offset (`asked_offset_m`), the fraction kept after the back-off (`kept_frac`) and, at
the release frame, the human, commanded and reached pinch positions with their differences to the
drop point and the IK error. It shows where the correction is lost.

Findings (Sharpa): the correction formula is sound; reachability limits it. Clip 10 asks 2.4/1.4 cm,
keeps half and ends 3 cm short with a 4.5 cm IK error at release. Clip 1 asks 6.5/5.8/6.3 cm, 51 % of
the frames are unreachable with it (12 % without), so all of it is dropped.

The back-off ladder is (1, 1), (1, 0.75), (1, 0.5), (1, 0.25), (1, 0), then both together; when the
full correction fails, the bound is the unreachable fraction of the grasp-only path (1, 0) + 1 %, so
the place correction is judged by the frames it adds. Clip 0 (place 0.5 -> 0.75) went from 3.9 to
1.8 cm place error; clips 1, 10, 100, 101, 108, 116 are unchanged.

Open: express the drop point in the receptacle's frame so it follows a rebase, and aim the pinch at
the held object's pose (measured hand-to-object offset at closure) instead of the pinch itself.

## Task breakdown

See `TASKS.md`, track K (`K.1` … `K.6`).
