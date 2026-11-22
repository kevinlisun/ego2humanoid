"""Trajectory/scene constants of ``scripts/retarget_estimated.py`` from ``SimParams`` (Track RSI)."""

import os
import sys

import numpy as np

from skills.params.resolve import accepted_path, resolve_params
from stages.manipulation.sim_eval import offset_grid

MIN_SCORE = 0.7
# defaults, the clip's accepted real2sim parameters, $E2H_SIM_PARAMS overlay, legacy E2H_* vars (Track RSI)
SIM_PARAMS = resolve_params(
    accepted=accepted_path(sys.argv[1], os.environ.get("HAND", "dex3")) if len(sys.argv) > 1 else None
)
TP = SIM_PARAMS.trajectory
ADJUST_STANCE = TP.adjust_stance  # L.17: move the robot only if the view needs it
PLACE_STANCE_SEARCH = TP.place_stance_search  # BF.9: re-score the standing spot on the place path
PLACE_REACH_WEIGHT_M = TP.place_reach_weight_m  # BF.9: weight of the place window in that score
MAX_OFFSET_M = TP.max_offset_m  # a larger object-relative shift pushes the wrist out of the arm's reach
PASSIVE_MIN_VALID = (
    0.5  # the other hand drives its arm only if it is reliably tracked in this fraction of frames
)
APPROACH_FRAMES = TP.approach_frames
OPEN_MARGIN_M = TP.open_margin_m
# The open gap is commanded between fingertip sites, but the Sharpa sites sit on the axes of 8-9 mm
# capsule pads, so a 2.8 cm site gap left 1 cm free and the index pad toppled a 1 cm pen (clip 108).
OPEN_PAD_M = {"dex3": 0.0, "sharpa": TP.open_pad_m}
# How far above the commanded fingertip point the finger pad reaches (the Dex3 tip is the far end of
# the distal link, the Sharpa tip a site on the pad), and the lowest the tips may close above the table.
PAD_REACH_M = {"dex3": 0.025, "sharpa": TP.pad_reach_m}
TIP_FLOOR_M = TP.tip_floor_m
MAX_OPEN_EXTRA_M = (
    TP.max_open_extra_m
)  # open aperture beyond the centre chord (clip 104: 3.9 cm chord, 6.3 cm across)
PINCH_AXIS_TOL_DEG = 10.0  # re-yaw the object when the commanded pinch axis is further off than this
GRASP_AXIS_FRAMES = 15  # open-pinch frames before the close starts over which that axis is measured
# Outward-wound triangles of a box whose corner k is (sign bits 4, 2, 1 of k) for (x, y, z).
BOX_FACES = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)  # fmt: skip
MIN_PICK_RISE_M = 0.05  # a smaller rise of the pinch is not a pick (hand jitter, a slide)
CLOSE_RAMP_FRAMES = TP.close_ramp_frames  # frames over which the pinch closes from open to squeezed (0.2 s)
CLOSE_LEAD_FRAMES = (
    TP.close_lead_frames
)  # start closing the pinch 0.4 s before the object is first seen lifted
GRIP_REACH_TOL_M = 0.03  # the human pinch is "at the object" within this of its rest point
HOVER_CLEARANCE_M = (
    TP.hover_clearance_m
)  # the open pinch hovers this far above the object's top before the descent
MAX_HOVER_RISE_M = (
    TP.max_hover_rise_m
)  # but no higher above the grasp point (a tall object is gripped from the side)
PLACE_DWELL_FRAMES = (
    TP.place_dwell_frames
)  # frames after the release during which the pinch is held still over the receptacle
PLACE_XY_LEAD_FRAMES = TP.place_xy_lead_frames  # extra frames over which the receptacle's xy offset fades in
PLACE_LEAD_FRAMES = (
    TP.place_lead_frames
)  # frames before the release over which the receptacle's frame takes over
START_SEARCH_M = TP.start_search_m
OFFSETS_M = offset_grid(START_SEARCH_M)
EGO_PANEL = "robot ego view (Unitree G1 head D435)"
MIN_OBJECT_THICKNESS_M = (
    TP.min_object_thickness_m
)  # below this the pinch lands on the table instead of the object
# An extruded object is at most this many times taller than its pinched width. Only for the short
# Sharpa pads: the Dex3 distal links reach 2.5 cm above the tip and straddle the top edge of a lower
# object (clip 108: an 8 mm pen extruded to 1.6 cm was rolled over; at 4 cm it was lifted).
MAX_EXTRUDE_ASPECT = {"dex3": float("inf"), "sharpa": 2.0}
SLAB_MAX_WIDTH_M = (
    TP.slab_max_width_m
)  # extruded objects pinched across less than this collide as a parallel slab
TABLE_Z_DEFAULT_M = 0.80
TABLE_MARGIN_M = TP.table_margin_m  # the table extends this far beyond the reach of the retargeted hands
TABLE_INSET_M = TP.table_inset_m  # keep the carried object this far inside the table rim
# Every object on the table is simulated when the clip has a tabletop inventory (``scene/``, from
# perceive_scene_objects.py); trajectory.scene_objects=false keeps the held object and receptacle only.
SCENE_OBJECTS = TP.scene_objects
SCENE_TABLE_MARGIN_M = TP.scene_table_margin_m  # the table reaches this far past the outermost scene object
OBSTACLE_MARGIN_M = (
    TP.obstacle_margin_m
)  # the robot wrist is bulkier than the hand landmarks kept off an object
# The carried object is kept above other objects' collision this far around it: the executed pinch lags
# the commanded one where the arm nears its reach (clip 117: 3 cm, and the block met the scooper rim).
HELD_MARGIN_M = TP.held_margin_m
PLACE_CLEARANCE_M = TP.place_clearance_m
RECEPTACLE_OPENING_MARGIN_M = TP.receptacle_opening_margin_m
PLACE_RETRY_BUDGET_S = TP.place_retry_budget_s
HELD_CLEARANCE_M = TP.held_clearance_m  # and this far above it
# The perceived receptacle stays where it was seen while the human release lands this far inside its
# rim (the drop point follows the release); further out it is re-anchored under the release.
REBASE_OUTSIDE_M = 0.03
TABLE_NEAR_EDGE_M = TP.table_near_edge_m  # closest the table may come to the pelvis
SQUEEZE_M = (
    TP.squeeze_m
)  # least squeeze: robot tip gap = object width - squeeze, so the PD controller preloads the grip
GRIP_MIN_N = TP.grip_min_n  # least grip force the squeeze is sized for
CARRY_ACCEL_M_S2 = TP.carry_accel_m_s2  # held-path acceleration limit
ANTIPODAL_EXTRA_WIDTH_M = 0.01  # the antipodal pinch line may be this much wider than the narrowest
# Release primitive (all clips): the fingers open over RELEASE_OPEN_FRAMES while the pinch holds
# still for PLACE_DWELL_FRAMES, then it rises RETREAT_HEIGHT_M over RETREAT_FRAMES and blends back
# onto the human path over RETREAT_BLEND_FRAMES; the fingers stay open until then.
RELEASE_OPEN_FRAMES = TP.release_open_frames
RETREAT_FRAMES = TP.retreat_frames
RETREAT_HEIGHT_M = TP.retreat_height_m
RETREAT_BLEND_FRAMES = TP.retreat_blend_frames
BARE_PLACE_LIFT_M = (
    TP.bare_place_lift_m
)  # a bare-table release puts the object down this far above its rest height
