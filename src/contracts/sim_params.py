"""Versioned, validated simulation parameters (Track RSI, RSI.2).

Defaults reproduce the values hard-coded in `stages/replay/mjcf_scene.py`; every leaf carries
`unit`, `range` and `affects` metadata (`sim`: only the physics rollout changes, `plan`: the value
is also an input to retargeting, so a cached trajectory is invalidated).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SIM_PARAMS_VERSION = "sim_params/1"


def _f(default, *, unit: str, doc: str, affects=("sim",), **bounds):
    return Field(default, json_schema_extra={"unit": unit, "affects": list(affects), "doc": doc}, **bounds)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContactParams(_Frozen):
    priority: int = _f(1, unit="1", doc="geom priority of scene objects", ge=0, le=5)
    condim: Literal[1, 3, 4, 6] = _f(4, unit="1", doc="contact dimensionality (4 = torsional friction)")
    solref: tuple[float, float] = _f(
        (0.01, 1.0), unit="[s,1]", doc="object contact time constant and damping ratio"
    )
    solimp: tuple[float, float, float] = _f(
        (0.95, 0.99, 0.001), unit="[1,1,m]", doc="object contact impedance d0, dwidth, width"
    )
    min_slide_friction: float = _f(
        1.0, unit="1", doc="floor of the object sliding friction", affects=("sim", "plan"), ge=0, le=3
    )
    torsion_m: float = _f(0.01, unit="m", doc="torsional friction coefficient", ge=0, le=0.05)
    roll_m: float = _f(0.0001, unit="m", doc="rolling friction coefficient", ge=0, le=0.01)
    impratio: float = _f(10.0, unit="1", doc="friction/normal impedance ratio", ge=1, le=100)
    cone: Literal["elliptic", "pyramidal"] = _f("elliptic", unit="enum", doc="friction cone")
    noslip_iterations: int = _f(5, unit="1", doc="noslip solver iterations", ge=0, le=50)
    table_friction: tuple[float, float, float] = _f(
        (1.0, 0.005, 0.0001), unit="[1,m,m]", doc="table sliding, torsional, rolling friction"
    )

    @model_validator(mode="after")
    def _check(self):
        tc, dr = self.solref
        if not 0.002 <= tc <= 0.1 or not 0.1 <= dr <= 2.0:
            raise ValueError("contact.solref: time constant in [0.002, 0.1] s, damping ratio in [0.1, 2]")
        d0, dw, width = self.solimp
        if not (0.5 <= d0 <= dw <= 0.9999) or not 1e-4 <= width <= 0.01:
            raise ValueError("contact.solimp: 0.5 <= d0 <= dwidth <= 0.9999 and width in [1e-4, 0.01] m")
        if any(v < 0 for v in self.table_friction):
            raise ValueError("contact.table_friction must be >= 0")
        return self


class PDGroup(_Frozen):
    kp: float = Field(gt=0, json_schema_extra={"unit": "N*m/rad", "affects": ["sim"], "doc": "position gain"})
    kd: float = Field(ge=0, json_schema_extra={"unit": "N*m*s/rad", "affects": ["sim"], "doc": "damping"})
    max_torque_nm: float = Field(gt=0, json_schema_extra={"unit": "N*m", "affects": ["sim"], "doc": "limit"})
    armature: float = Field(ge=0, json_schema_extra={"unit": "kg*m^2", "affects": ["sim"], "doc": "rotor"})


def _g(kp, kd, tq, arm):
    return PDGroup(kp=kp, kd=kd, max_torque_nm=tq, armature=arm)


class PDGroups(_Frozen):
    """Groups keep the joint-name substrings used by `PDConfig` (`hand` = `_hand_`, `legs` = catch-all)."""

    hand: PDGroup = _g(8.0, 0.4, 6.0, 0.005)
    wrist: PDGroup = _g(40.0, 2.0, 20.0, 0.01)
    shoulder: PDGroup = _g(150.0, 6.0, 60.0, 0.02)
    elbow: PDGroup = _g(150.0, 6.0, 60.0, 0.02)
    waist: PDGroup = _g(300.0, 10.0, 100.0, 0.02)
    legs: PDGroup = _g(300.0, 10.0, 150.0, 0.02)


class PDParams(_Frozen):
    groups: PDGroups = PDGroups()
    heavy_hand_wrist_scale: float = _f(2.5, unit="1", doc="Sharpa wrist actuator scale", ge=1, le=10)
    gravity_compensation: bool = _f(True, unit="bool", doc="robot links carry no net weight")
    timestep_s: float = _f(0.002, unit="s", doc="physics timestep", ge=0.0005, le=0.004)
    integrator: Literal["implicitfast", "implicit", "euler", "rk4"] = _f(
        "implicitfast", unit="enum", doc="MuJoCo integrator"
    )


Selector = Annotated[str, Field(min_length=1)]


class ObjectPhysics(_Frozen):
    """Override of one object selector: estimated physics (None keeps the estimate) and scene edits."""

    mass_kg: float | None = _f(None, unit="kg", doc="mass", affects=("sim", "plan"), ge=0.005, le=2.0)
    friction: float | None = _f(
        None, unit="1", doc="sliding friction", affects=("sim", "plan"), ge=0.05, le=3.0
    )
    enabled: bool = _f(True, unit="bool", doc="false drops the scene object", affects=("sim", "plan"))
    static: bool | None = _f(None, unit="bool", doc="weld (true) or free (false); None keeps the role")
    robot_collision: bool | None = _f(None, unit="bool", doc="collide with the robot; None keeps the role")
    pos_offset_m: tuple[float, float, float] = _f(
        (0.0, 0.0, 0.0), unit="m", doc="shift of the scene object", affects=("sim", "plan")
    )
    yaw_offset_deg: float = _f(
        0.0, unit="deg", doc="rotation about z (mesh objects)", affects=("sim", "plan"), ge=-180, le=180
    )
    size_scale: tuple[float, float, float] = _f(
        (1.0, 1.0, 1.0), unit="1", doc="per-axis scale (mesh objects)", affects=("sim", "plan")
    )

    @model_validator(mode="after")
    def _check(self):
        if any(abs(v) > 0.10 for v in self.pos_offset_m):
            raise ValueError("objects.*.pos_offset_m: each component within +-0.10 m")
        if any(not 0.5 <= v <= 2.0 for v in self.size_scale):
            raise ValueError("objects.*.size_scale: each component in [0.5, 2.0]")
        return self


class TableParams(_Frozen):
    top_z_m: float | None = _f(
        None,
        unit="m",
        doc="table top height; None = fit to the hand path",
        affects=("sim", "plan"),
        ge=0.3,
        le=1.2,
    )


def _p(default, unit, doc, **bounds):
    return _f(default, unit=unit, doc=doc, affects=("sim", "plan"), **bounds)


class TrajectoryParams(_Frozen):
    """Constants of `scripts/retarget_estimated.py` that shape the planned trajectory (all `plan`)."""

    squeeze_m: float = _p(0.012, "m", "least squeeze: tip gap = object width - squeeze", ge=0, le=0.03)
    grip_min_n: float = _p(4.0, "N", "least grip force the squeeze is sized for", ge=0, le=30)
    carry_accel_m_s2: float = _p(3.0, "m/s^2", "held-path acceleration limit", ge=0.5, le=10)
    approach_frames: int = _p(30, "frames", "approach window before the grasp", ge=1, le=120)
    open_margin_m: float = _p(0.02, "m", "open aperture margin over the object width", ge=0, le=0.06)
    open_pad_m: float = _p(0.015, "m", "Sharpa pad allowance added to the open gap", ge=0, le=0.04)
    pad_reach_m: float = _p(0.015, "m", "reach of the finger pad above the commanded tip", ge=0, le=0.04)
    tip_floor_m: float = _p(0.006, "m", "lowest the tips may close above the table", ge=0, le=0.03)
    max_open_extra_m: float = _p(0.04, "m", "open aperture beyond the centre chord", ge=0, le=0.1)
    close_ramp_frames: int = _p(6, "frames", "open to squeezed ramp", ge=1, le=30)
    close_lead_frames: int = _p(12, "frames", "start closing before the object is seen lifted", ge=0, le=60)
    hover_clearance_m: float = _p(0.015, "m", "hover height above the object top", ge=0, le=0.1)
    max_hover_rise_m: float = _p(0.06, "m", "highest hover above the grasp point", ge=0, le=0.2)
    place_dwell_frames: int = _p(15, "frames", "pinch holds still after the release", ge=0, le=90)
    place_lead_frames: int = _p(20, "frames", "receptacle frame takes over before the release", ge=0, le=90)
    place_xy_lead_frames: int = _p(25, "frames", "receptacle xy offset fade-in", ge=0, le=90)
    release_open_frames: int = _p(6, "frames", "fingers open over this many frames", ge=1, le=30)
    retreat_frames: int = _p(8, "frames", "retreat rise duration", ge=1, le=60)
    retreat_height_m: float = _p(0.04, "m", "retreat rise", ge=0, le=0.15)
    retreat_blend_frames: int = _p(10, "frames", "blend back onto the human path", ge=0, le=60)
    bare_place_lift_m: float = _p(0.003, "m", "bare-table put-down height above rest", ge=0, le=0.03)
    max_offset_m: float = _p(0.20, "m", "object/receptacle-relative shift cap", ge=0.02, le=0.5)
    start_search_m: float = _p(0.03, "m", "start-offset search half range", ge=0, le=0.1)
    min_object_thickness_m: float = _p(0.04, "m", "below this the pinch lands on the table", ge=0, le=0.1)
    slab_max_width_m: float = _p(0.025, "m", "pinch width below which the slab proxy collides", ge=0, le=0.06)
    table_margin_m: float = _p(0.30, "m", "table margin beyond the hand reach", ge=0, le=1.0)
    table_inset_m: float = _p(0.05, "m", "carried object stays this far inside the rim", ge=0, le=0.2)
    table_near_edge_m: float = _p(0.10, "m", "closest the table may come to the pelvis", ge=0, le=0.4)
    scene_table_margin_m: float = _p(0.08, "m", "table reach past the outermost scene object", ge=0, le=0.4)
    obstacle_margin_m: float = _p(0.02, "m", "wrist keep-off margin around objects", ge=0, le=0.1)
    held_margin_m: float = _p(0.02, "m", "carried object keep-off margin around objects", ge=0, le=0.1)
    held_clearance_m: float = _p(0.005, "m", "carried object clearance above objects", ge=0, le=0.05)
    place_clearance_m: float = _p(
        0.01, "m", "release height above the receptacle surface under the object", ge=0, le=0.08
    )
    receptacle_opening_margin_m: float = _p(
        0.0, "m", "inner wall of a receptacle's tub collision pulled out by this much", ge=0, le=0.05
    )
    place_retry_budget_s: float = _p(
        1800.0, "s", "wall time after which no place retry pass starts (0 = never retry)", ge=0, le=7200
    )
    place_reach_weight_m: float = _p(
        0.1,
        "m",
        "stance score: wrist error one fully unreachable place window is worth (0 = off)",
        ge=0,
        le=1,
    )
    place_stance_search: bool = _p(True, "bool", "re-score the standing spot on the corrected place path")
    scene_objects: bool = _p(True, "bool", "simulate every inventory object")
    adjust_stance: bool = _p(True, "bool", "move the robot only if the view needs it")
    allow_held_box_fallback: bool = _p(
        False, "bool", "true lets a held object without any mesh be simulated as a default box"
    )


class SceneParams(_Frozen):
    """Scene-inventory role assignment and registration (`stages/reconstruction/scene/roles.py`)."""

    source_from: Literal["grip", "name"] = _p(
        "grip",
        "enum",
        "source object: `grip` = the object under the grip wins over the task name, `name` = the name wins",
    )
    max_registration_m: float = _p(
        0.15, "m", "largest grip-to-source shift trusted; beyond it the inventory is dropped", ge=0.02, le=0.5
    )


class SimParams(_Frozen):
    version: Literal["sim_params/1"] = SIM_PARAMS_VERSION
    objects: dict[Selector, ObjectPhysics] = Field(default_factory=dict)
    table: TableParams = TableParams()
    scene: SceneParams = SceneParams()
    contact: ContactParams = ContactParams()
    pd: PDParams = PDParams()
    trajectory: TrajectoryParams = TrajectoryParams()
