"""Build a MuJoCo scene: fixed-pelvis G1+Dex3 at ``T_sim_robot_base``, table and rigid objects.

Actuators of the vendored MJCF are raw motors; here they are turned into position
actuators (PD) with per-group gains so a rollout is driven only by joint targets.
"""

import copy
import hashlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field

import mujoco
import numpy as np

from common.transforms import matrix_to_quat_wxyz
from robot.asset import MJCF_PATH, load_fixed_base_spec
from robot.sharpa import attach_sharpa_hands

CONTACT_IMPRATIO = 10.0


@dataclass(frozen=True)
class ObjectContact:
    """Contact model of the scene objects, applied through geom priority.

    MuJoCo mixes the parameters of two touching geoms of equal priority (solref averaged, friction and
    condim maxed). The Sharpa fingertip pads carry the menagerie's soft solref (0.06, 0.9), so a pinched
    object sat in a (0.04, 0.95) contact, sank 4.5-6.5 mm into the pads and was spring-ejected when the
    friction saturated (clip 114). Giving the object geoms a higher priority makes every contact they
    take part in (fingers, table, receptacle) use exactly these values, for any hand model.
    """

    priority: int = 1
    # 4 = sliding + torsional friction: a two-point pinch is a hinge without the torsional term.
    condim: int = 4
    solref: tuple[float, float] = (0.01, 1.0)  # time constant >= 2 dt (0.002 s), critically damped
    solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    # The finger pads are rubber; the old max-combination always gave >= 1.0, so the object's own
    # estimate (0.2-2.0) is floored here to keep that grip.
    min_slide_friction: float = 1.0
    torsion_m: float = 0.01  # torsional friction coefficient (m): about a fingertip pad radius
    roll_m: float = 0.0001

    @classmethod
    def from_params(cls, params) -> "ObjectContact":
        c = params.contact
        return cls(
            priority=c.priority,
            condim=c.condim,
            solref=tuple(c.solref),
            solimp=tuple(c.solimp),
            min_slide_friction=c.min_slide_friction,
            torsion_m=c.torsion_m,
            roll_m=c.roll_m,
        )


@dataclass(frozen=True)
class PDGains:
    kp: float
    kd: float
    max_torque_nm: float
    armature: float = 0.0


@dataclass(frozen=True)
class PDConfig:
    """PD gains per joint group. Groups are matched by substring of the joint name."""

    groups: tuple[tuple[str, PDGains], ...] = (
        ("_hand_", PDGains(kp=8.0, kd=0.4, max_torque_nm=6.0, armature=0.005)),
        ("wrist", PDGains(kp=40.0, kd=2.0, max_torque_nm=20.0, armature=0.01)),
        ("shoulder", PDGains(kp=150.0, kd=6.0, max_torque_nm=60.0, armature=0.02)),
        ("elbow", PDGains(kp=150.0, kd=6.0, max_torque_nm=60.0, armature=0.02)),
        ("waist", PDGains(kp=300.0, kd=10.0, max_torque_nm=100.0, armature=0.02)),
        ("", PDGains(kp=300.0, kd=10.0, max_torque_nm=150.0, armature=0.02)),  # legs
    )
    gravity_compensation: bool = True  # fixed pelvis: robot links carry no net weight
    timestep_s: float = 0.002

    @classmethod
    def from_params(cls, params) -> "PDConfig":
        # Group order matters: `_gains_for` takes the first substring match, "" (legs) is the catch-all.
        g = params.pd.groups
        named = (("_hand_", g.hand), ("wrist", g.wrist), ("shoulder", g.shoulder), ("elbow", g.elbow))
        named += (("waist", g.waist), ("", g.legs))
        return cls(
            groups=tuple((k, PDGains(v.kp, v.kd, v.max_torque_nm, v.armature)) for k, v in named),
            gravity_compensation=params.pd.gravity_compensation,
            timestep_s=params.pd.timestep_s,
        )


@dataclass(frozen=True)
class BoxObject:
    name: str
    half_size_m: tuple[float, float, float]
    pos_m: tuple[float, float, float]
    mass_kg: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.01, 0.001)
    rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0)
    # Optional generated mesh (body-frame vertices (V,3), faces (F,3)); collides as its convex hull.
    mesh_vertices: np.ndarray | None = field(default=None, compare=False)
    mesh_faces: np.ndarray | None = field(default=None, compare=False)
    # Optional per-vertex texture coordinates (V,2; v from the image top) and RGB image (H,W,3 uint8).
    mesh_uv: np.ndarray | None = field(default=None, compare=False)
    mesh_texture: np.ndarray | None = field(default=None, compare=False)
    # Optional convex collision pieces [(vertices (V,3), faces (F,3)), ...] in the body frame. When given, the
    # visual mesh no longer collides (it is concave in general) and carries only the mass.
    collision_pieces: tuple = field(default=(), compare=False)
    # Welded to the world: a receptacle has to stay where it was reconstructed, otherwise the arm
    # shoves it across the table and the place target moves out from under the carried object.
    static: bool = False
    # False: collides with the other objects and the table but not the robot. The container an
    # object is picked out of is welded at its reconstructed place; a 5-finger hand opening around
    # a small object on its floor otherwise lands on the rim and never reaches it (clip 10).
    robot_collision: bool = True


# Collision bits. Robot and table geoms use MuJoCo's default (contype = conaffinity = 1); objects also
# carry bit 2, which is all that robot-exempt bodies collide with.
_OBJECT_BITS = {"contype": 3, "conaffinity": 3}
_ROBOT_EXEMPT_BITS = {"contype": 2, "conaffinity": 2}


@dataclass(frozen=True)
class SceneConfig:
    T_sim_robot_base: np.ndarray = field(default_factory=lambda: np.eye(4))
    table_top_z_m: float = 0.0
    table_center_xy_m: tuple[float, float] = (0.45, 0.0)
    table_half_size_xy_m: tuple[float, float] = (0.35, 0.5)
    table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    hand: str = "dex3"  # dex3 (3-finger, 43 DoF) | sharpa (Sharpa Wave 5-finger tactile, 73 DoF)
    objects: tuple[BoxObject, ...] = ()
    # Optional egocentric camera (sim-from-camera, OpenCV axes) compiled into the scene as "ego".
    # This is the *human* camera recovered from the source video; use it to composite over a plate.
    ego_camera_T_sim: np.ndarray | None = field(default=None, compare=False)
    ego_camera_fovy_deg: float = 45.0
    # Vertical field of view of the robot's own head camera ("d435"); 42.5 deg is the D435 colour
    # sensor at 16:9 (69.4 deg horizontal); render it at D435_IMAGE_HW to keep that aspect.
    d435_fovy_deg: float = 42.5
    pd: PDConfig = field(default_factory=PDConfig)
    # None keeps MuJoCo's default parameter mixing for the object contacts.
    object_contact: ObjectContact | None = field(default_factory=ObjectContact)
    # Post-solve projection of the friction rows: removes the creep the soft friction allows inside
    # the cone (it does not widen the cone). 0 disables it.
    noslip_iterations: int = 5
    impratio: float = CONTACT_IMPRATIO
    cone: str = "elliptic"  # elliptic | pyramidal
    integrator: str = "implicitfast"
    # None: the per-hand table `HEAVY_HAND_WRIST_SCALE`.
    heavy_hand_wrist_scale: float | None = None

    @classmethod
    def from_params(cls, params, **geometry) -> "SceneConfig":
        """Physics fields from `SimParams`; scene geometry (pose, table, objects, hand) passes through."""
        return cls(
            table_friction=tuple(params.contact.table_friction),
            pd=PDConfig.from_params(params),
            object_contact=ObjectContact.from_params(params),
            noslip_iterations=params.contact.noslip_iterations,
            impratio=params.contact.impratio,
            cone=params.contact.cone,
            integrator=params.pd.integrator,
            heavy_hand_wrist_scale=params.pd.heavy_hand_wrist_scale,
            **geometry,
        )


def _sharpa_joint_names() -> set[str]:
    from robot.finger_map import sharpa_joint_names

    return {n for side in ("left", "right") for n in sharpa_joint_names(side)}


def _gains_for(joint_name: str, pd: PDConfig) -> PDGains:
    for key, gains in pd.groups:
        if key in joint_name:
            return gains
    raise KeyError(joint_name)


def _contact_kw(obj: BoxObject, contact: ObjectContact | None) -> dict:
    if contact is None:
        return {"friction": list(obj.friction)}
    return {
        "friction": [max(obj.friction[0], contact.min_slide_friction), contact.torsion_m, contact.roll_m],
        "priority": contact.priority,
        "condim": contact.condim,
        "solref": list(contact.solref),
        "solimp": [*contact.solimp, 0.5, 2.0],
    }


def mesh_shape(spec: mujoco.MjSpec, obj: BoxObject) -> tuple[dict, str | None]:
    """Add ``obj``'s mesh (and texture material) to the spec; return the geom shape kwargs and material."""
    material = None
    if obj.mesh_vertices is None:
        return {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": list(obj.half_size_m)}, None
    mesh = spec.add_mesh(
        name=f"{obj.name}_mesh",
        uservert=np.asarray(obj.mesh_vertices, np.float32).ravel(),
        userface=np.asarray(obj.mesh_faces, np.int32).ravel(),
    )
    if obj.mesh_uv is not None and obj.mesh_texture is not None:
        mesh.usertexcoord = np.asarray(obj.mesh_uv, np.float32).ravel()
        tex = np.ascontiguousarray(obj.mesh_texture, np.uint8)
        texture = spec.add_texture(
            name=f"{obj.name}_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=tex.shape[1],
            height=tex.shape[0],
        )
        texture.nchannel = 3
        texture.data = tex.tobytes()
        spec.add_material(name=f"{obj.name}_mat").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = (
            f"{obj.name}_tex"
        )
        material = f"{obj.name}_mat"
    return {"type": mujoco.mjtGeom.mjGEOM_MESH, "meshname": f"{obj.name}_mesh"}, material


def add_object(spec: mujoco.MjSpec, obj: BoxObject, contact: ObjectContact | None = None) -> None:
    body = spec.worldbody.add_body(name=obj.name, pos=list(obj.pos_m))
    ckw = _contact_kw(obj, contact)
    if not obj.static:
        body.add_freejoint(name=f"{obj.name}_free")
    shape, material = mesh_shape(spec, obj)
    pieces = obj.collision_pieces if obj.mesh_vertices is not None else ()
    bits = _OBJECT_BITS if obj.robot_collision else _ROBOT_EXEMPT_BITS
    body.add_geom(
        name=f"{obj.name}_geom",
        mass=obj.mass_kg,
        **ckw,
        rgba=list(obj.rgba),
        **({"contype": 0, "conaffinity": 0} if pieces else bits),
        **shape,
        **({"material": material} if material else {}),
    )
    for i, (verts, faces) in enumerate(pieces):
        spec.add_mesh(
            name=f"{obj.name}_cvx{i}",
            uservert=np.asarray(verts, np.float32).ravel(),
            userface=np.asarray(faces, np.int32).ravel(),
        )
        body.add_geom(
            name=f"{obj.name}_cvx{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{obj.name}_cvx{i}",
            mass=1e-9,  # the visual geom carries the true mass and inertia
            **ckw,
            **bits,
            group=3,
            rgba=[0, 0, 0, 0],
        )


# The Sharpa Wave hand weighs 1.77 kg against the Dex3's 0.78 kg and is longer, so the wrist PD tuned
# for Dex3 sags by tens of degrees under it. Scale the wrist gains and torque limit with that mass ratio.
HEAVY_HAND_WRIST_SCALE = {"dex3": 1.0, "sharpa": 2.5}

# Unitree G1 head camera (Intel RealSense D435), taken verbatim from the official description
# ``unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf``:
#   <joint name="d435_joint" type="fixed">
#     <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
#     <parent link="torso_link"/><child link="d435_link"/>
# i.e. rigidly mounted on the torso and pitched 47.6 deg down. The MJCF torso_link frame is the same
# frame (its imu_in_torso site matches the URDF), so the offset can be used as is.
D435_PARENT_BODY = "torso_link"
D435_POS_IN_TORSO_M = (0.0576235, 0.01753, 0.42987)
D435_PITCH_RAD = 0.8307767239493009
# Native 16:9 colour mode of the D435 (69.4 x 42.5 deg). A 4:3 render with the same fovy keeps only
# the central 54.8 deg horizontally, i.e. the hand looks 2.35x larger than in the 105 deg human view.
D435_IMAGE_HW = (360, 640)


def _d435_camera_rotation() -> np.ndarray:
    """Rotation of the MuJoCo camera frame in the torso frame.

    The URDF link frame is ROS style (+x forward, +y left, +z up) pitched down about +y. A MuJoCo
    camera looks along -z with +y up, so its right/up/back axes are -y_link / +z_link / -x_link.
    """
    p = D435_PITCH_RAD
    r_y = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
    cam_in_link = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return r_y @ cam_in_link


def build_scene_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    spec = load_fixed_base_spec(MJCF_PATH)
    if cfg.hand == "sharpa":
        attach_sharpa_hands(spec)
    elif cfg.hand != "dex3":
        raise ValueError(f"unknown hand {cfg.hand!r}")
    spec.option.timestep = cfg.pd.timestep_s
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{cfg.integrator.upper()}")
    # Elliptic friction cones with a high impratio, as MuJoCo recommends for grasping. With the
    # defaults (pyramidal, 1; the Sharpa MJCF's own option is dropped on attach) friction is soft
    # enough that a pinched object creeps out of a steady 2 N grip (clip 114 Dex3: a 13 g duck slid
    # down and fell during the carry).
    spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{cfg.cone.upper()}")
    spec.option.impratio = cfg.impratio
    spec.option.noslip_iterations = cfg.noslip_iterations

    pelvis = spec.body("pelvis")
    pelvis.pos = cfg.T_sim_robot_base[:3, 3]
    pelvis.quat = matrix_to_quat_wxyz(cfg.T_sim_robot_base[:3, :3])

    hand_joints = _sharpa_joint_names() if cfg.hand == "sharpa" else set()
    for joint in spec.joints:
        if joint.name in hand_joints:
            continue  # the Sharpa MJCF sets armature/damping/frictionloss per joint class
        gains = _gains_for(joint.name, cfg.pd)
        joint.armature = gains.armature
    for act in spec.actuators:
        if act.target in hand_joints:
            continue  # keep the hand's own position gains and force ranges
        gains = _gains_for(act.target, cfg.pd)
        wrist_scale = cfg.heavy_hand_wrist_scale
        if wrist_scale is None or cfg.hand != "sharpa":  # the override is the Sharpa scale
            wrist_scale = HEAVY_HAND_WRIST_SCALE.get(cfg.hand, 1.0)
        scale = wrist_scale if "wrist" in act.target else 1.0
        act.set_to_position(kp=gains.kp * scale, kv=gains.kd * scale)
        act.forcelimited = True
        act.forcerange = [-gains.max_torque_nm * scale, gains.max_torque_nm * scale]
        act.ctrllimited = False
    if cfg.pd.gravity_compensation:
        for body in spec.bodies:
            if body.name and body.name != "world":
                body.gravcomp = 1.0

    cx, cy = cfg.table_center_xy_m
    hx, hy = cfg.table_half_size_xy_m
    thick = 0.02
    table = spec.worldbody.add_body(name="table", pos=[cx, cy, cfg.table_top_z_m - thick])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, thick],
        friction=list(cfg.table_friction),
        rgba=[0.6, 0.45, 0.3, 1],
    )
    for obj in cfg.objects:
        add_object(spec, obj, cfg.object_contact)
    # The robot's own head camera, rigidly mounted on the torso as in the Unitree description. Unlike
    # the "ego" camera below it moves with the robot, so it is the real egocentric view.
    spec.body(D435_PARENT_BODY).add_camera(
        name="d435",
        pos=list(D435_POS_IN_TORSO_M),
        quat=matrix_to_quat_wxyz(_d435_camera_rotation()),
        fovy=cfg.d435_fovy_deg,
    )
    if cfg.ego_camera_T_sim is not None:
        # Egocentric camera with the measured roll. MuJoCo cameras look along -z with +y up, the
        # estimated pose is OpenCV (+z forward, +y down), hence the 180 deg flip about x.
        T = np.asarray(cfg.ego_camera_T_sim, float)
        R = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        spec.worldbody.add_camera(
            name="ego",
            pos=T[:3, 3],
            quat=matrix_to_quat_wxyz(R),
            fovy=cfg.ego_camera_fovy_deg,
        )
    return spec


_MODEL_CACHE: OrderedDict[bytes, mujoco.MjModel] = OrderedDict()
MODEL_CACHE_SIZE = 16


def _cache_key(cfg: SceneConfig) -> bytes | None:
    # Every field, including the ones excluded from dataclass equality (camera pose, mesh arrays).
    try:
        return hashlib.sha1(pickle.dumps(cfg, protocol=5)).digest()
    except Exception:  # noqa: BLE001 - an unpicklable config is simply compiled every time
        return None


def build_scene(cfg: SceneConfig) -> mujoco.MjModel:
    """Compiled scene of ``cfg``. Compiling the robot takes ~0.7 s and the retargeting stages ask for the
    same scene hundreds of times per clip, so compiled models are memoised per process. Each caller
    gets its own deep copy, so mutating the returned model never leaks into other callers."""
    key = _cache_key(cfg)
    if key is None:
        return build_scene_spec(cfg).compile()
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = build_scene_spec(cfg).compile()
        _MODEL_CACHE[key] = model
        if len(_MODEL_CACHE) > MODEL_CACHE_SIZE:
            _MODEL_CACHE.popitem(last=False)
    else:
        _MODEL_CACHE.move_to_end(key)
    return copy.deepcopy(model)
