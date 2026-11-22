"""Explicit third-party integration seams: not yet model implementations.

Keep FAH / SPIDER / SimFoundry source as pinned Git submodules. Adapters convert
upstream data into versioned Ego2Humanoid contracts and never claim that
research-only weights or robot configs have been verified.
"""

from pathlib import Path

from contracts import (
    AlignedInteractionBundle,
    EpisodeInput,
    HandTrackBundle,
    ObjectMotionBundle,
    QAReport,
    RobotTrajectoryBundle,
    SceneBundle,
    SimRunBundle,
)


class FAHAdapter:
    def estimate_hands(
        self, episode: EpisodeInput, video_localpath: Path, output_dir: Path
    ) -> HandTrackBundle:
        # FAH video/world/contact; preserve PTS; map per-frame MANO/2D/3D to NPZ;
        # mark monocular world scale uncertain, do not treat contact probabilities as forces.
        raise NotImplementedError("M1: bind pinned FAH with approved weights and canonical exports")


class SimFoundryAdapter:
    def reconstruct_scene(
        self, episode: EpisodeInput, video_localpath: Path, output_dir: Path
    ) -> SceneBundle:
        # SimFoundry Pipeline A; parse OG scene JSON, mesh+collision geometry,
        # physical parameter priors with uncertainty; test headless scene import.
        raise NotImplementedError("M2: bind SimFoundry reconstruction and scene asset export")


class ObjectMotionAdapter:
    def track_objects(
        self,
        episode: EpisodeInput,
        video_localpath: Path,
        scene: SceneBundle,
        hands: HandTrackBundle,
        output_dir: Path,
    ) -> ObjectMotionBundle:
        # Own RGB-only mask/point tracking + object 6D pose fitting; uncertainty & occlusion.
        raise NotImplementedError("M2: implement RGB dynamic object pose tracking")


class RegistrationAdapter:
    def align(
        self,
        episode: EpisodeInput,
        hands: HandTrackBundle,
        scene: SceneBundle,
        objects: ObjectMotionBundle,
        output_dir: Path,
    ) -> AlignedInteractionBundle:
        # Sim(3) scale inference from scene/priors; gravity/table registration;
        # G1 fixed-pelvis reachability; no silent metric ground-truth claim.
        raise NotImplementedError("M3: implement source-to-sim alignment, scale and reachability")


class SPIDERAdapter:
    def retarget(
        self,
        episode: EpisodeInput,
        hands: HandTrackBundle,
        scene: SceneBundle,
        objects: ObjectMotionBundle,
        aligned: AlignedInteractionBundle,
        output_dir: Path,
    ) -> RobotTrajectoryBundle:
        # Map 5 human fingers to 3 Dex3 fingers via object-relative contact;
        # 43-DoF joint ordering from pinned MJCF, actuator/arm-IK + physics checks.
        raise NotImplementedError("M3: implement G1+Dex3 combined SPIDER retarget adapter")


class OmniGibsonReplayAdapter:
    def simulate(
        self, episode: EpisodeInput, scene: SceneBundle, trajectory: RobotTrajectoryBundle, output_dir: Path
    ) -> SimRunBundle:
        # Import exact robot asset mapping, initialize once, send actuator commands;
        # run forward simulation without teleporting object or wrist, record physics.
        raise NotImplementedError("M4: implement G1+Dex3 OmniGibson action-controlled replay")


class PhysicsQAAdapter:
    def validate(
        self, episode: EpisodeInput, sim: SimRunBundle, trajectory: RobotTrajectoryBundle, output_dir: Path
    ) -> QAReport:
        # Numerical event-level validation of lift/translate/place, contacts,
        # joint limits, penetration, physics integration, ≥3/5 seeded trials.
        # Structural helper alone is not sufficient for PHYSICS_VALIDATED status.
        raise NotImplementedError("M4: implement independent physics/event validator")
