"""Stage interface protocols."""

from pathlib import Path
from typing import Protocol

from contracts.bundles import (
    AlignedInteractionBundle,
    HandTrackBundle,
    ObjectMotionBundle,
    RobotTrajectoryBundle,
    SceneBundle,
)
from contracts.episode import EpisodeInput
from contracts.qa import QAReport, SimRunBundle


class HandEstimator(Protocol):
    def estimate_hands(
        self, episode: EpisodeInput, video_localpath: Path, output_dir: Path
    ) -> HandTrackBundle: ...


class SceneReconstructor(Protocol):
    def reconstruct_scene(
        self, episode: EpisodeInput, video_localpath: Path, output_dir: Path
    ) -> SceneBundle: ...


class ObjectMotionEstimator(Protocol):
    def track_objects(
        self,
        episode: EpisodeInput,
        video_localpath: Path,
        scene: SceneBundle,
        hands: HandTrackBundle,
        output_dir: Path,
    ) -> ObjectMotionBundle: ...


class WorldAligner(Protocol):
    def align(
        self,
        episode: EpisodeInput,
        hands: HandTrackBundle,
        scene: SceneBundle,
        objects: ObjectMotionBundle,
        output_dir: Path,
    ) -> AlignedInteractionBundle: ...


class Retargeter(Protocol):
    def retarget(
        self,
        episode: EpisodeInput,
        hands: HandTrackBundle,
        scene: SceneBundle,
        objects: ObjectMotionBundle,
        aligned: AlignedInteractionBundle,
        output_dir: Path,
    ) -> RobotTrajectoryBundle: ...


class Simulator(Protocol):
    def simulate(
        self,
        episode: EpisodeInput,
        scene: SceneBundle,
        trajectory: RobotTrajectoryBundle,
        output_dir: Path,
    ) -> SimRunBundle: ...


class QualityGate(Protocol):
    def validate(
        self,
        episode: EpisodeInput,
        sim: SimRunBundle,
        trajectory: RobotTrajectoryBundle,
        output_dir: Path,
    ) -> QAReport: ...
