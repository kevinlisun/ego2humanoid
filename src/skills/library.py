"""Registry of the RSI skills: what each one detects, which knobs it hints at, which bug class it solves.

Data only. Modules are named, never imported, so the table loads without MuJoCo and lists skills
whose module is not merged yet (``available`` tells). Docs: ``docs/notes/skills_library.md``.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    module: str
    purpose: str
    events: tuple[str, ...] = ()  # FeedbackEvent kinds it emits
    knobs: tuple[str, ...] = ()  # SimParams paths its hints name (`{name}` = object selector)
    bug_class: str = ""


SKILLS: tuple[Skill, ...] = (
    Skill(
        "feedback",
        "skills.feedback",
        "Spatio-temporal event detectors on rollout and contact-probe arrays",
        (
            "SIM_UNSTABLE",
            "EJECT",
            "DROP",
            "AIRBORNE_RELEASE",
            "STUCK",
            "SLIP",
            "ROTATE_IN_GRASP",
            "PENETRATION",
            "GRASP_PENETRATION",
            "START_EMBEDDED",
            "DISTURBED",
            "TRACKING_ERROR",
            "GRIP_FORCE",
            "FIRST_TOUCH",
            "LAST_TOUCH",
            "LIFT_OFF",
            "CONTACT_ON",
            "CONTACT_OFF",
        ),
        (),
        "any physics failure that needs a time stamp and a phase",
    ),  # fmt: skip
    Skill(
        "attribution",
        "skills.attribution",
        "Failure pattern -> suggested SimParams knobs (one rule table, current value, range, re-run cost)",
        (),
        ("contact.solref", "trajectory.squeeze_m", "objects.held.friction", "objects.{name}.pos_offset_m"),
        "a detected event with no knob attached",
    ),
    Skill(
        "residuals",
        "skills.residuals",
        "Sim-vs-video residuals and the scalar match score",
        (),
        (),
        "sim and video disagree without a physics failure (carry path, timing, lift, place error)",
    ),
    Skill(
        "final_pose",
        "skills.final_pose",
        "Is the object left upright, at rest and at the right height after the place?",
        ("FINAL_TILTED", "FINAL_NOT_AT_REST", "FINAL_HEIGHT"),
        ("trajectory.place_clearance_m", "trajectory.bare_place_lift_m"),
        "object tipped, still sliding or floating at the end of the episode",
    ),
    Skill(
        "place_check",
        "skills.place_check",
        "Is the held mesh hidden inside the receptacle mesh after the place?",
        ("PLACE_BURIED",),
        ("trajectory.place_clearance_m", "trajectory.held_clearance_m", "objects.receptacle.pos_offset_m"),
        "collision proxy differs from the visual mesh: quiet physics, object invisible in the render",
    ),
    Skill(
        "reachability_check",
        "skills.reachability_check",
        "Is the place target inside the arm's workspace at the chosen stance? (BF.9)",
        ("PLACE_UNREACHABLE", "STANCE_GRASP_ONLY"),
        ("trajectory.place_reach_weight_m", "trajectory.place_stance_search", "trajectory.max_offset_m"),
        "kinematic limit: stance chosen for the grasp, place target out of reach (clip 117)",
    ),
    Skill(
        "real2sim_report",
        "skills.real2sim_report",
        "Builds feedback.json and feedback.md from one rollout: events, residuals, hints, QA",
        (),
        (),
        "glue: turns a rollout into the one report the agent reads",
    ),
    Skill(
        "feedback_md",
        "skills.feedback_md",
        "Renders the short feedback.md the agent must read",
        (),
        (),
        "report readability",
    ),
    Skill(
        "physics_fidelity",
        "skills.physics_fidelity",
        "MuJoCo-side fidelity checks on the full-rate sim state",
        (),
        (),
        "sim state that looks wrong independent of the renderer",
    ),
    Skill(
        "body_state",
        "skills.body_state",
        "Records body poses and exports the constant model part for the Blender replay",
        (),
        (),
        "renderer needs the sim state without a second physics engine",
    ),
    Skill(
        "twin_compare",
        "skills.twin_compare",
        "Scores a real RGB frame against its rendered digital-twin frame (hand regions masked)",
        (),
        (),
        "twin appearance differs from the video",
    ),
    Skill(
        "scene_layout",
        "skills.scene_layout",
        "Role swap, rebased receptacle and large grip gap in the scene inventory (BF.6, bugfix-a)",
        (),
        (),
        "source and receptacle roles swapped, scene differs from the human demo",
    ),
    Skill(
        "asset_provenance",
        "skills.asset_provenance",
        "Which mesh the held object really uses (BF.3, bugfix-b)",
        (),
        (),
        "silent fallback to a default box instead of the perceived mesh",
    ),
    Skill(
        "receptacle_check",
        "skills.receptacle_check",
        "Does the receptacle collision shape hold the object? (BF.5, bugfix-c)",
        (),
        (),
        "object placed at the goal but outside the receptacle's collision shape",
    ),
)


def module_path(skill: Skill, src: Path | None = None) -> Path:
    root = src if src is not None else Path(__file__).resolve().parents[1]
    return root / (skill.module.replace(".", "/") + ".py")


def available(skill: Skill, src: Path | None = None) -> bool:
    return module_path(skill, src).exists()
