import importlib

import pytest

from skills.attribution import RULES, param_exists
from skills.library import SKILLS, available

LISTED = [s for s in SKILLS if available(s)]


def test_names_and_modules_are_unique() -> None:
    assert len({s.name for s in SKILLS}) == len(SKILLS)
    assert len({s.module for s in SKILLS}) == len(SKILLS)


def test_every_skill_module_in_the_package_is_registered() -> None:
    from pathlib import Path

    import skills

    files = {p.stem for p in Path(skills.__file__).parent.glob("*.py")} - {"__init__", "library"}
    assert files <= {s.module.split(".")[1] for s in SKILLS}


@pytest.mark.parametrize("skill", LISTED, ids=lambda s: s.name)
def test_available_modules_import(skill) -> None:
    importlib.import_module(skill.module)


@pytest.mark.parametrize("skill", LISTED, ids=lambda s: s.name)
def test_listed_knobs_exist_and_events_have_rules(skill) -> None:
    assert all(param_exists(k.format(name="obj")) for k in skill.knobs)
    emitting = [e for e in skill.events if e in RULES]
    assert skill.name != "reachability_check" or emitting == list(skill.events)
