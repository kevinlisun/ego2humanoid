# RSI skills library

`src/skills/library.py` is the registry: one `Skill` row per module in `src/skills/` (name, module, purpose,
events it emits, `SimParams` knobs it hints at, bug class it solves). It is data only; modules are named, not
imported, so rows for skills that are not merged yet are allowed (`available()` tells). The unit test
`tests/unit/test_skills_library.py` imports every row whose module exists, checks that the listed knobs exist
in `SimParams` and that every module in `src/skills/` has a row.

## Recipe: from a bug case to a skill

Rule (TASKS.md "Bugfix backlog"): fix through `SimParams` / skills first, code second.

1. **Observe.** Find the symptom and a number that separates it from a good run (clip 117: the place window
   is out of IK reach, 19 % of the targets more than 3 cm away even without the place correction).
2. **Event detector.** Write a pure function in a new `src/skills/<name>.py` that turns the measurement into a
   `FeedbackEvent` kind (`PLACE_UNREACHABLE`, `STANCE_GRASP_ONLY`). Threshold constants live in the module.
   If the measurement exists only at plan time, store it in the bundle meta (`extra_meta=` of `save_bundle`)
   and read it through `ClipReference` and `FeedbackInputs`.
3. **Attribution hint.** Add the kind to `SEVERITY` (`feedback.py`), a rule row per candidate knob to `RULES`
   and the firing condition to `fired_patterns` (`attribution.py`). The hint names the knob, direction and why.
4. **SimParams knob.** If no knob can change the behaviour, add one to `contracts/sim_params.py` with unit,
   range and `affects` (`plan` when the trajectory changes), update `configs/sim_params/default.yaml` and the
   knob table in `docs/notes/track_r_agentic_real2sim.md`; `test_sim_params.py` checks the YAML against the
   model. Wire the knob through `stages/retargeting/script_params.py`.
5. **Rerun.** Rerun the clip (`PYTHONPATH=src HF_HOME=... scripts/run_pipeline.sh` / `retarget_estimated.py`)
   with the knob and confirm the event disappears and the QA result flips.
6. **Regression test.** A unit test of the detector on hand-made arrays, plus a parametrized row in
   `test_real2sim_attribution.py` (the pattern fires, every hinted path exists). Only then change code if the
   knob alone cannot fix the class.
7. **Register.** Add the `Skill` row to `library.py` and the bug to the backlog row in `TASKS.md`.
