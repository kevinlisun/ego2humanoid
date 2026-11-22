# Documentation

This directory holds the design documentation of Ego2Humanoid. Start with `TECH_DESIGN.md`: it describes the pipeline as built, and when it disagrees with the code, the code is right.
Per-topic notes and track designs are in `notes/` (`notes/README.md`).

## Top-level documents

| File | Content |
|---|---|
| `TECH_DESIGN.md` | Technical design as built: scope and invariants, end-to-end flow, perception (RGB only), retargeting, simulation, constants |
| `RETARGETING.md` | Contributor guide to retargeting: concepts, step-by-step flow, embodiment-gap handling, debugging tips, where to change what |
| `ARCHITECTURE.md` | System shape, coordinate and physical conventions, robot models, anti-cheating replay contract |
| `IMPLEMENTATION_PLAN.md` | Milestone status against the original plan and the fallbacks that were taken |
| `DATA_CONTRACTS.md` | Artifact / array contract v1.2 (NPZ shapes, validity, `SimRunBundle` evidence) for the stage CLI path |
| `API_EXAMPLES.md` | Integration examples and failure contracts of the archived v0.2 target architecture (adapter stubs raise `NotImplementedError`) |
| `COPILOT_RELAY.md` | How a Copilot session answers SimFoundry VLM requests through the file relay (exit code 75), plus the Track BT appearance loop |
| `DEVBOX.md` | Devbox facts: storage rule, Python environment, switching GPU |
| `LICENSES.md` | Licence register of components, models and data (non-commercial research) |
| `REFERENCES.md` | Shared bibliography; every document cites against it in author–year form |

## Subdirectories

| Directory | Content |
|---|---|
| `notes/` | Design notes, tracks, measurements and one patch (index in `notes/README.md`) |
| `assets/` | Images referenced by the READMEs (`assets/README.md`) |
| `archive/` | Original plans kept for reference only: `ARCHITECTURE_v0.2_plan.md`, `IMPLEMENTATION_PLAN_v0.3.md`, `TECH_DESIGN_v0.3_plan.md` |

Repository-level documents outside this directory: `README.md`, `run_pipeline.md` (runner options), `HARNESS.md`
(rules), `TASKS.md` (status and open questions), `data/README.md` (data layout), `contracts/README.md`.
