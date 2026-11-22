# docs/notes

Focused notes, track designs and measurements. The pipeline itself is documented in `../TECH_DESIGN.md`.

| File | Content |
|---|---|
| `agentic_real2sim.md` | Mode A playbook (Track RSI): a Copilot CLI session tunes one clip's `SimParams` through `scripts/real2sim_eval.py` |
| `artefacts.md` | Clip artefacts under `data/derived/egodex/<clip>/` and their physical meaning, review video panels |
| `batch_results.md` | Historical batch result tables (from `scripts/summarize_batch.py`) and Blender-only batch snapshots |
| `dwbc_profile.md` | Decoupled WBC capability profile of the G1 asset, measured by `scripts/profile_dwbc.py` (L.3) |
| `egotac.md` | EgoTac integration: per-vertex force and contact from RGB, setup, output |
| `hand_selection.md` | Unitree Dex3 vs Sharpa Wave: grasp quality on clip 104, stability (H.1, H.6) |
| `skills_library.md` | RSI skills library (`src/skills/`) and the recipe from a bug case to a skill |
| `spider_integration.md` | SPIDER integration notes: pipeline, robot assets, mapping to our bundles (T3.3) |
| `target_frame_switching.md` | Switching the hand-eye reference frame during a task via phase-indexed object frames (implemented, K.1-K.7) |
| `track_bt_blender_twin.md` | Track BT design: Blender digital-twin assets and RSI-tuned physics; replaces SimFoundry for scene reconstruction |
| `track_bt_mujoco_blender_link.md` | Track BT design and feasibility of the MuJoCo to Blender link (production pipeline, components) |
| `track_r_agentic_real2sim.md` | Track RSI technical design: tunable `SimParams`, resolver, bundle, probes, feedback, Mode A/B |
| `track_r_baseline.md` | Track RSI regression baseline (RSI.1 / RSI.4 gate) for the retarget stage |
| `simfoundry_conda_forge.patch` | Patch for the SimFoundry installation scripts (e.g. `scripts/installation/faiss_gpu.sh`); not a document |
