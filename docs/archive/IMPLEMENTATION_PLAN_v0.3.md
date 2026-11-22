# Implementation Roadmap — Ego2Humanoid v0.3

**Type:** milestone plan. Task-level breakdown, status and owner questions live in `TASKS.md`;
the technical design is `docs/TECH_DESIGN.md`. Effort bands are planning estimates for one
experienced robotics/ML infrastructure engineer, not delivery promises.

## Goal change from v0.2

1. **Single episode end-to-end first, cloud pipeline second.** All MVP work runs on one cloud GPU
   devbox with a local runner; cloud integration (M6) starts only after the MVP exits.
2. **MVP is split:** MVP-A uses HOT3D ground truth (oracle) to isolate robot/retarget/physics;
   MVP-B replaces the oracle with RGB-only perception on the same clip.
3. **MuJoCo is the MVP physics backend** (same MJCF as SPIDER). OmniGibson moves to post-MVP.

## Critical path

```text
M0 foundations (devbox, uv, layout, contracts v1.2, synthetic fixtures, node template)
 ├─ M1 HOT3D ingest + oracle bundles ───────────┐
 └─ M2 G1+Dex3 MuJoCo + replay + QA ────────────┤
                                                ├─ M3 align + SPIDER G1+Dex3 retarget
                                                └─ M4 MVP-A oracle E2E (local)
                                                     └─ M5 MVP-B RGB perception E2E (local)
                                                          └─ M6 Cloud pipeline integration
                                                               └─ M7 post-MVP (OG, batch, cache)
```

M1 and M2 can proceed in parallel. M2 includes the **synthetic physical success fixture**
(scripted cube grasp-lift-place ≥3/5 seeds), which must pass before SPIDER output is debugged.

## Milestones

| Milestone | Scope | Exit evidence | Effort band |
|---|---|---|---|
| M0 Foundations | Devbox spec, uv migration, `ego2humanoid` layout, contracts v1.2, NPZ validators, common utils, synthetic generators, node Dockerfile template, submodules + license register, local runner | Default tests green; `preflight` node image smoke passes | 4–7 d |
| M1 HOT3D + oracle | Clip reader, fisheye→pinhole MP4, GT landmarks, oracle bundles, golden clip selector, gravity check | Golden clip MP4 passes preflight; oracle bundles validate | 3–5 d |
| M2 Robot + MuJoCo | Pinned 43-DoF G1+Dex3 MJCF, fixed pelvis, scene builder, replay with anti-injection guard, QA events/checks, headless render | Synthetic cube fixture ≥3/5 | 4–8 d |
| M3 Align + retarget | Gravity/scale/base placement/reachability, SPIDER study, G1+Dex3 embodiment, converter, export | Golden clip 43-DoF commands, no limit violations, parity with replay MJCF | 5–12 d (research risk) |
| M4 MVP-A | Oracle DAG, SPIDER ctrl replay in MuJoCo, provenance, system test | `PHYSICS_VALIDATED` or documented `PHYSICS_FAILED` | 3–6 d |
| M5 MVP-B | FAH node, scene option decision + node, object 6D tracker, monocular scale, eval metrics | Module tests vs GT; estimated-mode system test | 8–15 d (highest algorithmic risk) |
| M6 Cloud pipeline | Registry images, per-node pipeline components, data upload, CI workflow | MVP-A then MVP-B succeed in the cloud | 3–6 d |
| M7 Post-MVP | OmniGibson backend + parity, batch-10, caching, bimanual | PRD §8/§10 gates | — |

## Fallbacks

| Trigger | Fallback |
|---|---|
| No 5 s HOT3D clip contains the full task | Concatenate consecutive clips from one sequence (`clip_definitions.json`) or relax the `place` gate for MVP-A (owner decision Q5) |
| SPIDER cannot host a combined G1+Dex3 embodiment cleanly | Minimal patch in `third_party_patches/spider/`; worst case arm IK (ours) + SPIDER hand-only retarget with wrist from IK |
| MuJoCo Warp vs CPU MuJoCo replay diverges | Run validation replay in MuJoCo Warp with identical options; document |
| SimFoundry via Copilot relay (S-A) not workable (rounds, parsing, install) | Local S-B: depth + table plane + primitive mesh; local VLM backend later (T7.5) |
| Object 6D tracking unreliable on golden clip | Choose a better-textured clip; never fabricate occluded motion |

## Test fixture ladder

1. Unit tests (CPU) for all non-trivial functions.
2. Smoke tests on synthetic data for every stage and every node image.
3. Synthetic physical success fixture (M2).
4. Module tests on the HOT3D golden clip vs GT (M1–M5).
5. System tests: oracle (M4), estimated (M5), then in the cloud (M6).
