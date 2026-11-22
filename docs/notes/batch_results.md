Historical snapshots, not regenerated automatically. The first table comes from `scripts/summarize_batch.py data/derived/egodex <out.md>` (also run at the end of `scripts/run_pipeline.sh`, output `$LOG_DIR/batch_results.md`); the second is a Blender-only batch scored with `physics_fidelity`. Both predate the place-aware 3-pass retarget retry, adaptive settle and the always-simulated scene inventory (bugfix backlog BF.1-BF.11 in `TASKS.md`), so rerun the pipeline for current numbers.

| clip | hand | success | lift m | carry m | contact | selfcol | ik err mm | valid | grid video |
|---|---|---|---|---|---|---|---|---|---|
| 0 | dex3 | no | 0.018 | 0.200 | 313 | 0 | 33.0 | 0.68 | data/derived/egodex/basic_pick_place_0/review/result_grid.mp4 |
| 0 | sharpa | no | 0.000 | 0.000 | 0 | 0 | 5.8 | 0.68 | data/derived/egodex/basic_pick_place_0/review/result_grid_sharpa.mp4 |
| 1 | dex3 | no | 0.029 | 0.229 | 1712 | 0 | 0.7 | 1.00 | data/derived/egodex/basic_pick_place_1/review/result_grid.mp4 |
| 1 | sharpa | no | 0.000 | 0.000 | 0 | 0 | 0.8 | 1.00 | data/derived/egodex/basic_pick_place_1/review/result_grid_sharpa.mp4 |
| 10 | dex3 | no | 0.037 | 0.009 | 337 | 0 | 0.7 | 1.00 | data/derived/egodex/basic_pick_place_10/review/result_grid.mp4 |
| 10 | sharpa | no | 0.169 | 0.175 | 1609 | 0 | 0.7 | 1.00 | data/derived/egodex/basic_pick_place_10/review/result_grid_sharpa.mp4 |
| 100 | dex3 | no | 0.114 | 0.206 | 15 | 0 | 24.5 | 1.00 | data/derived/egodex/basic_pick_place_100/review/result_grid.mp4 |
| 100 | sharpa | no | 0.181 | 0.191 | 21 | 0 | 39.5 | 1.00 | data/derived/egodex/basic_pick_place_100/review/result_grid_sharpa.mp4 |
| 101 | dex3 | no | nan | nan | - | 0 | 0.7 | 1.00 | - |
| 101 | sharpa | no | nan | nan | - | 0 | 4.1 | 1.00 | - |
| 104 | dex3 | yes | 0.117 | 0.292 | 1010 | 52 | 0.7 | 0.80 | data/derived/egodex/basic_pick_place_104/review/result_grid.mp4 |
| 104 | sharpa | yes | 0.154 | 0.322 | 977 | 135 | 0.7 | 0.80 | data/derived/egodex/basic_pick_place_104/review/result_grid_sharpa.mp4 |
| 108 | dex3 | no | 0.000 | 0.025 | 0 | 0 | 0.7 | 1.00 | data/derived/egodex/basic_pick_place_108/review/result_grid.mp4 |
| 108 | sharpa | no | 0.000 | 0.025 | 0 | 0 | 0.7 | 1.00 | data/derived/egodex/basic_pick_place_108/review/result_grid_sharpa.mp4 |
| 114 | dex3 | no | 0.008 | 0.011 | 261 | 0 | 6.4 | 1.00 | data/derived/egodex/basic_pick_place_114/review/result_grid.mp4 |
| 114 | sharpa | yes | 0.056 | 0.280 | 1499 | 21 | 7.4 | 1.00 | data/derived/egodex/basic_pick_place_114/review/result_grid_sharpa.mp4 |
| 116 | dex3 | yes | 0.177 | 0.242 | 550 | 0 | 0.8 | 0.99 | data/derived/egodex/basic_pick_place_116/review/result_grid.mp4 |
| 116 | sharpa | no | 0.007 | 0.039 | 532 | 0 | 0.8 | 0.99 | data/derived/egodex/basic_pick_place_116/review/result_grid_sharpa.mp4 |
| 117 | dex3 | no | 0.018 | 0.123 | 159 | 0 | 16.5 | 0.89 | data/derived/egodex/basic_pick_place_117/review/result_grid.mp4 |
| 117 | sharpa | no | 0.174 | 0.281 | 3148 | 0 | 29.1 | 0.89 | data/derived/egodex/basic_pick_place_117/review/result_grid_sharpa.mp4 |

4/20 rollouts succeeded.

## Blender-only batch, tag `_bt_20261120_1148` (2026-11-20 11:48)

Renderer: Blender (`twin_render.py`); MuJoCo headless. Videos: `review/result_grid_sharpa_bt_20261120_1148.mp4`, `third_blender_…mp4`, `ego_blender_…mp4`.
Columns: pipeline task result, then `physics_fidelity` (max drift mm, lift s, release s, peak pad force N; sensor range 0..8.2 N).

| clip | task result | fidelity | drift mm | lift | release | pad N | problems |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | success | FAIL | 145.4 | 1.78 | 2.62 | 13.0 | cube drift, pad force |
| 1 | success | FAIL | 20.6 | 3.18 | 5.76 | 31.5 | cube drift, pad force |
| 10 | success | FAIL | 2.4 | 2.34 | 3.30 | 35.8 | pad force |
| 100 | success | OK | 2.4 | 1.92 | 7.66 | 5.5 | - |
| 101 | NOT_SETTLED, AIRBORNE_RELEASE | FAIL | 32.5 | 3.28 | 8.14 | 13.8 | cup drift, pad force |
| 104 | NOT_PLACED, AIRBORNE_RELEASE | FAIL | 5.0 | 2.44 | 4.04 | 22.1 | pad force |
| 108 | success (place error NaN) | OK | 2.4 | 0.04 | - | 1.1 | nothing lifted, result not trusted |
| 114 | see log | FAIL | 835.2 | 4.06 | 9.76 | 10.3 | cube sinks 20 mm and drifts 835 mm |
| 116 | AIRBORNE_RELEASE | FAIL | 51.5 | 1.74 | 3.24 | 35.3 | toys drift, pad force |
| 117 | NOT_PLACED, EJECTED, DROPPED_IN_CARRY | FAIL | 2.4 | 0.02 | - | 21.6 | pad force |

Only 2 of 10 clips pass the fidelity report, so the physics is not yet faithful. Pad force is 2–4x above the sensor range on 8 clips (contact stiffness) and objects drift on 5. Clip 108 passes only because nothing was lifted (BT.18). Next: tune MuJoCo contact parameters.
