# `data/` — local data layout (git-ignored)

Everything in `data/` except this file is git-ignored and must never be pushed (see
`HARNESS.md` §5).

```text
data/
  raw/egodex/                 # source EgoDex clips
  derived/egodex/
    basic_pick_place_<n>/     # one dir per clip: the retarget use case
      video.mp4, episode.json, hands_*.npz, objects_sam3.npz, retarget_rgb_sharpa.{json,npz}
      alignment/ gt/ scene/   # contract bundle, EgoDex GT, tabletop inventory
      review/                 # deliverables: result_grid_sharpa.mp4, trajectory*.npz, third/ego_blender.mp4
    loco_manip_<case>/        # one dir per loco-manip use case: result.json, review/<case>.mp4
  scenes/egodex_104/          # SimFoundry reconstruction (only clip 104)
  weights/  envs/             # model weights, conda envs for the simfoundry step
```

Experiment-tagged outputs (`*_bt_<stamp>`, `*_rsi`, `hands_fah_f<focal>` variants, `orig/`,
`real2sim_*/`) are scratch from one-off runs: delete them freely, the pipeline regenerates what it needs.

Large, regenerable render intermediates (`bg_rgbd*.npz`, `ego_d435*.npz`, per-panel `tl/tr/br*.mp4`)
live outside the repo at `/cloudfile/nvme_tmp/egodex/<clip>/panels` (override with `E2H_BULK_DIR`).
Per-stage runs (`DOCKER=1 scripts/run_pipeline.sh`) read the same layout; weights/envs are mounted into the node images.
Every `RENDER=1` run of `scripts/retarget_estimated.py` first deletes that hand's old review files
and panels.
