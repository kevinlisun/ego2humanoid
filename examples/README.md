# `examples/` — contract-path templates

Templates for the stage CLI contract path (`src/contracts/`, `docs/DATA_CONTRACTS.md`). The real-clip
pipeline (`scripts/run_pipeline.sh`) does not need them; it writes its own `episode.json` per clip.

| File | Content |
|---|---|
| `episode.template.json` | A schema 1.2 `EpisodeInput`: video URI, task description, robot `unitree_g1_dex3_v1` (43 DoF, `fixed_pelvis`), `usage_class` `noncommercial_research`, `physics_required`, `scale_policy` |

```bash
PYTHONPATH=src python -m cli.main validate examples/episode.template.json
```

The video URI is a placeholder. Do not put real data or credentials here.
