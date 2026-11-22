"""Generate contracts/*.schema.json from the Pydantic authority."""

import json
from pathlib import Path

from contracts import EpisodeInput
from contracts.locomanip import ArrivalReport, ManipulationTask, PoseGoal
from contracts.sim_params import SimParams
from stages.reconstruction.objects.twin_schema import AssetSpec

root = Path(__file__).resolve().parents[1]
(root / "contracts" / "episode.schema.json").write_text(
    json.dumps(EpisodeInput.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
)
(root / "contracts" / "locomanip.schema.json").write_text(
    json.dumps(
        {m.__name__: m.model_json_schema() for m in (ManipulationTask, PoseGoal, ArrivalReport)},
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
(root / "contracts" / "twin_asset.schema.json").write_text(
    json.dumps(AssetSpec.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
)
(root / "contracts" / "sim_params.schema.json").write_text(
    json.dumps(SimParams.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
)
