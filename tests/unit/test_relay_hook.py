import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

HOOK = Path(__file__).parents[2] / "src/stages/reconstruction/scene/relay_hook"


def test_hook_patches_simfoundry_vlm_and_exits_75(tmp_path):
    pkg = tmp_path / "simfoundry" / "models"
    pkg.mkdir(parents=True)
    (tmp_path / "simfoundry" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "vlm.py").write_text("genai = None\n")
    script = textwrap.dedent(
        """
        from types import SimpleNamespace as N
        from simfoundry.models import vlm
        c = vlm.genai.Client()
        part = N(text="hello", inline_data=None)
        c.models.generate_content_stream(model="m", contents=[N(parts=[part])], config=None)
        """
    )
    env = {
        **os.environ,
        "E2H_RELAY_DIR": str(tmp_path / "relay"),
        "PYTHONPATH": f"{HOOK}{os.pathsep}{tmp_path}",
    }
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 75, r.stderr
    (req,) = (tmp_path / "relay" / "outbox").iterdir()
    assert json.loads((req / "request.json").read_text())["model"] == "m"
