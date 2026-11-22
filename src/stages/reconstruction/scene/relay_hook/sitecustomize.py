"""Auto-imported by every SimFoundry stage process when this directory is on PYTHONPATH.

Replaces ``simfoundry.models.vlm.genai.Client`` with the file-based Copilot relay (D8) and turns an
unanswered request into exit code 75. Stdlib only: runs inside SimFoundry's conda envs.
"""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_RELAY_DIR = os.environ.get("E2H_RELAY_DIR")
_TARGET = "simfoundry.models.vlm"


def _load_relay():
    path = Path(__file__).resolve().parents[1] / "vlm_relay.py"
    spec = importlib.util.spec_from_file_location("_e2h_vlm_relay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_e2h_vlm_relay"] = mod
    spec.loader.exec_module(mod)
    return mod


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, inner, relay):
        self._inner, self._relay = inner, relay

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        module.genai = SimpleNamespace(Client=self._relay.RelayClient.factory(_RELAY_DIR))
        # The relay needs no credentials; skip Vertex/API-key resolution.
        module.resolve_gemini_auth = lambda **_: ({}, "relay")


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, relay):
        self._relay = relay

    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, self._relay)
        return spec


def _install():
    relay = _load_relay()
    sys.meta_path.insert(0, _Finder(relay))
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, relay.RelayPending):
            sys.stderr.write(f"RELAY PENDING: {exc}\n")
            sys.stderr.flush()
            os._exit(relay.PENDING_RELAY_EXIT_CODE)
        previous(exc_type, exc, tb)

    sys.excepthook = hook


if _RELAY_DIR:
    _install()
