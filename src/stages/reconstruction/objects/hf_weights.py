"""Preflight for Hugging Face weights that perception scripts load from the local cache.

Without ``HF_HOME`` (or ``HF_HUB_CACHE``) pointing at the cache that holds the model, ``from_pretrained``
fails deep inside transformers with an unrelated-looking error (clip 101: "SAM3 missing"). Run as
``python -m stages.reconstruction.objects.hf_weights facebook/sam3`` for a shell preflight.
"""

import os
import sys
from pathlib import Path


def hub_dir(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    return Path(env.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def cached_model(repo_id: str, env: dict[str, str] | None = None) -> bool:
    snaps = hub_dir(env) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


def require_model(repo_id: str, label: str, env: dict[str, str] | None = None) -> None:
    """Exit with a clear message when ``repo_id`` is not in the local Hugging Face cache."""
    if not cached_model(repo_id, env):
        raise SystemExit(
            f"ERROR WEIGHTS_NOT_FOUND: {label} weights ({repo_id}) not found in {hub_dir(env)}: "
            "set HF_HOME to the cache that holds them (in Docker: mount it and pass -e HF_HOME)"
        )


if __name__ == "__main__":
    require_model(sys.argv[1], sys.argv[1].split("/")[-1].upper())
