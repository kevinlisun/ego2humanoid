"""SHA256 helpers."""

import hashlib
from pathlib import Path

_BLOCK = 8 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(_BLOCK), b""):
            h.update(block)
    return h.hexdigest()
