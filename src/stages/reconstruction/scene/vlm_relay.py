"""File-based Copilot relay that stands in for the Gemini client of SimFoundry.

Pipeline code never calls an external VLM API (D8). A request that has no answer yet is
written to ``outbox/<id>/`` and :class:`RelayPending` is raised; a Copilot session answers
it by writing ``inbox/<id>.json``; re-running the stage then consumes the answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PENDING_RELAY_EXIT_CODE = 75


class RelayPending(BaseException):
    """Raised when a request awaits a Copilot answer.

    Derives from BaseException so that retry loops catching ``Exception`` do not swallow it.
    """

    def __init__(self, request_ids: list[str]):
        super().__init__(f"{len(request_ids)} relay request(s) pending: {', '.join(request_ids)}")
        self.request_ids = request_ids


def _split_parts(contents: Any) -> tuple[list[str], list[bytes]]:
    texts: list[str] = []
    images: list[bytes] = []
    for content in contents if isinstance(contents, (list, tuple)) else [contents]:
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None) is not None:
                texts.append(part.text)
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None) is not None:
                images.append(bytes(inline.data))
    return texts, images


def request_id(model: str, texts: list[str], images: list[bytes], config: Any = None) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    for t in texts:
        h.update(b"\x00T" + t.encode())
    for im in images:
        h.update(b"\x00I" + hashlib.sha256(im).digest())
    for key in ("temperature", "top_p", "seed"):
        h.update(f"\x00{key}={getattr(config, key, None)}".encode())
    return h.hexdigest()[:16]


class _Models:
    def __init__(self, client: RelayClient):
        self._client = client

    def generate_content_stream(self, model: str, contents: Any, config: Any = None):
        return self._client.answer(model, contents, config)


class RelayClient:
    """Drop-in replacement for ``google.genai.Client`` backed by a relay directory."""

    def __init__(self, relay_dir: str | Path | None = None, **_ignored: Any):
        self.relay_dir = Path(relay_dir) if relay_dir is not None else self._default_dir
        self.models = _Models(self)

    _default_dir: Path = Path("relay")

    @classmethod
    def factory(cls, relay_dir: str | Path):
        """Return a ``genai.Client``-compatible callable bound to ``relay_dir``."""
        return lambda **kwargs: cls(relay_dir=relay_dir, **kwargs)

    @property
    def outbox(self) -> Path:
        return self.relay_dir / "outbox"

    @property
    def inbox(self) -> Path:
        return self.relay_dir / "inbox"

    def answer(self, model: str, contents: Any, config: Any = None) -> list[Any]:
        texts, images = _split_parts(contents)
        rid = request_id(model, texts, images, config)
        reply = self.inbox / f"{rid}.json"
        if reply.exists():
            text = json.loads(reply.read_text())["text"]
            return [SimpleNamespace(text=text, candidates=None, prompt_feedback=None)]
        self._write_request(rid, model, texts, images, config)
        raise RelayPending([rid])

    def _write_request(
        self, rid: str, model: str, texts: list[str], images: list[bytes], config: Any
    ) -> None:
        d = self.outbox / rid
        (d / "images").mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(images):
            (d / "images" / f"{i:02d}.png").write_bytes(data)
        (d / "prompt.md").write_text("\n\n".join(texts))
        meta = {
            "id": rid,
            "model": model,
            "n_images": len(images),
            "temperature": getattr(config, "temperature", None),
            "seed": getattr(config, "seed", None),
        }
        (d / "request.json").write_text(json.dumps(meta, indent=2))
        self.inbox.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RelayStatus:
    pending: list[str]
    answered: list[str]


def relay_status(relay_dir: str | Path) -> RelayStatus:
    root = Path(relay_dir)
    asked = sorted(p.name for p in (root / "outbox").glob("*") if p.is_dir())
    answered = {p.stem for p in (root / "inbox").glob("*.json")}
    return RelayStatus([a for a in asked if a not in answered], [a for a in asked if a in answered])


def validate_inbox(relay_dir: str | Path) -> list[str]:
    """Return human-readable problems with answered files (empty list when all are valid)."""
    problems = []
    for f in sorted((Path(relay_dir) / "inbox").glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{f.name}: invalid JSON ({e})")
            continue
        if not isinstance(data.get("text"), str) or not data["text"].strip():
            problems.append(f"{f.name}: missing non-empty 'text'")
        if data.get("answered_by") != "copilot-relay":
            problems.append(f"{f.name}: 'answered_by' must be 'copilot-relay'")
    return problems


def submit_request(
    relay_dir: str | Path,
    rid: str,
    prompt_md: str,
    images: list[bytes | Path],
    meta: dict[str, Any],
) -> Path:
    """Write ``outbox/<rid>/{prompt.md,request.json,images/}`` for a non-VLM relay request."""
    root = Path(relay_dir)
    d = root / "outbox" / rid
    (d / "images").mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        if isinstance(im, Path):
            (d / "images" / f"{i:02d}{im.suffix or '.png'}").write_bytes(im.read_bytes())
        else:
            (d / "images" / f"{i:02d}.png").write_bytes(im)
    (d / "prompt.md").write_text(prompt_md)
    (d / "request.json").write_text(json.dumps({"id": rid, **meta, "n_images": len(images)}, indent=2))
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    return d


def read_answer(relay_dir: str | Path, rid: str) -> dict[str, Any] | None:
    """Return ``inbox/<rid>.json`` (``{"answered_by", "text"}``) or None while unanswered."""
    reply = Path(relay_dir) / "inbox" / f"{rid}.json"
    if not reply.exists():
        return None
    data = json.loads(reply.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{reply.name}: answer must be a JSON object")
    return data
