import json
from types import SimpleNamespace

import pytest

from stages.reconstruction.scene.vlm_relay import RelayClient, RelayPending, relay_status, validate_inbox


def _contents(text="describe", img=b"\x89PNGdata"):
    parts = [
        SimpleNamespace(text=None, inline_data=SimpleNamespace(data=img)),
        SimpleNamespace(text=text, inline_data=None),
    ]
    return [SimpleNamespace(role="user", parts=parts)]


def test_round_trip(tmp_path):
    client = RelayClient(tmp_path)
    with pytest.raises(RelayPending) as e:
        client.models.generate_content_stream(model="m", contents=_contents(), config=None)
    (rid,) = e.value.request_ids
    assert (tmp_path / "outbox" / rid / "images" / "00.png").exists()
    assert relay_status(tmp_path).pending == [rid]
    (tmp_path / "inbox" / f"{rid}.json").write_text(
        json.dumps({"text": "a cup", "answered_by": "copilot-relay"})
    )
    assert validate_inbox(tmp_path) == []
    chunks = client.models.generate_content_stream(model="m", contents=_contents(), config=None)
    assert "".join(c.text for c in chunks) == "a cup"
    assert relay_status(tmp_path).pending == []


def test_pending_not_swallowed_by_except_exception(tmp_path):
    client = RelayClient(tmp_path)
    with pytest.raises(RelayPending):
        try:
            client.answer("m", _contents(), None)
        except Exception:  # noqa: BLE001
            pytest.fail("RelayPending must not be an Exception")


def test_validate_flags_bad_answers(tmp_path):
    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox" / "x.json").write_text(json.dumps({"text": ""}))
    assert len(validate_inbox(tmp_path)) == 2
