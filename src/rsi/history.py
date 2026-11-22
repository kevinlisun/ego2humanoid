"""`history.jsonl` of a real2sim run: one `HistoryEntry` per evaluated iteration."""

from pathlib import Path

from contracts.real2sim import HistoryEntry

HISTORY_FILE = "history.jsonl"


def iter_dir(root: str | Path, n: int) -> Path:
    return Path(root) / f"iter_{n:03d}"


def append_entry(root: str | Path, entry: HistoryEntry) -> None:
    path = Path(root) / HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry.model_dump_json() + "\n")


def read_history(root: str | Path) -> list[HistoryEntry]:
    path = Path(root) / HISTORY_FILE
    if not path.exists():
        return []
    return [HistoryEntry.model_validate_json(ln) for ln in path.read_text().splitlines() if ln.strip()]


def next_iter(root: str | Path) -> int:
    entries = read_history(root)
    return max((e.iter for e in entries), default=-1) + 1


def best_entry(entries: list[HistoryEntry]) -> HistoryEntry | None:
    """Lowest `match_score`, successful runs first; ties keep the earliest iteration."""
    if not entries:
        return None
    return min(entries, key=lambda e: (not e.success, e.match_score, e.iter))


def _keys(patch: dict) -> str:
    return ", ".join(sorted(patch)) or "-"


def format_table(entries: list[HistoryEntry]) -> str:
    rows = [
        "| iter | parent | tier | changed | success | match_score | wall s |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        parent = "-" if e.parent is None else str(e.parent)
        rows.append(
            f"| {e.iter} | {parent} | {e.tier_run} | {_keys(e.patch)} | {'yes' if e.success else 'no'}"
            f" | {e.match_score:.2f} | {e.wall_time_s:.0f} |"
        )
    return "\n".join(rows)
