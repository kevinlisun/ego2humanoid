"""Render `feedback.md` (spec section 9): the one short file the agent must read."""

from contracts.real2sim import FeedbackEvent, FeedbackHint, Real2SimFeedback
from skills.residuals import SCALES

MAX_LINES = 60
MAX_TIMELINE = 26
MAX_HINTS = 14
ARROW = {"up": "↑", "down": "↓", "set": "=", "tune": "~"}
RESIDUAL_ORDER = tuple(SCALES)


def _num(v: object) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _cm(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f} cm"


def _qa_line(fb: Real2SimFeedback) -> str:
    q = fb.qa
    carry = _cm(q.translation_m)
    if q.expected_translation_m is not None:
        carry += f" (need {100 * q.expected_translation_m:.1f})"
    return f"QA: lift {_cm(q.lift_height_m)}, carry {carry}, place err {_cm(q.place_error_m)}"


def _residual_text(name: str, value: float | None) -> str:
    label = (
        "tactile 1-IoU"
        if name == "tactile_iou"
        else name.removesuffix("_m").removesuffix("_s").removesuffix("_deg")
    )
    if value is None:
        return f"{label} —"
    if name.endswith("_m"):
        shown = f"{100 * value:.1f} cm"
    elif name.endswith("_deg"):
        shown = f"{value:.0f} deg"
    elif name.endswith("_s"):
        shown = f"{value:.2f} s"
    else:
        shown = f"{value:.2f}"
    return f"{label} {shown} ({min(value / SCALES[name], 1.0):.2f})"


def _event_line(e: FeedbackEvent) -> str:
    text = f"  {e.t_s:.2f} s {e.phase:<8}{e.kind}"
    if e.object:
        text += f" {e.object}"
    return f"{text} {e.detail}".rstrip()


def _hint_line(h: FeedbackHint) -> str:
    rng = f" [{_num(h.range[0])}, {_num(h.range[1])}]" if h.range else ""
    cur = _num(list(h.current)) if isinstance(h.current, (list, tuple)) else _num(h.current)
    return f"  {h.param} {cur} → {ARROW[h.direction]}{rng} ({'+'.join(h.affects)})"


def render_feedback_md(
    fb: Real2SimFeedback, changed_vs_parent: list[tuple[str, object, object]] | None = None
) -> str:
    verdict = "OK" if fb.qa.success else "FAIL"
    reasons = f" ({', '.join(fb.qa.failure_reasons)})" if fb.qa.failure_reasons else ""
    parent = f" (parent {fb.parent_iter})" if fb.parent_iter is not None else ""
    tier = "rollout-only" if fb.tier_run == "rollout" else "full"
    lines = [
        f"# clip {fb.clip} · {fb.hand} · iter {fb.iter}{parent} · {tier} · params {fb.params_hash[:4]}…"
        f"  → {verdict} {fb.match_score:.2f}{reasons}",
        "Reference is monocular: 'matches the video' = matches video estimates (UNVERIFIED_MONOCULAR).",
        _qa_line(fb),
        "Residuals (normalized): "
        + " · ".join(_residual_text(k, fb.residuals.get(k)) for k in RESIDUAL_ORDER),
        "Timeline:",
    ]
    shown = fb.events[:MAX_TIMELINE]
    lines += [_event_line(e) for e in shown] or ["  (no events)"]
    if len(fb.events) > len(shown):
        lines.append(f"  … {len(fb.events) - len(shown)} more events in feedback.json")
    lines.append("Hints (affects):")
    lines += [_hint_line(h) for h in fb.hints[:MAX_HINTS]] or ["  (none fired)"]
    if len(fb.hints) > MAX_HINTS:
        lines.append(f"  … {len(fb.hints) - MAX_HINTS} more hints in feedback.json")
    if changed_vs_parent:
        lines.append(
            "Changed vs parent: " + ", ".join(f"{p} {_num(a)} → {_num(b)}" for p, a, b in changed_vs_parent)
        )
    return "\n".join(lines[:MAX_LINES]) + "\n"
