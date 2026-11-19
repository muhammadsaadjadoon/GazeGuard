"""Pure-Python offline session analytics and self-contained HTML reports.

Reads summaries only; never loads or requests camera, audio, biometrics or frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class SessionSummary:
    name: str
    duration_seconds: float
    observable_seconds: float
    attentive_percent: float | None
    gestures: dict[str, int]
    speech_alerts: int
    states_seconds: dict[str, float]


def read_session(path: Path) -> SessionSummary:
    """Strictly validate session JSON; do not deserialize arbitrary Python objects."""
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Session must be a JSON object")
    duration = float(obj["duration_seconds"])
    observable = float(obj["observable_seconds"])
    pct = obj.get("attentive_percent_observable")
    pct = float(pct) if pct is not None else None
    gestures = obj.get("gestures", {})
    states = obj.get("states_seconds", {})
    if (not 0 <= observable <= duration + 0.05 or duration < 0
            or pct is not None and not 0 <= pct <= 100
            or not isinstance(gestures, dict) or not isinstance(states, dict)):
        raise ValueError("Invalid session metrics")
    import math
    if not all(math.isfinite(v) for v in (duration, observable, *([] if pct is None else [pct]))):
        raise ValueError("Nonfinite session metrics")
    return SessionSummary(
        name=path.stem, duration_seconds=duration, observable_seconds=observable,
        attentive_percent=pct,
        gestures={str(k): int(v) for k, v in gestures.items()},
        speech_alerts=int(obj.get("speech_alerts", 0)),
        states_seconds={str(k): float(v) for k, v in states.items()},
    )


def load_history(folder: Path, limit: int = 100) -> tuple[list[SessionSummary], list[str]]:
    if limit < 1 or limit > 10_000:
        raise ValueError("limit must be between 1 and 10000")
    result: list[SessionSummary] = []
    warnings: list[str] = []
    # Only specific session summary JSON; exclude calibration/settings, no recursion.
    if not folder.is_dir():
        return result, warnings
    for path in sorted(folder.glob("session_*.json"), reverse=True)[:limit]:
        try:
            result.append(read_session(path))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            warnings.append(f"Skipped {path.name}: {exc}")
    return list(reversed(result)), warnings


def aggregate(sessions: list[SessionSummary]) -> dict:
    observed = sum(s.observable_seconds for s in sessions)
    attentive = sum(s.states_seconds.get("ATTENTIVE", 0) for s in sessions)
    # Prefer time-weighted percentage, not an unweighted average of sessions.
    score = round(100 * min(attentive, observed) / observed, 1) if observed else None
    return {
        "sessions": len(sessions),
        "total_seconds": round(sum(s.duration_seconds for s in sessions), 1),
        "observable_seconds": round(observed, 1),
        "attention_percent": score,
        "speech_alerts": sum(s.speech_alerts for s in sessions),
        "gestures": sum(sum(s.gestures.values()) for s in sessions),
    }


def render_html(sessions: list[SessionSummary], target: float = 80.0) -> str:
    if not 0 <= target <= 100:
        raise ValueError("target must be 0..100")
    agg = aggregate(sessions)
    percent_text = "N/A" if agg["attention_percent"] is None else f'{agg["attention_percent"]:.1f}%'
    chart_rows = []
    table_rows = []
    for s in sessions:
        score = s.attentive_percent
        pct = 0 if score is None else score
        label = escape(s.name)
        color = "#15b8a6" if score is not None and score >= target else "#e6a34b"
        chart_rows.append(f'<div class="bar-row"><span title="{label}">{label[-23:]}</span>'
                          f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
                          f'<b>{"N/A" if score is None else f"{score:.1f}%"}</b></div>')
        table_rows.append(f'<tr><td>{label}</td><td>{s.duration_seconds/60:.1f}</td>'
                          f'<td>{s.observable_seconds/60:.1f}</td>'
                          f'<td>{"N/A" if score is None else f"{score:.1f}%"}</td>'
                          f'<td>{sum(s.gestures.values())}</td><td>{s.speech_alerts}</td></tr>')
    cards = [("Sessions", str(agg["sessions"])),
             ("Total minutes", f'{agg["total_seconds"]/60:.1f}'),
             ("Time-weighted eye contact", percent_text),
             ("Gesture events", str(agg["gestures"])),
             ("Possible speech alerts", str(agg["speech_alerts"]))]
    cards_html = "".join(f'<div class="metric"><small>{escape(k)}</small><strong>{escape(v)}</strong></div>' for k, v in cards)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GazeGuard | Session Analytics</title>
<style>
:root{{--bg:#0b1220;--card:#131e32;--line:#26354b;--ink:#eaf2ff;--muted:#9ab0ca}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,Arial,sans-serif}}
main{{max-width:1080px;margin:auto;padding:40px 24px}}h1{{font-size:2.1rem;margin-bottom:5px}}h2{{font-size:1.2rem}}
p,small{{color:var(--muted)}}.eyebrow{{color:#33d3c3;text-transform:uppercase;letter-spacing:2px;font-weight:700}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:26px 0}}
.metric,.section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}}
.metric small{{display:block}}.metric strong{{display:block;font-size:1.7rem;margin-top:10px}}
.section{{margin:20px 0;overflow-x:auto}}.bar-row{{display:grid;grid-template-columns:190px 1fr 65px;gap:14px;align-items:center;margin:13px 0;font-size:12px}}
.bar-row span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.track{{height:13px;background:#2c3d56;border-radius:10px;overflow:hidden}}
.fill{{height:100%;border-radius:10px}}table{{border-collapse:collapse;width:100%;min-width:650px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line)}}th{{color:var(--muted)}}
footer{{color:var(--muted);padding-top:18px}}@media(max-width:680px){{.bar-row{{grid-template-columns:92px 1fr 50px}}main{{padding:24px 12px}}}}
</style></head><body><main><div class="eyebrow">Local analytics · privacy first</div>
<h1>GazeGuard <span style="color:#2fd3c0">/ Analytics</span></h1><p>Generated {created} · Target {target:.0f}% · Based on stored session summaries</p>
<div class="metrics">{cards_html}</div>
<section class="section"><h2>Session attention trend</h2>{''.join(chart_rows) if chart_rows else '<p>No sessions yet. Run the monitor and save a session first.</p>'}</section>
<section class="section"><h2>Session breakdown</h2><table><thead><tr><th>Session</th><th>Duration (min)</th><th>Observed (min)</th><th>Eye contact</th><th>Gestures</th><th>Speech alerts</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></section>
<footer>Private by default: this report contains aggregate monitoring data, not webcam video, audio, iris images, or biometric templates. Eye-contact and speech are visual estimates, not medical or scientific measurements.</footer>
</main></body></html>'''


def save_html(path: Path, sessions: list[SessionSummary], target: float = 80.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    from os import replace
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(render_html(sessions, target), encoding="utf-8")
        replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
