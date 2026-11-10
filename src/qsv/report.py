"""JSON and Markdown reports for a run."""

from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path
from typing import Any

from qsv import __version__
from qsv.load import AssertionResult, LoadResult, summarize

SCHEMA_VERSION = 1


def _f(value: float | None, digits: int = 0, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _pct(fraction: float | None) -> float | None:
    return None if fraction is None else fraction * 100


def build_report(
    load: LoadResult,
    *,
    mode: str,
    transport: str,
    run_id: str,
    assertion_results: list[AssertionResult] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = dt.datetime.fromtimestamp(load.started_unix_ns / 1e9, tz=dt.timezone.utc)
    summary = summarize(load.results)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "qsv", "version": __version__},
        "run": {
            "id": run_id,
            "mode": mode,
            "transport": transport,
            "scenario": load.scenario.name,
            "description": load.scenario.description,
            "sample_rate_hz": load.scenario.sample_rate,
            "callers": load.config.callers,
            "concurrency": load.config.concurrency or load.config.callers,
            "seed": load.config.seed,
            "started_utc": started.isoformat(timespec="seconds"),
            "wall_time_s": round(load.wall_time_s, 3),
            "python": platform.python_version(),
            **(extra_meta or {}),
        },
        "summary": summary,
        "assertions": [a.to_dict() for a in assertion_results or []],
        "calls": [r.to_dict() for r in load.results],
    }


def _stat_row(label: str, s: dict[str, Any]) -> str:
    return (
        f"| {label} | {s['count']} | {_f(s['p50'])} | {_f(s['p95'])} | {_f(s['p99'])} "
        f"| {_f(s['mean'])} | {_f(s['max'])} |"
    )


def render_markdown(report: dict[str, Any], max_call_rows: int = 25) -> str:
    run, s = report["run"], report["summary"]
    lines: list[str] = []
    add = lines.append
    add(f"# qsv report: {run['scenario']}")
    add("")
    add(
        f"Run `{run['id']}` ({run['mode']}) on the `{run['transport']}` transport, "
        f"started {run['started_utc']} UTC."
    )
    if run.get("description"):
        add("")
        add(run["description"])
    add("")
    add("## Run")
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    add(f"| Callers | {run['callers']} (concurrency {run['concurrency']}) |")
    add(f"| Audio | {run['sample_rate_hz']} Hz mono |")
    add(f"| Seed | {run['seed']} |")
    add(f"| Wall time | {run['wall_time_s']} s |")
    add(f"| Calls ok / failed | {s['calls_ok']} / {s['calls_failed']} |")
    add(f"| Turns scored | {s['turns']} |")
    add("")
    add("## Latency")
    add("")
    add("Milliseconds. Turn gap is caller end of speech to agent first audio. Time to first audio is")
    add("measured from the start of the call.")
    add("")
    add("| Metric | n | p50 | p95 | p99 | mean | max |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    add(_stat_row("Turn gap", s["turn_gap_ms"]))
    add(_stat_row("Time to first audio", s["time_to_first_audio_ms"]))
    add("")
    b = s["bargein"]
    add("## Barge-in")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| Interrupting turns | {b['interrupt_turns']} |")
    add(f"| Agent was speaking when interrupted | {b['interruptions_attempted']} |")
    add(f"| Agent yielded | {b['interruptions_yielded']} ({_f(_pct(b['yield_rate']), 0, '%')}) |")
    st = b["stop_latency_ms"]
    add(f"| Stop latency p50 / p95 / p99 | {_f(st['p50'])} / {_f(st['p95'])} / {_f(st['p99'])} ms |")
    fi = s["false_interruptions"]
    add(
        f"| False interruptions (agent cut the caller off) | {fi['count']} "
        f"({_f(_pct(fi['rate_per_turn']), 1, '% of turns')}) |"
    )
    ov = s["overlap_ms_per_call"]
    add(f"| Overlap per call, p50 / p95 | {_f(ov['p50'])} / {_f(ov['p95'])} ms |")
    add("")
    w = s["wer"]
    if w["words"]:
        add("## Word error rate")
        add("")
        add(
            f"Corpus WER {_f(_pct(w['corpus']), 1, '%')} over {w['words']} reference words "
            f"({w['errors']} errors)."
        )
        add("")
    if report["assertions"]:
        add("## Assertions")
        add("")
        add("| Check | Limit | Actual | Result |")
        add("| --- | --- | --- | --- |")
        for a in report["assertions"]:
            actual = "n/a" if a["actual"] is None else f"{a['actual']:.3f}".rstrip("0").rstrip(".")
            add(f"| {a['name']} | {a['limit']} | {actual} | {'pass' if a['passed'] else 'FAIL'} |")
        add("")
    add("## Calls")
    add("")
    add("| Call | Status | Turns | Gap p50 (ms) | Stop (ms) | False int. | Overlap (ms) | WER |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for call in report["calls"][:max_call_rows]:
        m = call["metrics"]
        if not m:
            add(f"| {call['call_id']} | failed | | | | | | |")
            continue
        sm = m["summary"]
        stops = [
            t["barge_in"]["stop_latency_ms"]
            for t in m["turns"]
            if t["barge_in"] and t["barge_in"]["stop_latency_ms"] is not None
        ]
        add(
            f"| {call['call_id']} | ok | {len(m['turns'])} | {_f(sm['turn_gap_ms']['p50'])} "
            f"| {', '.join(f'{x:.0f}' for x in stops) or 'n/a'} | {sm['false_interruptions']} "
            f"| {_f(m['total_overlap_ms'])} | {_f(None if m['wer'] is None else m['wer'] * 100, 1, '%')} |"
        )
    if len(report["calls"]) > max_call_rows:
        add("")
        add(f"Showing {max_call_rows} of {len(report['calls'])} calls. The JSON report has all of them.")
    if s["errors"]:
        add("")
        add("## Errors")
        add("")
        for e in s["errors"]:
            add(f"- {e}")
    add("")
    add("---")
    add(f"Generated by qsv {report['tool']['version']}. Definitions: docs/metrics.md.")
    add("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` into ``out_dir`` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
