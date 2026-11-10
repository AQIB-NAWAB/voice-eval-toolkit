from __future__ import annotations

import json
import time

import pytest

from conftest import SCENARIOS, run
from qsv.load import LoadConfig, evaluate_assertions, run_load, summarize
from qsv.report import build_report, render_markdown, write_reports
from qsv.scenario import Assertions, load_scenario, parse_scenario
from qsv.transports import get_transport_factory
from qsv.transports.loopback import LoopbackTransport


def basic_load(callers=5, **kw):
    sc = load_scenario(SCENARIOS / "basic.yaml")
    return run(run_load(sc, LoopbackTransport, LoadConfig(callers=callers, **kw), run_id="t"))


def test_five_concurrent_callers_all_succeed():
    res = basic_load(5)
    assert len(res.results) == 5
    assert all(r.ok for r in res.results), [r.error for r in res.results]
    assert [r.call_id for r in res.results] == [f"t-00{i}" for i in range(1, 6)]
    s = summarize(res.results)
    assert s["calls"] == 5 and s["calls_ok"] == 5 and s["success_rate"] == 1.0
    assert s["turns"] == 15
    assert s["turn_gap_ms"]["count"] == 15


def test_percentiles_are_ordered_and_jitter_spreads_them():
    s = summarize(basic_load(20).results)
    g = s["turn_gap_ms"]
    assert g["min"] <= g["p50"] <= g["p95"] <= g["p99"] <= g["max"]
    assert g["max"] - g["min"] > 50  # jitter_ms 150 in basic.yaml


def test_load_is_reproducible_for_a_seed():
    a = summarize(basic_load(6, seed=9).results)["turn_gap_ms"]
    b = summarize(basic_load(6, seed=9).results)["turn_gap_ms"]
    c = summarize(basic_load(6, seed=10).results)["turn_gap_ms"]
    assert a == b and a != c


def test_concurrency_limit_is_respected():
    active = {"now": 0, "peak": 0}

    class Counting(LoopbackTransport):
        async def connect(self, ctx):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await super().connect(ctx)

        async def close(self):
            if self._agent is not None:
                active["now"] -= 1
            await super().close()

    sc = load_scenario(SCENARIOS / "basic.yaml")
    run(run_load(sc, Counting, LoadConfig(callers=6, concurrency=2)))
    assert active["peak"] == 2


def test_ramp_up_staggers_starts():
    sc = parse_scenario(
        {"name": "q", "turns": [{"say": "x", "duration_ms": 300}], "tail_ms": 0, "response_timeout_ms": 100}
    )
    t0 = time.perf_counter()
    res = run(run_load(sc, LoopbackTransport, LoadConfig(callers=3, ramp_up_s=0.4)))
    assert time.perf_counter() - t0 >= 0.39
    assert len(res.results) == 3


def test_failed_calls_are_counted_and_do_not_stop_the_run():
    class Flaky(LoopbackTransport):
        count = 0

        async def connect(self, ctx):
            Flaky.count += 1
            if Flaky.count % 2 == 0:
                raise ConnectionError("agent not found")
            await super().connect(ctx)

    sc = load_scenario(SCENARIOS / "basic.yaml")
    res = run(run_load(sc, Flaky, LoadConfig(callers=4)))
    s = summarize(res.results)
    assert s["calls_ok"] == 2 and s["calls_failed"] == 2 and s["success_rate"] == 0.5
    assert any("agent not found" in e for e in s["errors"])


def test_invalid_caller_count():
    sc = load_scenario(SCENARIOS / "basic.yaml")
    with pytest.raises(ValueError):
        run(run_load(sc, LoopbackTransport, LoadConfig(callers=0)))


def test_barge_in_scenario_summary():
    sc = load_scenario(SCENARIOS / "barge_in.yaml")
    res = run(run_load(sc, LoopbackTransport, LoadConfig(callers=3)))
    s = summarize(res.results)
    assert s["bargein"]["interrupt_turns"] == 9
    assert s["bargein"]["interruptions_yielded"] == 9
    assert s["bargein"]["stop_latency_ms"]["p95"] == pytest.approx(220, abs=60)
    checks = evaluate_assertions(sc.assertions, s)
    assert checks and all(c.passed for c in checks)


def test_wer_scenario_reports_corpus_wer():
    sc = load_scenario(SCENARIOS / "telephony_wer.yaml")
    s = summarize(run(run_load(sc, LoopbackTransport, LoadConfig(callers=10))).results)
    assert s["wer"]["words"] == 10 * 30
    assert 0.02 < s["wer"]["corpus"] < 0.3


def test_assertions_pass_fail_and_missing_data():
    summary = {
        "success_rate": 0.9,
        "bargein": {"stop_latency_ms": {"p95": 300.0}},
        "turn_gap_ms": {"p95": None},
        "false_interruptions": {"count": 2},
        "wer": {"corpus": 0.1},
    }
    out = evaluate_assertions(
        Assertions(
            max_bargein_stop_p95_ms=250,
            max_turn_gap_p95_ms=1000,
            max_false_interruptions=2,
            max_wer=0.2,
            min_success_rate=0.95,
        ),
        summary,
    )
    by = {c.name: c.passed for c in out}
    assert by == {
        "max_bargein_stop_p95_ms": False,
        "max_turn_gap_p95_ms": False,  # no data fails a ceiling
        "max_false_interruptions": True,
        "max_wer": True,
        "min_success_rate": False,
    }
    assert evaluate_assertions(Assertions(), summary) == []


def test_report_json_and_markdown(tmp_path):
    sc = load_scenario(SCENARIOS / "barge_in.yaml")
    res = run(run_load(sc, LoopbackTransport, LoadConfig(callers=2), run_id="rep"))
    checks = evaluate_assertions(sc.assertions, summarize(res.results))
    report = build_report(res, mode="load", transport="loopback", run_id="rep", assertion_results=checks)
    jp, mp = write_reports(report, tmp_path / "out")
    data = json.loads(jp.read_text())
    assert data["schema_version"] == 1
    assert data["run"]["scenario"] == "barge-in" and data["run"]["callers"] == 2
    assert len(data["calls"]) == 2
    assert data["summary"]["bargein"]["interruptions_yielded"] == 6
    md = mp.read_text()
    for heading in ("# qsv report: barge-in", "## Latency", "## Barge-in", "## Assertions", "## Calls"):
        assert heading in md
    assert "rep-001" in md and "pass" in md
    assert "\u2014" not in md and "\u2013" not in md


def test_markdown_truncates_long_call_lists():
    res = basic_load(4)
    report = build_report(res, mode="load", transport="loopback", run_id="x")
    md = render_markdown(report, max_call_rows=2)
    assert "Showing 2 of 4 calls" in md


def test_factory_roundtrip_through_load():
    sc = load_scenario(SCENARIOS / "basic.yaml")
    res = run(run_load(sc, get_transport_factory("loopback"), LoadConfig(callers=2)))
    assert all(r.ok for r in res.results)
