"""Concurrent simulated callers and run-level aggregation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qsv.metrics import AnalysisConfig, describe
from qsv.scenario import Assertions, Scenario
from qsv.session import CallResult, run_call
from qsv.transports import TransportFactory


@dataclass
class LoadConfig:
    callers: int = 5
    concurrency: int | None = None  # defaults to all callers at once
    ramp_up_s: float = 0.0
    seed: int = 1
    realtime: bool | None = None
    keep_audio_for: int = 0  # keep waveforms for the first N calls
    agent_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadResult:
    scenario: Scenario
    results: list[CallResult]
    started_unix_ns: int
    wall_time_s: float
    config: LoadConfig


async def run_load(
    scenario: Scenario,
    factory: TransportFactory,
    config: LoadConfig | None = None,
    *,
    analysis: AnalysisConfig | None = None,
    on_result: Callable[[CallResult], None] | None = None,
    run_id: str = "run",
) -> LoadResult:
    """Run ``config.callers`` independent calls, at most ``config.concurrency`` at once.

    Every call gets its own transport and a distinct seed (``seed + index``), so jitter and
    synthetic audio differ between callers but a whole run is reproducible.
    """
    cfg = config or LoadConfig()
    if cfg.callers < 1:
        raise ValueError("callers must be at least 1")
    limit = cfg.concurrency or cfg.callers
    sem = asyncio.Semaphore(max(1, limit))
    started_ns = time.time_ns()
    t0 = time.perf_counter()
    results: list[CallResult | None] = [None] * cfg.callers

    async def one(index: int) -> None:
        if cfg.ramp_up_s > 0 and cfg.callers > 1:
            await asyncio.sleep(cfg.ramp_up_s * index / (cfg.callers - 1))
        async with sem:
            res = await run_call(
                scenario,
                factory(),
                call_id=f"{run_id}-{index + 1:03d}",
                seed=cfg.seed + index,
                analysis=analysis,
                agent_options=cfg.agent_options,
                realtime=cfg.realtime,
                keep_audio=index < cfg.keep_audio_for,
            )
        results[index] = res
        if on_result:
            on_result(res)

    await asyncio.gather(*(one(i) for i in range(cfg.callers)))
    return LoadResult(
        scenario=scenario,
        results=[r for r in results if r is not None],
        started_unix_ns=started_ns,
        wall_time_s=time.perf_counter() - t0,
        config=cfg,
    )


def summarize(results: list[CallResult]) -> dict[str, Any]:
    """Aggregate per-call metrics into run-level numbers (percentiles pool all turns)."""
    ok = [r for r in results if r.ok and r.metrics]
    gaps: list[float] = []
    stops: list[float] = []
    ttfa: list[float] = []
    overlaps: list[float] = []
    false_int = attempted = yielded = turns = interrupt_turns = 0
    wer_errors = wer_words = 0
    per_call_wer: list[float] = []
    for r in ok:
        m = r.metrics
        assert m is not None
        gaps += m.turn_gaps_ms
        stops += m.stop_latencies_ms
        if m.time_to_first_audio_ms is not None:
            ttfa.append(m.time_to_first_audio_ms)
        overlaps.append(m.total_overlap_ms)
        false_int += m.false_interruptions
        attempted += m.interruptions_attempted
        yielded += m.interruptions_yielded
        turns += len(m.turns)
        interrupt_turns += sum(1 for t in m.turns if t.interrupt)
        wer_errors += m.wer_errors
        wer_words += m.wer_words
        if m.wer is not None:
            per_call_wer.append(m.wer)
    n = len(results)
    return {
        "calls": n,
        "calls_ok": len(ok),
        "calls_failed": n - len(ok),
        "success_rate": (len(ok) / n) if n else 0.0,
        "turns": turns,
        "time_to_first_audio_ms": describe(ttfa),
        "turn_gap_ms": describe(gaps),
        "bargein": {
            "interrupt_turns": interrupt_turns,
            "interruptions_attempted": attempted,
            "interruptions_yielded": yielded,
            "yield_rate": (yielded / attempted) if attempted else None,
            "stop_latency_ms": describe(stops),
        },
        "false_interruptions": {
            "count": false_int,
            "rate_per_turn": (false_int / turns) if turns else None,
        },
        "overlap_ms_per_call": describe(overlaps),
        "wer": {
            "corpus": (wer_errors / wer_words) if wer_words else None,
            "words": wer_words,
            "errors": wer_errors,
            "per_call": describe(per_call_wer),
        },
        "errors": sorted({r.error for r in results if r.error}),
    }


@dataclass
class AssertionResult:
    name: str
    limit: float
    actual: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "limit": self.limit, "actual": self.actual, "passed": self.passed}


def evaluate_assertions(assertions: Assertions, summary: dict[str, Any]) -> list[AssertionResult]:
    """Check a summary against limits. A missing measurement fails a ceiling check, because
    "no data" should not read as "within budget"."""
    checks: list[AssertionResult] = []

    def ceiling(name: str, limit: float | None, actual: float | None) -> None:
        if limit is None:
            return
        checks.append(AssertionResult(name, limit, actual, actual is not None and actual <= limit))

    ceiling(
        "max_bargein_stop_p95_ms",
        assertions.max_bargein_stop_p95_ms,
        summary["bargein"]["stop_latency_ms"]["p95"],
    )
    ceiling("max_turn_gap_p95_ms", assertions.max_turn_gap_p95_ms, summary["turn_gap_ms"]["p95"])
    ceiling(
        "max_false_interruptions", assertions.max_false_interruptions, summary["false_interruptions"]["count"]
    )
    ceiling("max_wer", assertions.max_wer, summary["wer"]["corpus"])
    if assertions.min_success_rate is not None:
        actual = summary["success_rate"]
        checks.append(
            AssertionResult(
                "min_success_rate", assertions.min_success_rate, actual, actual >= assertions.min_success_rate
            )
        )
    return checks
