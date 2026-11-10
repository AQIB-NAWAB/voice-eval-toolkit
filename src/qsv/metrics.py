"""Metric definitions and the call analyser.

``analyze_call`` takes the two recorded channels (caller and agent) and the plan of
when each scripted caller turn was spoken. It runs VAD on both channels and derives
barge-in, turn-taking and overlap metrics from the detected segments only. Nothing
here looks at agent internals, so the same code scores any transport.

Definitions are in docs/metrics.md.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from qsv.vad import FrameVAD, Segment, detect_segments, make_vad

# --------------------------------------------------------------------------- statistics


def round_floats(obj: Any, digits: int = 2) -> Any:
    """Recursively round floats so reports do not carry 200.00000000000017."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100). ``None`` for empty input."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def describe(values: list[float]) -> dict[str, float | int | None]:
    """Count, mean, min, p50, p95, p99 and max of a list of numbers."""
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return round_floats(
        {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        },
        3,
    )


# --------------------------------------------------------------------------- WER

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation (keeping apostrophes) and collapse whitespace."""
    return " ".join(_PUNCT.sub(" ", text.lower()).replace("_", " ").split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return ``(errors, reference_words)``. Uses jiwer when installed, else a built-in
    Levenshtein over words (identical results)."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0, 0
    if not hyp:
        return n_ref, n_ref
    try:
        import jiwer
    except ImportError:
        return _edit_distance(ref.split(), hyp.split()), n_ref
    out = jiwer.process_words(ref, hyp)
    return int(out.substitutions + out.deletions + out.insertions), n_ref


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    errors, n = word_error_counts(reference, hypothesis)
    return None if n == 0 else errors / n


# --------------------------------------------------------------------------- data classes


@dataclass
class AnalysisConfig:
    """Knobs for the analyser. All durations in milliseconds."""

    min_silence_gap_ms: float = 300.0
    stop_timeout_ms: float = 2000.0
    onset_tolerance_ms: float = 250.0
    vad: str = "energy"
    vad_threshold_db: float = -45.0
    vad_onset_ms: float = 40.0
    vad_hangover_ms: float = 200.0
    frame_ms: int = 20


@dataclass
class TurnPlan:
    """When the caller actually spoke a scripted turn (call time, seconds)."""

    index: int
    start_s: float
    end_s: float
    interrupt: bool = False
    reference_text: str | None = None


@dataclass
class TranscriptEvent:
    """A caller-side transcript the agent produced (from its STT), stamped in call time."""

    time_s: float
    text: str


@dataclass
class BargeIn:
    """Outcome of one interruption."""

    agent_was_speaking: bool
    yielded: bool = False
    stop_latency_ms: float | None = None
    talk_over_ms: float = 0.0


@dataclass
class TurnMetrics:
    index: int
    caller_start_s: float
    caller_end_s: float
    interrupt: bool
    agent_onset_s: float | None = None
    agent_end_s: float | None = None
    turn_gap_ms: float | None = None
    false_interruption: bool = False
    overlap_ms: float = 0.0
    barge_in: BargeIn | None = None
    transcript: str | None = None
    wer: float | None = None
    wer_errors: int | None = None
    wer_words: int | None = None


@dataclass
class CallMetrics:
    turns: list[TurnMetrics] = field(default_factory=list)
    time_to_first_audio_ms: float | None = None
    total_overlap_ms: float = 0.0
    caller_segments: int = 0
    agent_segments: int = 0
    duration_s: float = 0.0
    wer: float | None = None
    wer_errors: int = 0
    wer_words: int = 0

    @property
    def turn_gaps_ms(self) -> list[float]:
        return [t.turn_gap_ms for t in self.turns if t.turn_gap_ms is not None and t.turn_gap_ms >= 0]

    @property
    def stop_latencies_ms(self) -> list[float]:
        return [
            t.barge_in.stop_latency_ms
            for t in self.turns
            if t.barge_in and t.barge_in.yielded and t.barge_in.stop_latency_ms is not None
        ]

    @property
    def false_interruptions(self) -> int:
        return sum(1 for t in self.turns if t.false_interruption)

    @property
    def interruptions_attempted(self) -> int:
        return sum(1 for t in self.turns if t.barge_in and t.barge_in.agent_was_speaking)

    @property
    def interruptions_yielded(self) -> int:
        return sum(1 for t in self.turns if t.barge_in and t.barge_in.yielded)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["summary"] = {
            "turn_gap_ms": describe(self.turn_gaps_ms),
            "bargein_stop_ms": describe(self.stop_latencies_ms),
            "false_interruptions": self.false_interruptions,
            "interruptions_attempted": self.interruptions_attempted,
            "interruptions_yielded": self.interruptions_yielded,
        }
        return round_floats(d)


# --------------------------------------------------------------------------- analysis


def _overlap_total(a: list[Segment], b: list[Segment], lo: float, hi: float) -> float:
    """Total time (s) in [lo, hi] where a segment from ``a`` and one from ``b`` coincide."""
    total = 0.0
    for x in a:
        clipped = Segment(max(x.start, lo), min(x.end, hi))
        if clipped.end <= clipped.start:
            continue
        total += sum(clipped.overlap(y) for y in b)
    return total


def analyze_call(
    caller: np.ndarray,
    agent: np.ndarray,
    sample_rate: int,
    plans: list[TurnPlan],
    *,
    transcripts: list[TranscriptEvent] | None = None,
    config: AnalysisConfig | None = None,
    vad_factory: Any = None,
) -> CallMetrics:
    """Score one recorded call. ``caller`` and ``agent`` are same-rate mono float arrays."""
    cfg = config or AnalysisConfig()

    def make() -> FrameVAD:
        if vad_factory is not None:
            return vad_factory()
        if cfg.vad == "energy":
            return make_vad("energy", sample_rate, threshold_db=cfg.vad_threshold_db)
        return make_vad(cfg.vad, sample_rate)

    seg_kwargs = {
        "frame_ms": cfg.frame_ms,
        "onset_ms": cfg.vad_onset_ms,
        "hangover_ms": cfg.vad_hangover_ms,
    }
    caller_segs = detect_segments(caller, sample_rate, vad=make(), **seg_kwargs)
    agent_segs = detect_segments(agent, sample_rate, vad=make(), **seg_kwargs)

    tol = cfg.onset_tolerance_ms / 1000.0
    min_gap = cfg.min_silence_gap_ms / 1000.0
    metrics = CallMetrics(
        caller_segments=len(caller_segs),
        agent_segments=len(agent_segs),
        duration_s=max(len(caller), len(agent)) / sample_rate,
    )
    if agent_segs:
        metrics.time_to_first_audio_ms = agent_segs[0].start * 1000.0

    # Measured caller extent per turn: caller VAD segments that begin inside the scripted turn.
    extents: list[tuple[float, float]] = []
    for plan in plans:
        inside = [s for s in caller_segs if plan.start_s - tol <= s.start <= plan.end_s]
        if inside:
            extents.append((inside[0].start, max(s.end for s in inside)))
        else:
            extents.append((plan.start_s, plan.end_s))

    # Agent segments that kept playing through an interruption they did not yield to. If one of
    # them later ends during another interruption, that is its natural end, not a barge-in stop.
    survived: set[float] = set()

    for pos, plan in enumerate(plans):
        c_start, c_end = extents[pos]
        next_start = extents[pos + 1][0] if pos + 1 < len(plans) else float("inf")
        tm = TurnMetrics(
            index=plan.index, caller_start_s=c_start, caller_end_s=c_end, interrupt=plan.interrupt
        )

        # The agent segment (if any) that was playing when the caller started speaking.
        active = next((s for s in agent_segs if s.start < c_start and s.end > c_start), None)

        # First agent audio that starts after the caller began this turn and before the next one.
        response = next((s for s in agent_segs if c_start < s.start < next_start), None)
        if response is not None:
            tm.agent_onset_s = response.start
            tm.agent_end_s = response.end
            tm.turn_gap_ms = (response.start - c_end) * 1000.0
            if response.start < c_end + min_gap:
                tm.false_interruption = True

        if plan.interrupt:
            bi = BargeIn(agent_was_speaking=active is not None)
            if active is not None:
                if active.end >= c_end or active.start in survived:
                    survived.add(active.start)
                    bi.yielded = False
                    bi.talk_over_ms = max(0.0, (min(active.end, c_end) - c_start) * 1000.0)
                else:
                    latency = (active.end - c_start) * 1000.0
                    bi.stop_latency_ms = latency
                    bi.yielded = latency <= cfg.stop_timeout_ms
                    bi.talk_over_ms = latency
            tm.barge_in = bi

        tm.overlap_ms = _overlap_total(caller_segs, agent_segs, c_start, min(c_end, next_start)) * 1000.0
        metrics.turns.append(tm)

    metrics.total_overlap_ms = _overlap_total(caller_segs, agent_segs, 0.0, metrics.duration_s) * 1000.0

    _attach_transcripts(metrics, plans, extents, transcripts or [])
    return metrics


def _attach_transcripts(
    metrics: CallMetrics,
    plans: list[TurnPlan],
    extents: list[tuple[float, float]],
    transcripts: list[TranscriptEvent],
) -> None:
    errors = words = 0
    for pos, (plan, tm) in enumerate(zip(plans, metrics.turns)):
        start = extents[pos][0]
        end = extents[pos + 1][0] if pos + 1 < len(extents) else float("inf")
        texts = [e.text for e in sorted(transcripts, key=lambda e: e.time_s) if start <= e.time_s < end]
        if texts:
            tm.transcript = " ".join(t.strip() for t in texts if t.strip())
        if plan.reference_text and transcripts:
            e, n = word_error_counts(plan.reference_text, tm.transcript or "")
            tm.wer_errors, tm.wer_words = e, n
            tm.wer = e / n if n else None
            errors += e
            words += n
    metrics.wer_errors = errors
    metrics.wer_words = words
    metrics.wer = errors / words if words else None
