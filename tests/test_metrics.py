from __future__ import annotations

import pytest

from conftest import SR, timeline
from qsv.metrics import (
    AnalysisConfig,
    TranscriptEvent,
    TurnPlan,
    analyze_call,
    describe,
    normalize_text,
    percentile,
    word_error_counts,
    word_error_rate,
)


def test_percentile_and_describe():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile([], 50) is None
    d = describe([10, 20, 30])
    assert d["count"] == 3 and d["p50"] == 20 and d["max"] == 30 and d["mean"] == 20
    assert describe([])["p95"] is None


def test_normalize_text():
    assert normalize_text("Hello,  World! It's 5pm.") == "hello world it's 5pm"


def test_wer_basic_cases():
    assert word_error_rate("the cat sat", "the cat sat") == 0
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "the cat") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "the big cat sat") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "") == 1.0
    assert word_error_rate("", "anything") is None


def test_wer_jiwer_and_builtin_agree(monkeypatch):
    pytest.importorskip("jiwer")
    cases = [("a b c d e", "a x c e"), ("hello world", "hello there big world"), ("one", "two")]
    with_jiwer = [word_error_counts(r, h) for r, h in cases]
    import builtins

    real_import = builtins.__import__

    def no_jiwer(name, *a, **k):
        if name == "jiwer":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_jiwer)
    builtin = [word_error_counts(r, h) for r, h in cases]
    assert with_jiwer == builtin


def _analyze(caller_bursts, agent_bursts, plans, total=14.0, **cfg):
    caller = timeline(total, caller_bursts)
    agent = timeline(total, agent_bursts)
    return analyze_call(caller, agent, SR, plans, config=AnalysisConfig(**cfg))


def test_turn_gap_and_time_to_first_audio():
    m = _analyze(
        [(1.0, 3.0)],
        [(3.8, 6.0)],
        [TurnPlan(0, 1.0, 3.0)],
    )
    (t,) = m.turns
    assert t.turn_gap_ms == pytest.approx(800, abs=45)
    assert not t.false_interruption
    assert m.time_to_first_audio_ms == pytest.approx(3800, abs=25)
    assert m.turn_gaps_ms == [t.turn_gap_ms]


def test_barge_in_stop_latency():
    m = _analyze(
        [(1.0, 3.0), (5.0, 6.5)],
        [(3.8, 5.6)],  # agent speaks 3.8 to 5.6; the caller cuts in at 5.0
        [TurnPlan(0, 1.0, 3.0), TurnPlan(1, 5.0, 6.5, interrupt=True)],
    )
    bi = m.turns[1].barge_in
    assert bi is not None and bi.agent_was_speaking and bi.yielded
    assert bi.stop_latency_ms == pytest.approx(600, abs=45)
    assert m.stop_latencies_ms == [bi.stop_latency_ms]
    assert m.turns[1].overlap_ms == pytest.approx(600, abs=45)


def test_agent_that_talks_over_the_caller_does_not_yield():
    m = _analyze(
        [(1.0, 3.0), (5.0, 6.5)],
        [(3.8, 9.0)],
        [TurnPlan(0, 1.0, 3.0), TurnPlan(1, 5.0, 6.5, interrupt=True)],
    )
    bi = m.turns[1].barge_in
    assert bi is not None and bi.agent_was_speaking and not bi.yielded
    assert bi.stop_latency_ms is None
    assert bi.talk_over_ms == pytest.approx(1500, abs=45)
    assert m.stop_latencies_ms == []
    assert m.interruptions_attempted == 1 and m.interruptions_yielded == 0


def test_stop_slower_than_timeout_counts_as_not_yielded():
    m = _analyze(
        [(1.0, 3.0), (5.0, 9.5)],
        [(3.8, 8.0)],
        [TurnPlan(0, 1.0, 3.0), TurnPlan(1, 5.0, 9.5, interrupt=True)],
        stop_timeout_ms=2000,
    )
    bi = m.turns[1].barge_in
    assert bi is not None and not bi.yielded
    assert bi.stop_latency_ms == pytest.approx(3000, abs=45)


def test_interrupt_when_agent_is_already_silent_is_not_scored():
    m = _analyze(
        [(1.0, 3.0), (9.0, 10.0)],
        [(3.8, 5.0)],
        [TurnPlan(0, 1.0, 3.0), TurnPlan(1, 9.0, 10.0, interrupt=True)],
    )
    bi = m.turns[1].barge_in
    assert bi is not None and not bi.agent_was_speaking
    assert m.interruptions_attempted == 0


def test_survived_segment_ending_later_is_not_a_late_stop():
    m = _analyze(
        [(1.0, 3.0), (4.0, 5.0), (6.0, 7.0)],
        [(3.5, 6.5)],
        [
            TurnPlan(0, 1.0, 3.0),
            TurnPlan(1, 4.0, 5.0, interrupt=True),
            TurnPlan(2, 6.0, 7.0, interrupt=True),
        ],
    )
    assert not m.turns[1].barge_in.yielded
    assert not m.turns[2].barge_in.yielded
    assert m.stop_latencies_ms == []


def test_false_interruption_when_agent_starts_before_caller_finishes():
    m = _analyze(
        [(1.0, 4.0)],
        [(3.2, 6.0)],
        [TurnPlan(0, 1.0, 4.0)],
    )
    t = m.turns[0]
    assert t.false_interruption
    assert t.turn_gap_ms == pytest.approx(-800, abs=45)
    assert m.false_interruptions == 1
    assert m.turn_gaps_ms == []  # negative gaps are cut-offs, not latencies
    assert t.overlap_ms == pytest.approx(800, abs=45)


def test_agent_starting_inside_the_min_silence_gap_counts_as_cut_off():
    bursts = {
        "caller_bursts": [(1.0, 3.0)],
        "agent_bursts": [(3.15, 5.0)],
        "plans": [TurnPlan(0, 1.0, 3.0)],
    }
    assert _analyze(**bursts, min_silence_gap_ms=300).false_interruptions == 1
    assert _analyze(**bursts, min_silence_gap_ms=100).false_interruptions == 0


def test_hesitation_gap_is_not_the_end_of_the_turn():
    # Caller speaks 1.0-2.0, pauses, then speaks again 3.0-4.0. Agent starts at 2.6 (inside the pause).
    m = _analyze(
        [(1.0, 2.0), (3.0, 4.0)],
        [(2.6, 5.0)],
        [TurnPlan(0, 1.0, 4.0)],
    )
    t = m.turns[0]
    assert t.caller_end_s == pytest.approx(4.0, abs=0.03)
    assert t.false_interruption


def test_total_overlap_and_counts():
    m = _analyze(
        [(1.0, 3.0), (5.0, 6.0)], [(2.5, 3.5), (5.5, 7.0)], [TurnPlan(0, 1.0, 3.0), TurnPlan(1, 5.0, 6.0)]
    )
    assert m.total_overlap_ms == pytest.approx(1000, abs=60)
    assert m.caller_segments == 2 and m.agent_segments == 2


def test_transcripts_are_attached_per_turn_and_scored():
    caller = timeline(12.0, [(1.0, 3.0), (6.0, 8.0)])
    agent = timeline(12.0, [(4.0, 5.0), (9.0, 10.0)])
    plans = [
        TurnPlan(0, 1.0, 3.0, reference_text="book a table for four"),
        TurnPlan(1, 6.0, 8.0, reference_text="seven thirty please"),
    ]
    events = [TranscriptEvent(3.6, "book a table for"), TranscriptEvent(8.6, "seven thirty please")]
    m = analyze_call(caller, agent, SR, plans, transcripts=events)
    assert m.turns[0].transcript == "book a table for"
    assert m.turns[0].wer == pytest.approx(1 / 5)
    assert m.turns[1].wer == 0
    assert m.wer_words == 8 and m.wer_errors == 1
    assert m.wer == pytest.approx(1 / 8)


def test_no_transcripts_means_no_wer():
    m = _analyze([(1.0, 3.0)], [(3.8, 5.0)], [TurnPlan(0, 1.0, 3.0, reference_text="hello there")])
    assert m.wer is None and m.turns[0].wer is None
