from __future__ import annotations

import numpy as np
import pytest

from qsv.audio import frame_samples
from qsv.caller import CallerSession, Phase
from qsv.scenario import parse_scenario
from qsv.vad import EnergyVAD

SR = 8000
FRAME_MS = 20


def drive(session: CallerSession, agent_active_at, max_ticks: int = 5000):
    """Run the caller against a function tick -> bool describing when the agent is audible."""
    frames = []
    while not session.done and session.tick < max_ticks:
        frames.append(session.next_frame(agent_active_at(session.tick)))
    return np.concatenate(frames)


def test_single_turn_speaks_after_initial_silence_then_finishes():
    sc = parse_scenario(
        {
            "name": "a",
            "initial_silence_ms": 400,
            "tail_ms": 200,
            "response_timeout_ms": 500,
            "turns": [{"say": "x", "duration_ms": 1000}],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: False)
    (plan,) = s.plans
    assert plan.start_s == pytest.approx(0.4, abs=0.001)
    assert plan.end_s == pytest.approx(1.4, abs=0.001)
    assert s.timed_out_turns == [0]
    assert s.done


def test_frames_are_fixed_length_and_speech_is_detected_where_planned():
    sc = parse_scenario({"name": "a", "turns": [{"say": "x", "duration_ms": 1000}]})
    s = CallerSession(sc)
    audio = drive(s, lambda t: False)
    assert len(audio) % frame_samples(SR, FRAME_MS) == 0
    from qsv.vad import detect_segments

    (seg,) = detect_segments(audio, SR, vad=EnergyVAD())
    assert seg.start == pytest.approx(s.plans[0].start_s, abs=0.03)
    assert seg.end == pytest.approx(s.plans[0].end_s, abs=0.03)


def test_polite_caller_waits_for_agent_to_finish():
    sc = parse_scenario(
        {
            "name": "a",
            "end_silence_ms": 300,
            "turns": [
                {"say": "x", "duration_ms": 1000},
                {"say": "y", "duration_ms": 500, "pause_ms": 200},
            ],
        }
    )
    s = CallerSession(sc)
    # Agent talks from 2.0 s to 4.0 s.
    drive(s, lambda t: 100 <= t < 200)
    second = s.plans[1]
    assert not second.interrupt
    assert second.start_s >= 4.0 + 0.3 + 0.2 - 0.05


def test_interrupt_starts_at_offset_after_agent_onset():
    sc = parse_scenario(
        {
            "name": "a",
            "turns": [
                {"say": "x", "duration_ms": 1000},
                {"say": "y", "duration_ms": 500, "interrupt_after_ms": 600},
            ],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: 100 <= t < 400)  # agent audible from 2.0 s
    second = s.plans[1]
    assert second.interrupt
    assert second.start_s == pytest.approx(2.0 + 0.6, abs=0.03)


def test_interrupt_is_skipped_when_agent_never_replies():
    sc = parse_scenario(
        {
            "name": "a",
            "response_timeout_ms": 400,
            "turns": [
                {"say": "x", "duration_ms": 500},
                {"say": "y", "duration_ms": 500, "interrupt_after_ms": 100},
            ],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: False)
    assert s.skipped_interrupts == [1]
    assert not s.plans[1].interrupt


def test_greeting_can_be_interrupted():
    sc = parse_scenario(
        {
            "name": "a",
            "wait_for_greeting_ms": 1000,
            "turns": [
                {"say": "x", "duration_ms": 500, "interrupt_after_ms": 300},
            ],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: 10 <= t < 120)  # greeting from 0.2 s
    assert s.plans[0].interrupt
    assert s.plans[0].start_s == pytest.approx(0.5, abs=0.03)


def test_waits_for_greeting_before_a_polite_first_turn():
    sc = parse_scenario(
        {
            "name": "a",
            "wait_for_greeting_ms": 1000,
            "end_silence_ms": 200,
            "turns": [
                {"say": "x", "duration_ms": 500, "pause_ms": 100},
            ],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: 10 <= t < 60)  # greeting 0.2 to 1.2 s
    assert s.plans[0].start_s >= 1.2 + 0.2 + 0.1 - 0.05


def test_agent_still_talking_when_caller_finishes_counts_as_a_response():
    sc = parse_scenario(
        {
            "name": "a",
            "turns": [
                {"say": "x", "duration_ms": 1000},
                {"say": "y", "duration_ms": 500, "interrupt_after_ms": 200},
                {"say": "z", "duration_ms": 500, "interrupt_after_ms": 200},
            ],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: 100 <= t < 600)  # one long agent reply, never yields
    assert [p.interrupt for p in s.plans] == [False, True, True]


def test_max_call_duration_stops_a_stuck_call():
    sc = parse_scenario(
        {
            "name": "a",
            "max_call_s": 2,
            "turns": [{"say": "x", "duration_ms": 500}, {"say": "y", "duration_ms": 500}],
        }
    )
    s = CallerSession(sc)
    drive(s, lambda t: True)  # agent never stops talking
    assert s.done
    assert s.elapsed_s <= 2.05


def test_phase_is_done_at_end():
    sc = parse_scenario(
        {"name": "a", "tail_ms": 100, "response_timeout_ms": 200, "turns": [{"say": "x", "duration_ms": 200}]}
    )
    s = CallerSession(sc)
    drive(s, lambda t: False)
    assert s.phase is Phase.DONE
