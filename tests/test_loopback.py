from __future__ import annotations

import numpy as np
import pytest

from conftest import run
from qsv.metrics import AnalysisConfig
from qsv.scenario import builtin_bargein_scenario, parse_scenario
from qsv.session import run_call
from qsv.transports import CallContext, get_transport_factory
from qsv.transports.loopback import LoopbackConfig, LoopbackTransport, degrade_text


def call(scenario, **kw):
    return run(run_call(scenario, LoopbackTransport(), call_id="t-1", seed=kw.pop("seed", 1), **kw))


def sc(turns, **extra):
    return parse_scenario({"name": "t", "turns": turns, **extra})


def test_config_from_options_coerces_strings():
    cfg = LoopbackConfig.from_options(
        {"barge_in": "false", "stop_latency_ms": "150", "response_ms": "1000;2000"}
    )
    assert cfg.barge_in is False and cfg.stop_latency_ms == 150.0 and cfg.response_ms == [1000.0, 2000.0]
    with pytest.raises(ValueError, match="unknown loopback agent option"):
        LoopbackConfig.from_options({"nope": 1})


def test_degrade_text_is_seeded_and_lossy(rng):
    text = "one two three four five six seven eight nine ten"
    a = degrade_text(text, 0.5, np.random.default_rng(3))
    b = degrade_text(text, 0.5, np.random.default_rng(3))
    assert a == b != text
    assert degrade_text(text, 0.0, rng) == text


def test_polite_call_measures_the_configured_turn_gap():
    s = sc([{"say": "x", "duration_ms": 1500}], agent={"endpointing_ms": 400, "response_delay_ms": 300})
    r = call(s)
    assert r.ok, r.error
    gap = r.metrics.turns[0].turn_gap_ms
    assert gap == pytest.approx(700, abs=60)
    assert r.metrics.false_interruptions == 0


def test_barge_in_stop_latency_tracks_the_agent_setting():
    base = {"response_ms": [5000], "barge_in_min_speech_ms": 100}
    fast = call(builtin_bargein_scenario([500]), agent_options={**base, "stop_latency_ms": 60})
    slow = call(builtin_bargein_scenario([500]), agent_options={**base, "stop_latency_ms": 400})
    f = fast.metrics.stop_latencies_ms
    s = slow.metrics.stop_latencies_ms
    assert len(f) == len(s) == 1
    assert f[0] == pytest.approx(160, abs=45)
    assert s[0] == pytest.approx(500, abs=45)
    assert s[0] - f[0] == pytest.approx(340, abs=40)


def test_agent_without_barge_in_never_yields():
    r = call(builtin_bargein_scenario([500, 900]), agent_options={"barge_in": "false", "response_ms": "4000"})
    assert r.ok
    assert r.metrics.interruptions_attempted >= 1
    assert r.metrics.interruptions_yielded == 0
    assert r.metrics.stop_latencies_ms == []
    assert r.metrics.total_overlap_ms > 500


def test_impatient_endpointing_cuts_the_caller_off_during_a_hesitation():
    turns = [{"say": "x", "duration_ms": 3000, "hesitation": {"at_ms": 1200, "duration_ms": 900}}]
    impatient = call(sc(turns, agent={"endpointing_ms": 250, "response_delay_ms": 100}))
    patient = call(sc(turns, agent={"endpointing_ms": 1200, "response_delay_ms": 100}))
    assert impatient.metrics.false_interruptions == 1
    assert patient.metrics.false_interruptions == 0
    assert patient.metrics.turns[0].turn_gap_ms > 1000


def test_late_returning_caller_cancels_a_reply_that_has_not_started():
    # endpointing 300 ms and 700 ms of reply delay: the 900 ms pause ends before the reply plays.
    turns = [{"say": "x", "duration_ms": 3000, "hesitation": {"at_ms": 1200, "duration_ms": 600}}]
    r = call(sc(turns, agent={"endpointing_ms": 300, "response_delay_ms": 700}))
    assert r.metrics.false_interruptions == 0


def test_interrupting_a_greeting():
    s = sc(
        [{"say": "x", "duration_ms": 1000, "interrupt_after_ms": 300}],
        wait_for_greeting_ms=1500,
        agent={"greeting_ms": 3000, "greeting_delay_ms": 200},
    )
    r = call(s)
    assert r.ok
    (turn,) = r.metrics.turns
    assert turn.interrupt and turn.barge_in.agent_was_speaking and turn.barge_in.yielded
    assert r.metrics.time_to_first_audio_ms == pytest.approx(200, abs=45)


def test_simulated_transcripts_drive_wer():
    s = sc(
        [{"say": "please book a table for four people tonight", "duration_ms": 2500}] * 3,
        agent={"stt_word_error_rate": 0.0},
    )
    perfect = call(s)
    assert perfect.metrics.wer == 0
    noisy = call(
        sc(
            [{"say": "please book a table for four people tonight", "duration_ms": 2500}] * 6,
            agent={"stt_word_error_rate": 0.4},
        ),
        seed=3,
    )
    assert 0.1 < noisy.metrics.wer < 0.8
    assert noisy.metrics.wer_words == 6 * 8


def test_runs_are_deterministic_for_a_seed_and_differ_across_seeds():
    s = sc([{"say": "x", "duration_ms": 1200}] * 2, agent={"jitter_ms": 200})
    a = call(s, seed=5).metrics.turn_gaps_ms
    b = call(s, seed=5).metrics.turn_gaps_ms
    c = call(s, seed=6).metrics.turn_gaps_ms
    assert a == b and a != c


def test_keep_audio_returns_waveforms():
    r = call(builtin_bargein_scenario([500]), keep_audio=True)
    assert r.caller_audio is not None and r.agent_audio is not None
    assert len(r.caller_audio) == len(r.agent_audio) > 8000


def test_transport_requires_connect_first():
    t = LoopbackTransport()
    with pytest.raises(RuntimeError):
        run(t.send_frame(np.zeros(160, np.float32)))


def test_factory_builds_loopback_and_rejects_unknown():
    assert isinstance(get_transport_factory("loopback")(), LoopbackTransport)
    with pytest.raises(ValueError, match="unknown transport"):
        get_transport_factory("carrier-pigeon")


def test_connect_failure_is_reported_not_raised():
    class Broken(LoopbackTransport):
        async def connect(self, ctx: CallContext) -> None:
            raise ConnectionError("no route")

    r = run(run_call(sc([{"say": "x", "duration_ms": 500}]), Broken(), call_id="b"))
    assert not r.ok and "no route" in r.error


def test_analysis_config_is_respected():
    s = sc([{"say": "x", "duration_ms": 1500}], agent={"endpointing_ms": 300, "response_delay_ms": 100})
    strict = call(s, analysis=AnalysisConfig(min_silence_gap_ms=900))
    assert strict.metrics.false_interruptions == 1


def test_transport_recording_overrides_tick_frames_and_lag_shortens_the_countdown():
    class Delayed(LoopbackTransport):
        live_lag_ms = 100.0

        def agent_recording(self):
            return None

    from qsv.caller import CallerSession

    sc_ = sc([{"say": "x", "duration_ms": 500}, {"say": "y", "duration_ms": 500, "interrupt_after_ms": 600}])
    plain = CallerSession(sc_)
    lagged = CallerSession(sc_, reaction_lag_ms=100)
    for s_ in (plain, lagged):
        while not s_.done:
            s_.next_frame(100 <= s_.tick < 400)
    assert plain.plans[1].start_s - lagged.plans[1].start_s == pytest.approx(0.1, abs=0.021)
    r = run(run_call(sc_, Delayed(), call_id="lag"))
    assert r.ok
