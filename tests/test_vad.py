from __future__ import annotations

import numpy as np
import pytest

from conftest import SR, timeline
from qsv.audio import frame_samples, silence
from qsv.vad import EnergyVAD, SpeechGate, detect_segments, make_vad


def test_energy_vad_classifies_speech_and_silence():
    vad = EnergyVAD()
    n = frame_samples(SR)
    speech = timeline(0.02, [(0.0, 0.02)])[:n]
    assert vad.is_speech(speech)
    assert not vad.is_speech(np.zeros(n, np.float32))


def test_segments_match_ground_truth_within_a_frame():
    x = timeline(5.0, [(0.5, 1.7), (3.0, 4.2)])
    segs = detect_segments(x, SR)
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(0.5, abs=0.021)
    assert segs[0].end == pytest.approx(1.7, abs=0.021)
    assert segs[1].start == pytest.approx(3.0, abs=0.021)
    assert segs[1].end == pytest.approx(4.2, abs=0.021)


def test_hangover_bridges_short_pauses_but_not_long_ones():
    x = timeline(4.0, [(0.5, 1.0), (1.1, 1.6), (2.5, 3.0)])
    segs = detect_segments(x, SR, hangover_ms=200)
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(0.5, abs=0.021)
    assert segs[0].end == pytest.approx(1.6, abs=0.021)


def test_segment_end_excludes_hangover():
    x = timeline(2.0, [(0.5, 1.0)])
    (seg,) = detect_segments(x, SR, hangover_ms=400)
    assert seg.end == pytest.approx(1.0, abs=0.021)


def test_single_frame_blip_is_ignored_by_onset_filter():
    x = silence(1000, SR)
    n = frame_samples(SR)
    x[4000 : 4000 + n] = timeline(0.02, [(0.0, 0.02)])[:n]
    assert detect_segments(x, SR, onset_ms=40) == []
    assert len(detect_segments(x, SR, onset_ms=20)) == 1


def test_gate_active_flips_with_onset_and_hangover():
    gate = SpeechGate(EnergyVAD(), SR, onset_ms=40, hangover_ms=100)
    n = frame_samples(SR)
    speech = timeline(0.2, [(0.0, 0.2)])
    quiet = np.zeros(n, np.float32)
    states = []
    for i in range(10):
        states.append(gate.update(speech[i * n : (i + 1) * n]))
    assert states[:2] == [False, True]
    off = [gate.update(quiet) for _ in range(6)]
    assert off[:4] == [True, True, True, True]
    assert off[4] is False


def test_adaptive_floor_tolerates_a_noisy_line():
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(SR * 2) * 10 ** (-38 / 20)).astype(np.float32)
    fixed = detect_segments(noise, SR, vad=EnergyVAD(-45, adaptive=False))
    assert fixed, "fixed threshold mistakes steady -38 dB noise for speech"
    n = frame_samples(SR)
    adaptive = EnergyVAD(-45, adaptive=True, margin_db=6, initial_floor_db=-38)
    flags = [adaptive.is_speech(noise[i : i + n]) for i in range(0, len(noise) - n, n)]
    assert not any(flags[5:])


def test_make_vad_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_vad("nope", SR)


def test_webrtc_vad_when_installed():
    pytest.importorskip("webrtcvad")
    x = timeline(2.0, [(0.5, 1.5)])
    segs = detect_segments(x, SR, vad=make_vad("webrtc", SR, aggressiveness=1))
    assert segs
    assert segs[0].start == pytest.approx(0.5, abs=0.15)
