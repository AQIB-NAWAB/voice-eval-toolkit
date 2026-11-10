from __future__ import annotations

import numpy as np
import pytest

from qsv import audio


def test_synth_speech_level_and_length():
    x = audio.synth_speech(1000, 8000, seed=1, level_db=-20)
    assert len(x) == 8000
    assert audio.rms_db(x) == pytest.approx(-20, abs=0.5)
    assert np.max(np.abs(x)) < 1.0


def test_synth_speech_is_deterministic_per_seed():
    a = audio.synth_speech(300, 8000, seed=5)
    b = audio.synth_speech(300, 8000, seed=5)
    c = audio.synth_speech(300, 8000, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_silence_with_noise_has_requested_level(rng):
    x = audio.silence(2000, 8000, noise_db=-60, rng=rng)
    assert audio.rms_db(x) == pytest.approx(-60, abs=1.0)
    assert not audio.silence(100, 8000).any()


def test_resample_changes_length_and_keeps_tone():
    t = np.arange(16000) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    down = audio.resample(tone, 16000, 8000)
    assert len(down) == 8000
    spectrum = np.abs(np.fft.rfft(down))
    assert np.argmax(spectrum) == pytest.approx(440, abs=2)


def test_wav_roundtrip(tmp_path):
    x = audio.synth_speech(500, 8000, seed=2)
    path = tmp_path / "a.wav"
    audio.write_wav(path, x, 8000)
    y, rate = audio.read_wav(path)
    assert rate == 8000
    assert len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-3


def test_stereo_wav_has_two_channels(tmp_path):
    import wave

    path = tmp_path / "s.wav"
    audio.write_stereo_wav(path, np.ones(100, np.float32) * 0.1, np.zeros(50, np.float32), 8000)
    with wave.open(str(path)) as wf:
        assert wf.getnchannels() == 2
        assert wf.getnframes() == 100


def test_mulaw_roundtrip_is_close_but_quantised():
    x = audio.synth_speech(200, 8000, seed=3, level_db=-18)
    y = audio.mulaw_roundtrip(x)
    assert len(y) == len(x)
    err = audio.rms(x - y)
    assert 0 < err < 0.02


def test_to_telephony_outputs_8khz():
    x = audio.synth_speech(250, 16000, seed=4)
    y = audio.to_telephony(x, 16000)
    assert len(y) == 2000


def test_estimate_speech_duration():
    assert audio.estimate_speech_duration_ms("one two three four five") == 2000
    assert audio.estimate_speech_duration_ms("") == 400


def test_iter_frames_pads_last_frame():
    frames = list(audio.iter_frames(np.ones(250, np.float32), 160))
    assert len(frames) == 2
    assert frames[1][89] == 1.0 and frames[1][90] == 0.0
