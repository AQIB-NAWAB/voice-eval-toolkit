"""Small audio helpers: synthetic speech, resampling, WAV I/O and telephony coding.

Everything is mono float32 in the range [-1, 1] unless a function says otherwise.
Only numpy and the standard library are required.
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 8000
DEFAULT_FRAME_MS = 20
TELEPHONY_RATE = 8000


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float, floor_db: float = -120.0) -> float:
    if amp <= 0:
        return floor_db
    return max(floor_db, float(20.0 * np.log10(amp)))


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(x * x)))


def rms_db(samples: np.ndarray, floor_db: float = -120.0) -> float:
    return amp_to_db(rms(samples), floor_db)


def frame_samples(sample_rate: int, frame_ms: int = DEFAULT_FRAME_MS) -> int:
    return int(sample_rate * frame_ms / 1000)


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000.0))


def silence(
    duration_ms: float,
    sample_rate: int,
    noise_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return silence, optionally with low-level white noise at ``noise_db`` dBFS RMS."""
    n = ms_to_samples(duration_ms, sample_rate)
    if noise_db is None or n == 0:
        return np.zeros(n, dtype=np.float32)
    rng = rng or np.random.default_rng(0)
    return (rng.standard_normal(n) * db_to_amp(noise_db)).astype(np.float32)


def synth_speech(
    duration_ms: float,
    sample_rate: int,
    *,
    seed: int = 0,
    level_db: float = -20.0,
    f0: float = 125.0,
) -> np.ndarray:
    """Generate a speech-like signal: a gliding harmonic voice with a syllabic envelope.

    It is not intelligible speech. It has the properties an energy VAD cares about: a
    voiced spectrum, a syllable-rate amplitude wobble that never drops to silence, and
    short onset/offset ramps. ``level_db`` is the RMS level in dBFS.
    """
    n = ms_to_samples(duration_ms, sample_rate)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    glide_phase = rng.uniform(0, 2 * np.pi)
    f = f0 * (1.0 + 0.12 * np.sin(2 * np.pi * 0.7 * t + glide_phase))
    phase = 2 * np.pi * np.cumsum(f) / sample_rate
    nyquist = sample_rate / 2.0
    sig = np.zeros(n)
    for h in range(1, 9):
        if h * f0 * 1.2 >= nyquist:
            break
        sig += np.sin(h * phase) / h
    syll_phase = rng.uniform(0, 2 * np.pi)
    env = 0.62 + 0.38 * np.sin(2 * np.pi * 3.6 * t + syll_phase)
    sig *= env
    ramp = min(ms_to_samples(5, sample_rate), n // 2)
    if ramp > 0:
        fade = np.linspace(0.0, 1.0, ramp)
        sig[:ramp] *= fade
        sig[-ramp:] *= fade[::-1]
    current = rms(sig)
    if current > 0:
        sig *= db_to_amp(level_db) / current
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def estimate_speech_duration_ms(text: str, words_per_minute: float = 150.0) -> int:
    """Rough speaking time for ``text``; used when a scenario turn gives no duration."""
    words = max(1, len(text.split()))
    return int(round(words / words_per_minute * 60_000))


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample with linear interpolation and a short moving-average anti-alias filter."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    x = samples.astype(np.float64, copy=False)
    if dst_rate < src_rate:
        taps = max(1, int(round(src_rate / dst_rate)))
        if taps > 1:
            kernel = np.ones(taps) / taps
            x = np.convolve(x, kernel, mode="same")
    n_out = int(round(len(x) * dst_rate / src_rate))
    src_pos = np.arange(n_out) * (src_rate / dst_rate)
    out = np.interp(src_pos, np.arange(len(x)), x)
    return out.astype(np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def from_int16(samples: np.ndarray) -> np.ndarray:
    return (samples.astype(np.float32) / 32768.0).astype(np.float32)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file and return mono float32 samples and the sample rate."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV files are supported")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return from_int16(data), rate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples as 16-bit PCM."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(to_int16(samples).tobytes())


def write_stereo_wav(path: str | Path, left: np.ndarray, right: np.ndarray, sample_rate: int) -> None:
    """Write two channels (caller left, agent right) so a call can be inspected in an editor."""
    n = max(len(left), len(right))
    lp = np.zeros(n, dtype=np.float32)
    rp = np.zeros(n, dtype=np.float32)
    lp[: len(left)] = left
    rp[: len(right)] = right
    inter = np.empty(n * 2, dtype=np.int16)
    inter[0::2] = to_int16(lp)
    inter[1::2] = to_int16(rp)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(inter.tobytes())


_MULAW_MU = 255.0


def mulaw_roundtrip(samples: np.ndarray) -> np.ndarray:
    """Encode and decode G.711 style mu-law companding to imitate PSTN quantisation."""
    x = np.clip(samples.astype(np.float64), -1.0, 1.0)
    companded = np.sign(x) * np.log1p(_MULAW_MU * np.abs(x)) / np.log1p(_MULAW_MU)
    quantised = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0
    decoded = np.sign(quantised) * ((1.0 + _MULAW_MU) ** np.abs(quantised) - 1.0) / _MULAW_MU
    return decoded.astype(np.float32)


def to_telephony(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Band-limit to 8 kHz and apply mu-law quantisation, then return at 8 kHz."""
    narrow = resample(samples, sample_rate, TELEPHONY_RATE)
    return mulaw_roundtrip(narrow)


def iter_frames(samples: np.ndarray, frame_len: int) -> Iterator[np.ndarray]:
    """Yield consecutive full frames; a trailing partial frame is zero padded."""
    for start in range(0, len(samples), frame_len):
        frame = samples[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.concatenate([frame, np.zeros(frame_len - len(frame), dtype=frame.dtype)])
        yield frame
