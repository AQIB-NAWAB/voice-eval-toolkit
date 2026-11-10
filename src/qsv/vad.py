"""Voice activity detection.

Two layers:

* A per-frame classifier (``FrameVAD``): ``EnergyVAD`` works with numpy alone, and
  ``WebRTCVAD`` wraps the optional ``webrtcvad`` package.
* ``SpeechGate``: turns raw per-frame decisions into stable speech/non-speech state with
  an onset filter and a hangover, and records closed speech segments.

The same gate is used offline (``detect_segments``) and in streaming code (the caller
driver and the loopback agent), so both views of a call agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qsv.audio import DEFAULT_FRAME_MS, frame_samples, rms_db, to_int16


@dataclass(frozen=True)
class Segment:
    """A stretch of detected speech, in seconds from the start of the recording."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, other: Segment) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end


class FrameVAD(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None: ...


class EnergyVAD:
    """Frame energy against a dBFS threshold, with an optional adaptive noise floor.

    The effective threshold is ``max(threshold_db, noise_floor_db + margin_db)``. The noise
    floor is an exponential average of frames that were classified as non-speech, so a
    steadily noisy line raises the bar instead of reading as continuous speech.
    """

    def __init__(
        self,
        threshold_db: float = -45.0,
        *,
        adaptive: bool = True,
        margin_db: float = 15.0,
        initial_floor_db: float = -80.0,
        floor_alpha: float = 0.05,
    ) -> None:
        self.threshold_db = threshold_db
        self.adaptive = adaptive
        self.margin_db = margin_db
        self.initial_floor_db = initial_floor_db
        self.floor_alpha = floor_alpha
        self.noise_floor_db = initial_floor_db

    def reset(self) -> None:
        self.noise_floor_db = self.initial_floor_db

    @property
    def effective_threshold_db(self) -> float:
        if not self.adaptive:
            return self.threshold_db
        return max(self.threshold_db, self.noise_floor_db + self.margin_db)

    def is_speech(self, frame: np.ndarray) -> bool:
        level = rms_db(frame)
        speech = level > self.effective_threshold_db
        if self.adaptive and not speech:
            a = self.floor_alpha
            self.noise_floor_db = (1 - a) * self.noise_floor_db + a * level
        return speech


class WebRTCVAD:
    """Wrapper around the optional ``webrtcvad`` package (install the ``vad`` extra)."""

    def __init__(self, sample_rate: int, aggressiveness: int = 2) -> None:
        try:
            import webrtcvad
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "webrtcvad is not installed. Install it with `pip install voice-eval-toolkit[vad]`"
                " or use the default energy VAD."
            ) from exc
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("webrtcvad supports 8, 16, 32 and 48 kHz audio only")
        self.sample_rate = sample_rate
        self._vad = webrtcvad.Vad(aggressiveness)

    def reset(self) -> None:
        return None

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(self._vad.is_speech(to_int16(frame).tobytes(), self.sample_rate))


def make_vad(kind: str, sample_rate: int, **options: float) -> FrameVAD:
    """Build a frame classifier by name: ``energy`` (default) or ``webrtc``."""
    kind = kind.lower()
    if kind == "energy":
        return EnergyVAD(**options)  # type: ignore[arg-type]
    if kind in ("webrtc", "webrtcvad"):
        return WebRTCVAD(sample_rate, **options)  # type: ignore[arg-type]
    raise ValueError(f"unknown VAD '{kind}'. Choose 'energy' or 'webrtc'.")


class SpeechGate:
    """Stable speech state from raw frame decisions.

    ``active`` turns true once ``onset_ms`` of consecutive speech frames has been seen
    and turns false after ``hangover_ms`` of non-speech. Closed segments run from the
    first speech frame of the onset run to the end of the last speech frame, so the
    hangover never inflates a segment.
    """

    def __init__(
        self,
        vad: FrameVAD,
        sample_rate: int,
        *,
        frame_ms: int = DEFAULT_FRAME_MS,
        onset_ms: float = 40.0,
        hangover_ms: float = 200.0,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = frame_samples(sample_rate, frame_ms)
        self.onset_frames = max(1, math.ceil(onset_ms / frame_ms))
        self.hangover_frames = max(0, math.ceil(hangover_ms / frame_ms))
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.frame_index = 0
        self.active = False
        self.raw = False
        self.consecutive_speech = 0
        self._silence_run = 0
        self._last_speech_end = 0
        self._run_start = 0
        self._seg_start: int | None = None
        self.segments: list[Segment] = []

    @property
    def time_s(self) -> float:
        return self.frame_index * self.frame_ms / 1000.0

    def update(self, frame: np.ndarray) -> bool:
        """Feed one frame and return the new ``active`` state."""
        raw = self.vad.is_speech(frame)
        i = self.frame_index
        self.raw = raw
        if raw:
            if self.consecutive_speech == 0:
                self._run_start = i
            self.consecutive_speech += 1
            self._silence_run = 0
            self._last_speech_end = i + 1
            if not self.active and self.consecutive_speech >= self.onset_frames:
                self.active = True
                self._seg_start = self._run_start
        else:
            self.consecutive_speech = 0
            if self.active:
                self._silence_run += 1
                if self._silence_run >= self.hangover_frames:
                    self._close()
        self.frame_index += 1
        return self.active

    def _close(self) -> None:
        if self._seg_start is not None:
            ms = self.frame_ms / 1000.0
            self.segments.append(Segment(self._seg_start * ms, self._last_speech_end * ms))
        self._seg_start = None
        self.active = False
        self._silence_run = 0

    def finish(self) -> list[Segment]:
        """Close any open segment and return all segments."""
        if self.active:
            self._close()
        return list(self.segments)


def detect_segments(
    samples: np.ndarray,
    sample_rate: int,
    *,
    vad: FrameVAD | None = None,
    frame_ms: int = DEFAULT_FRAME_MS,
    onset_ms: float = 40.0,
    hangover_ms: float = 200.0,
) -> list[Segment]:
    """Run the gate over a whole recording and return its speech segments."""
    gate = SpeechGate(
        vad or EnergyVAD(),
        sample_rate,
        frame_ms=frame_ms,
        onset_ms=onset_ms,
        hangover_ms=hangover_ms,
    )
    n = gate.frame_len
    for start in range(0, len(samples) - n + 1, n):
        gate.update(samples[start : start + n])
    return gate.finish()
