"""In-process mock voice agent.

``LoopbackTransport`` needs no network. It hosts a small simulated agent that behaves like
a cascaded STT, LLM, TTS pipeline from the outside:

* it listens with its own VAD and treats ``endpointing_ms`` of silence as end of turn,
* it waits ``response_delay_ms`` (plus seeded jitter) and then "speaks" synthetic audio,
* if the caller speaks while it is talking it stops after ``barge_in_min_speech_ms`` of
  caller speech plus ``stop_latency_ms`` of audio tail (or never, with ``barge_in: false``),
* it can emit simulated STT transcripts with a configurable word error rate.

It exists so the harness, the metrics and the load runner can be developed and tested
anywhere, and to give the analyser a controllable ground truth. Numbers from it describe
the mock, not a real agent. Time is simulated frame by frame, so runs finish much faster
than real time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from qsv.audio import frame_samples, silence, synth_speech
from qsv.metrics import TranscriptEvent
from qsv.transports.base import CallContext, Transport
from qsv.vad import EnergyVAD, SpeechGate

_SUBSTITUTES = ["the", "a", "uh", "and", "to", "that", "it", "is"]


@dataclass
class LoopbackConfig:
    endpointing_ms: float = 500.0
    response_delay_ms: float = 300.0
    jitter_ms: float = 0.0
    response_ms: list[float] = field(default_factory=lambda: [2500.0, 2000.0, 3000.0])
    greeting_ms: float = 0.0
    greeting_delay_ms: float = 200.0
    barge_in: bool = True
    barge_in_min_speech_ms: float = 120.0
    stop_latency_ms: float = 80.0
    stop_jitter_ms: float = 0.0
    stt_word_error_rate: float = 0.0
    vad_threshold_db: float = -45.0
    level_db: float = -22.0

    @classmethod
    def from_options(cls, options: dict[str, Any] | None) -> LoopbackConfig:
        """Build a config from loose options (YAML values or ``key=value`` strings)."""
        options = options or {}
        known = {f.name: f for f in fields(cls)}
        unknown = set(options) - set(known)
        if unknown:
            raise ValueError(f"unknown loopback agent option(s) {sorted(unknown)}; known: {sorted(known)}")
        cfg = cls()
        for key, raw in options.items():
            current = getattr(cfg, key)
            if isinstance(current, bool):
                value: Any = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
            elif isinstance(current, list):
                if isinstance(raw, str):
                    value = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
                elif isinstance(raw, (int, float)):
                    value = [float(raw)]
                else:
                    value = [float(x) for x in raw]
                if not value:
                    raise ValueError(f"'{key}' needs at least one value")
            else:
                value = float(raw)
            setattr(cfg, key, value)
        return cfg


def degrade_text(text: str, word_error_rate: float, rng: np.random.Generator) -> str:
    """Simulate STT errors: each word is dropped or replaced with probability ``word_error_rate``."""
    if word_error_rate <= 0:
        return text
    out: list[str] = []
    for word in text.split():
        if rng.random() < word_error_rate:
            if rng.random() < 0.5:
                continue
            out.append(_SUBSTITUTES[int(rng.integers(len(_SUBSTITUTES)))])
        else:
            out.append(word)
    return " ".join(out)


class LoopbackAgent:
    """The simulated agent, advanced one frame at a time."""

    def __init__(self, cfg: LoopbackConfig, ctx: CallContext) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.sr = ctx.sample_rate
        self.frame_ms = ctx.frame_ms
        self.frame_len = frame_samples(ctx.sample_rate, ctx.frame_ms)
        self.rng = np.random.default_rng(ctx.seed + 104729)
        self.gate = SpeechGate(
            EnergyVAD(cfg.vad_threshold_db),
            ctx.sample_rate,
            frame_ms=ctx.frame_ms,
            onset_ms=40,
            hangover_ms=cfg.endpointing_ms,
        )
        self.tick = 0
        self.transcripts: list[TranscriptEvent] = []
        self._pending_at: int | None = None
        self._out: np.ndarray | None = None
        self._out_pos = 0
        self._stop_at: int | None = None
        self._responses_started = 0
        self._utt_turn: int | None = None
        self._transcribed: set[int | None] = set()
        if cfg.greeting_ms > 0:
            self._pending_at = self._ticks(cfg.greeting_delay_ms)

    def _ticks(self, ms: float) -> int:
        return int(math.ceil(ms / self.frame_ms))

    @property
    def speaking(self) -> bool:
        return self._out is not None

    def _start_response(self) -> None:
        cfg = self.cfg
        if self._responses_started == 0 and cfg.greeting_ms > 0:
            duration = cfg.greeting_ms
        else:
            index = self._responses_started - (1 if cfg.greeting_ms > 0 else 0)
            duration = cfg.response_ms[max(0, index) % len(cfg.response_ms)]
        self._out = synth_speech(
            duration, self.sr, seed=self.ctx.seed * 31 + self._responses_started, level_db=cfg.level_db
        )
        self._out_pos = 0
        self._stop_at = None
        self._responses_started += 1

    def _transcribe(self) -> None:
        turn = self._utt_turn
        text = ""
        if turn is not None and turn not in self._transcribed and turn < len(self.ctx.reference_texts):
            ref = self.ctx.reference_texts[turn]
            if ref:
                text = degrade_text(ref, self.cfg.stt_word_error_rate, self.rng)
        self._transcribed.add(turn)
        if text:
            self.transcripts.append(TranscriptEvent(time_s=self.tick * self.frame_ms / 1000.0, text=text))

    def process(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        was_active = self.gate.active
        active = self.gate.update(frame)

        if active and not was_active:
            self._utt_turn = self.ctx.current_turn()
            self._pending_at = None  # caller started again: drop any reply that has not begun
        if was_active and not active:
            self._transcribe()
            jitter = float(self.rng.uniform(-cfg.jitter_ms, cfg.jitter_ms)) if cfg.jitter_ms else 0.0
            self._pending_at = self.tick + self._ticks(max(0.0, cfg.response_delay_ms + jitter))

        if (
            cfg.barge_in
            and self.speaking
            and self._stop_at is None
            and self.gate.consecutive_speech * self.frame_ms >= cfg.barge_in_min_speech_ms
        ):
            stop_jitter = (
                float(self.rng.uniform(-cfg.stop_jitter_ms, cfg.stop_jitter_ms))
                if cfg.stop_jitter_ms
                else 0.0
            )
            self._stop_at = self.tick + self._ticks(max(0.0, cfg.stop_latency_ms + stop_jitter))

        if self._pending_at is not None and self.tick >= self._pending_at and not self.speaking:
            self._pending_at = None
            self._start_response()

        out = self._next_output()
        self.tick += 1
        return out

    def _next_output(self) -> np.ndarray:
        if self._out is None:
            return silence(self.frame_ms, self.sr)
        if self._stop_at is not None and self.tick >= self._stop_at:
            self._out = None
            return silence(self.frame_ms, self.sr)
        chunk = self._out[self._out_pos : self._out_pos + self.frame_len]
        self._out_pos += self.frame_len
        if self._out_pos >= len(self._out):
            self._out = None
        if len(chunk) < self.frame_len:
            chunk = np.concatenate([chunk, np.zeros(self.frame_len - len(chunk), dtype=np.float32)])
        return chunk


class LoopbackTransport(Transport):
    name = "loopback"
    realtime = False

    def __init__(self, config: LoopbackConfig | None = None) -> None:
        super().__init__()
        self._base_config = config
        self._agent: LoopbackAgent | None = None
        self._out: np.ndarray | None = None

    async def connect(self, ctx: CallContext) -> None:
        if self._base_config is not None:
            cfg = self._base_config
        else:
            cfg = LoopbackConfig.from_options(ctx.agent_options)
        self._agent = LoopbackAgent(cfg, ctx)

    async def send_frame(self, frame: np.ndarray) -> None:
        if self._agent is None:
            raise RuntimeError("transport is not connected")
        self._out = self._agent.process(frame)
        self.transcripts = self._agent.transcripts

    async def recv_frame(self) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("recv_frame called before send_frame")
        return self._out

    async def close(self) -> None:
        self._agent = None
