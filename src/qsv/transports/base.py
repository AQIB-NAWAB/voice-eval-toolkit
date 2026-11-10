"""Transport interface.

A transport carries audio between the simulated caller and the agent under test. The
session driver is tick-synchronous: on every tick (one frame) it calls ``send_frame``
with the caller's audio and then ``recv_frame`` for whatever the agent produced during
that same tick. Real network transports run in real time; the loopback transport can run
faster than real time because it simulates time frame by frame.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qsv.metrics import TranscriptEvent


@dataclass
class CallContext:
    """Everything a transport may need to place one call."""

    call_id: str
    scenario_name: str
    sample_rate: int
    frame_ms: int
    seed: int = 0
    # Oracle information for mock agents only. Real transports must ignore these.
    reference_texts: list[str | None] = field(default_factory=list)
    current_turn: Callable[[], int | None] = lambda: None
    agent_options: dict = field(default_factory=dict)


class Transport(abc.ABC):
    """One call's audio path. Create a fresh instance per call."""

    name: str = "base"
    #: True when audio is exchanged over a real clock and the driver must pace itself.
    realtime: bool = True
    #: How far (ms) the live ``recv_frame`` view trails the true timeline. The caller
    #: compensates when it schedules interruptions. Zero for transports that are exact.
    live_lag_ms: float = 0.0

    def __init__(self) -> None:
        self.transcripts: list[TranscriptEvent] = []

    @abc.abstractmethod
    async def connect(self, ctx: CallContext) -> None:
        """Establish the call. Raise on failure; the load runner records it as a failed call."""

    @abc.abstractmethod
    async def send_frame(self, frame: np.ndarray) -> None:
        """Send one caller frame (mono float32 at ``ctx.sample_rate``)."""

    @abc.abstractmethod
    async def recv_frame(self) -> np.ndarray:
        """Return the agent audio for this tick, exactly one frame long (zeros if silent)."""

    def agent_recording(self) -> np.ndarray | None:
        """Authoritative agent audio for the whole call, if the transport can place audio on a
        timeline better than the tick-by-tick ``recv_frame`` view. ``None`` means "use the
        frames returned by ``recv_frame``". Called once, after the last tick."""
        return None

    @abc.abstractmethod
    async def close(self) -> None:
        """Hang up and release resources. Must be safe to call twice."""

    async def __aenter__(self) -> Transport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
