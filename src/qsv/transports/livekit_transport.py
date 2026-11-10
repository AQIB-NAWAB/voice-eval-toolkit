"""LiveKit transport.

Joins a LiveKit room as a caller participant, publishes the scripted caller audio as a
microphone track and records the first remote audio track (the agent). Requires the
``livekit`` extra: ``pip install voice-eval-toolkit[livekit]``. The SDK is imported lazily,
so the rest of qsv works without it.

How it maps onto the tick-synchronous Transport contract
--------------------------------------------------------
* ``send_frame`` copies the frame into an ``rtc.AudioFrame`` and calls
  ``AudioSource.capture_frame``.
* A background task reads ``rtc.AudioStream`` for the agent track. Each incoming frame is
  placed on an agent timeline at its wall-clock arrival time (never earlier than the end of
  the previous frame), so network jitter shows up as timing in the recording instead of
  silently stretching it.
* Audio for a tick only exists once it has arrived, so the live ``recv_frame`` view trails the
  timeline by ``LIVE_LAG_FRAMES`` frames (60 ms), which is what the simulated caller reacts to.
  Metrics never use that lagged view: ``agent_recording`` returns the full arrival-time
  timeline and the session analyses that.
* If the agent worker publishes user transcriptions on the ``lk.transcription`` text
  stream (LiveKit Agents does this by default), the transcripts of the *caller* track are
  collected for WER. Agents that do not publish them simply yield no WER.

Dispatch
--------
By default the token has no agent dispatch, which works with LiveKit Agents workers that use
automatic dispatch (they join every new room). For explicit dispatch pass ``agent_name``; the
token then carries a ``RoomAgentDispatch`` so the named agent is started when we join.

Configuration comes from constructor arguments or the environment: ``LIVEKIT_URL``,
``LIVEKIT_API_KEY`` and ``LIVEKIT_API_SECRET``. One room is created per call.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from qsv.audio import frame_samples, from_int16, to_int16
from qsv.metrics import TranscriptEvent
from qsv.transports.base import CallContext, Transport

LIVE_LAG_FRAMES = 3
TRANSCRIPTION_TOPIC = "lk.transcription"
_TRANSCRIBED_TRACK_ATTR = "lk.transcribed_track_id"
_FINAL_ATTR = "lk.transcription_final"


@dataclass
class LiveKitConfig:
    url: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    room_prefix: str = "qsv"
    agent_name: str | None = None
    identity_prefix: str = "qsv-caller"
    agent_join_timeout_s: float = 20.0
    token_ttl_s: int = 600

    def resolved(self) -> LiveKitConfig:
        url = self.url or os.environ.get("LIVEKIT_URL")
        key = self.api_key or os.environ.get("LIVEKIT_API_KEY")
        secret = self.api_secret or os.environ.get("LIVEKIT_API_SECRET")
        missing = [
            name
            for name, value in (
                ("LIVEKIT_URL", url),
                ("LIVEKIT_API_KEY", key),
                ("LIVEKIT_API_SECRET", secret),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "LiveKit transport needs " + ", ".join(missing) + " (environment variables or CLI flags)."
            )
        return LiveKitConfig(
            url=url,
            api_key=key,
            api_secret=secret,
            room_prefix=self.room_prefix,
            agent_name=self.agent_name,
            identity_prefix=self.identity_prefix,
            agent_join_timeout_s=self.agent_join_timeout_s,
            token_ttl_s=self.token_ttl_s,
        )


def _require_sdk():
    try:
        from livekit import rtc
    except ImportError as exc:
        raise ImportError(
            "The LiveKit transport needs the livekit SDK. "
            "Install it with `pip install voice-eval-toolkit[livekit]`."
        ) from exc
    return rtc


def build_token(cfg: LiveKitConfig, *, room: str, identity: str) -> str:
    """Create a join token for ``room`` (with optional explicit agent dispatch)."""
    try:
        from livekit import api
    except ImportError as exc:
        raise ImportError(
            "Creating LiveKit tokens needs livekit-api. "
            "Install it with `pip install voice-eval-toolkit[livekit]`."
        ) from exc
    import datetime

    token = (
        api.AccessToken(cfg.api_key, cfg.api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(datetime.timedelta(seconds=cfg.token_ttl_s))
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
    )
    if cfg.agent_name:
        token = token.with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=cfg.agent_name)])
        )
    return token.to_jwt()


class LiveKitTransport(Transport):
    name = "livekit"
    realtime = True

    def __init__(self, config: LiveKitConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or LiveKitConfig()
        self._rtc = None
        self._room = None
        self._source = None
        self._ctx: CallContext | None = None
        self._frame_len = 0
        self._t0 = 0.0
        self._timeline = np.zeros(0, dtype=np.float32)
        self._cursor = 0  # next free sample on the agent timeline
        self._tick = 0  # recv_frame calls so far
        self._agent_track_ready = asyncio.Event()
        self._reader_task: asyncio.Task | None = None
        self._pending: deque[asyncio.Task] = deque()
        self._local_track_sid: str | None = None
        self.room_name = ""

    # ------------------------------------------------------------------ connect

    async def connect(self, ctx: CallContext) -> None:
        cfg = self._cfg.resolved()
        rtc = _require_sdk()
        self._rtc = rtc
        self._ctx = ctx
        self._frame_len = frame_samples(ctx.sample_rate, ctx.frame_ms)
        self.live_lag_ms = LIVE_LAG_FRAMES * ctx.frame_ms
        self.room_name = f"{cfg.room_prefix}-{ctx.call_id}"
        identity = f"{cfg.identity_prefix}-{ctx.call_id}"
        token = build_token(cfg, room=self.room_name, identity=identity)

        room = rtc.Room()
        self._room = room

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):  # noqa: ARG001
            if track.kind == rtc.TrackKind.KIND_AUDIO and self._reader_task is None:
                self._reader_task = asyncio.ensure_future(self._read_agent_audio(track))
                self._agent_track_ready.set()

        room.register_text_stream_handler(TRANSCRIPTION_TOPIC, self._on_text_stream)

        await room.connect(cfg.url, token, options=rtc.RoomOptions(auto_subscribe=True))

        self._source = rtc.AudioSource(ctx.sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track("qsv-caller", self._source)
        publish_options = rtc.TrackPublishOptions()
        publish_options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await room.local_participant.publish_track(track, publish_options)
        self._local_track_sid = publication.sid

        # The agent may already be in the room with its track published.
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None and pub.kind == rtc.TrackKind.KIND_AUDIO:
                    _on_track(pub.track, pub, participant)

        try:
            await asyncio.wait_for(self._agent_track_ready.wait(), cfg.agent_join_timeout_s)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise TimeoutError(
                f"no agent audio track appeared in room '{self.room_name}' within "
                f"{cfg.agent_join_timeout_s:.0f}s. Is an agent worker running"
                + (f" with agent_name '{cfg.agent_name}'?" if cfg.agent_name else "?")
            ) from exc
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ audio in/out

    async def send_frame(self, frame: np.ndarray) -> None:
        rtc = self._rtc
        if rtc is None or self._source is None or self._ctx is None:
            raise RuntimeError("transport is not connected")
        audio_frame = rtc.AudioFrame.create(self._ctx.sample_rate, 1, len(frame))
        np.frombuffer(audio_frame.data, dtype=np.int16)[:] = to_int16(frame)
        await self._source.capture_frame(audio_frame)

    async def recv_frame(self) -> np.ndarray:
        # Yield so the reader task can deliver frames that arrived during this tick.
        await asyncio.sleep(0)
        n = self._frame_len
        start = (self._tick - LIVE_LAG_FRAMES) * n
        self._tick += 1
        out = np.zeros(n, dtype=np.float32)
        if start < 0 or self._timeline.size <= start:
            return out
        chunk = self._timeline[start : start + n]
        out[: len(chunk)] = chunk
        return out

    def agent_recording(self) -> np.ndarray:
        return self._timeline[: self._cursor].copy()

    async def _read_agent_audio(self, track) -> None:
        rtc = self._rtc
        assert rtc is not None and self._ctx is not None
        sr = self._ctx.sample_rate
        stream = rtc.AudioStream(track, sample_rate=sr, num_channels=1)
        try:
            async for event in stream:
                data = from_int16(np.frombuffer(event.frame.data, dtype=np.int16))
                if self._t0 == 0.0:
                    continue  # still connecting; drop audio from before the call clock started
                arrival = int(round((time.perf_counter() - self._t0) * sr)) - len(data)
                self._place(data, arrival)
        except asyncio.CancelledError:
            raise
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    def _place(self, data: np.ndarray, at: int) -> None:
        at = max(at, self._cursor)  # frames never overlap; a burst queues behind the previous one
        end = at + len(data)
        if self._timeline.size < end:
            grown = np.zeros(max(end, int(self._timeline.size * 1.5) + self._frame_len), dtype=np.float32)
            grown[: self._timeline.size] = self._timeline
            self._timeline = grown
        self._timeline[at:end] = data
        self._cursor = end

    # ------------------------------------------------------------------ transcripts

    def _on_text_stream(self, reader, participant_identity: str) -> None:
        self._pending.append(asyncio.ensure_future(self._collect_text(reader, participant_identity)))

    async def _collect_text(self, reader, participant_identity: str) -> None:
        try:
            text = await reader.read_all()
        except Exception:
            return
        attributes = getattr(reader.info, "attributes", None) or {}
        track_id = attributes.get(_TRANSCRIBED_TRACK_ATTR)
        # Keep only transcripts of our own published track, and only final ones when the
        # sender marks interim results.
        if not track_id or track_id != self._local_track_sid:
            return
        if attributes.get(_FINAL_ATTR, "true") != "true":
            return
        if text.strip():
            self.transcripts.append(TranscriptEvent(time_s=time.perf_counter() - self._t0, text=text.strip()))

    # ------------------------------------------------------------------ teardown

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        for task in list(self._pending):
            if not task.done():
                task.cancel()
        self._pending.clear()
        if self._source is not None:
            with contextlib.suppress(Exception):
                await self._source.aclose()
            self._source = None
        if self._room is not None:
            with contextlib.suppress(Exception):
                await self._room.disconnect()
            self._room = None
