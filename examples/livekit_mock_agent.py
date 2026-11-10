"""A mock voice agent for trying the LiveKit transport without your own agent.

It watches a LiveKit server for rooms whose name starts with ``qsv-`` and joins each one.
For every caller track it feeds the audio through the same simulated agent that the loopback
transport uses (endpointing, response delay, barge-in) and publishes the reply as its own
audio track. That makes ``qsv bargein --transport livekit`` runnable end to end against a
local ``livekit-server --dev`` with nothing else installed.

    livekit-server --dev
    python examples/livekit_mock_agent.py --agent-opt stop_latency_ms=150 &
    export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
    qsv bargein --transport livekit

Needs the ``livekit`` extra. It is a development aid, not a model of a real agent.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os

import numpy as np
from livekit import api, rtc

from qsv.audio import frame_samples, from_int16, to_int16
from qsv.transports.base import CallContext
from qsv.transports.loopback import LoopbackAgent, LoopbackConfig

SAMPLE_RATE = 8000
FRAME_MS = 20


async def serve_room(url: str, key: str, secret: str, room_name: str, cfg: LoopbackConfig) -> None:
    token = (
        api.AccessToken(key, secret)
        .with_identity("mock-agent")
        .with_ttl(datetime.timedelta(hours=1))
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    done = asyncio.Event()
    tasks: list[asyncio.Task] = []

    ctx = CallContext(call_id=room_name, scenario_name="live", sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
    agent = LoopbackAgent(cfg, ctx)
    source = rtc.AudioSource(SAMPLE_RATE, 1)
    n = frame_samples(SAMPLE_RATE, FRAME_MS)

    async def pump(track: rtc.Track) -> None:
        buf = np.zeros(0, dtype=np.float32)
        async for ev in rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1):
            buf = np.concatenate([buf, from_int16(np.frombuffer(ev.frame.data, dtype=np.int16))])
            while len(buf) >= n:
                out = agent.process(buf[:n])
                buf = buf[n:]
                frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, n)
                np.frombuffer(frame.data, dtype=np.int16)[:] = to_int16(out)
                await source.capture_frame(frame)

    def _report(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            print(f"{room_name}: audio pump failed: {task.exception()!r}", flush=True)

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):  # noqa: ARG001
        if track.kind == rtc.TrackKind.KIND_AUDIO and not tasks:
            tasks.append(asyncio.ensure_future(pump(track)))
            tasks[-1].add_done_callback(_report)

    @room.on("disconnected")
    def _on_disc(*_):
        done.set()

    await room.connect(url, token)
    track = rtc.LocalAudioTrack.create_audio_track("agent", source)
    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, opts)
    print(f"joined {room_name}", flush=True)
    # Leave when the caller does.
    while not done.is_set() and len(room.remote_participants) + (0 if tasks else 1) > 0:
        await asyncio.sleep(0.5)
        if tasks and not room.remote_participants:
            break
    for t in tasks:
        t.cancel()
    await room.disconnect()
    print(f"left {room_name}: {agent.tick} frames heard, {agent._responses_started} replies", flush=True)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"))
    p.add_argument("--key", default=os.environ.get("LIVEKIT_API_KEY", "devkey"))
    p.add_argument("--secret", default=os.environ.get("LIVEKIT_API_SECRET", "secret"))
    p.add_argument("--prefix", default="qsv-")
    p.add_argument("--agent-opt", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = LoopbackConfig.from_options(dict(i.split("=", 1) for i in args.agent_opt))

    http = args.url.replace("ws://", "http://").replace("wss://", "https://")
    lk = api.LiveKitAPI(http, args.key, args.secret)
    joined: set[str] = set()
    running: list[asyncio.Task] = []
    print(f"watching {http} for rooms starting with '{args.prefix}'")
    try:
        while True:
            rooms = await lk.room.list_rooms(api.ListRoomsRequest())
            for r in rooms.rooms:
                if r.name.startswith(args.prefix) and r.name not in joined:
                    joined.add(r.name)
                    running.append(
                        asyncio.ensure_future(serve_room(args.url, args.key, args.secret, r.name, cfg))
                    )
            await asyncio.sleep(0.3)
    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
