from __future__ import annotations

import builtins

import numpy as np
import pytest

from conftest import run
from qsv.transports import CallContext, get_transport_factory
from qsv.transports.livekit_transport import (
    LIVE_LAG_FRAMES,
    LiveKitConfig,
    LiveKitTransport,
    build_token,
)
from qsv.transports.sip import SipConfig, SipTransport


def ctx():
    return CallContext(call_id="c1", scenario_name="s", sample_rate=8000, frame_ms=20)


def test_sip_stub_raises_with_guidance():
    t = SipTransport(SipConfig(uri="sip:agent@example.com"))
    with pytest.raises(NotImplementedError, match="LiveKit"):
        run(t.connect(ctx()))
    with pytest.raises(NotImplementedError):
        run(t.send_frame(np.zeros(160, np.float32)))
    run(t.close())  # safe


def test_factory_passes_options_through():
    t = get_transport_factory("sip", {"uri": "sip:x@y", "codec": "PCMA"})()
    assert isinstance(t, SipTransport) and t.config.codec == "PCMA"
    lk = get_transport_factory("livekit", {"room_prefix": "bench", "url": None})()
    assert isinstance(lk, LiveKitTransport)


def test_livekit_config_reads_environment(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 32)
    cfg = LiveKitConfig(room_prefix="r").resolved()
    assert cfg.url == "wss://example.livekit.cloud" and cfg.api_key == "key" and cfg.room_prefix == "r"


def test_livekit_config_lists_missing_settings(monkeypatch):
    for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET"):
        LiveKitConfig().resolved()


def test_build_token_contains_room_grant_and_dispatch():
    pytest.importorskip("livekit.api")
    import jwt

    cfg = LiveKitConfig(url="wss://x", api_key="devkey", api_secret="x" * 40, agent_name="support-agent")
    token = build_token(cfg, room="qsv-call-1", identity="caller-1")
    claims = jwt.decode(token, "x" * 40, algorithms=["HS256"], options={"verify_aud": False})
    assert claims["sub"] == "caller-1" and claims["iss"] == "devkey"
    assert claims["video"]["room"] == "qsv-call-1" and claims["video"]["roomJoin"] is True
    assert claims["roomConfig"]["agents"][0]["agentName"] == "support-agent"

    plain = build_token(LiveKitConfig(api_key="devkey", api_secret="x" * 40), room="r", identity="i")
    assert "roomConfig" not in jwt.decode(
        plain, "x" * 40, algorithms=["HS256"], options={"verify_aud": False}
    )


def test_livekit_sdk_missing_gives_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "livekit" or name.startswith("livekit."):
            raise ImportError("no livekit")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    t = LiveKitTransport(LiveKitConfig(url="wss://x", api_key="k", api_secret="s"))
    with pytest.raises(ImportError, match=r"voice-eval-toolkit\[livekit\]"):
        run(t.connect(ctx()))


def test_livekit_timeline_places_audio_by_arrival_time():
    t = LiveKitTransport()
    t._frame_len = 160
    t._place(np.ones(160, np.float32), 320)  # arrived two frames in
    rec = t.agent_recording()
    assert len(rec) == 480 and not rec[:320].any() and rec[320:].min() == 1.0


def test_livekit_live_view_trails_the_timeline_by_the_lag():
    t = LiveKitTransport()
    t._frame_len = 160
    t._place(np.ones(160, np.float32), 320)  # slot 2
    frames = [run(t.recv_frame()) for _ in range(LIVE_LAG_FRAMES + 3)]
    nonzero = [i for i, f in enumerate(frames) if f.any()]
    assert nonzero == [2 + LIVE_LAG_FRAMES]


def test_livekit_frames_never_overlap_on_the_timeline():
    t = LiveKitTransport()
    t._frame_len = 160
    t._place(np.ones(160, np.float32), 0)
    t._place(np.full(160, 0.5, np.float32), 0)  # arrived in a burst: queued after the first
    rec = t.agent_recording()
    assert len(rec) == 320 and rec[0] == 1.0 and rec[160] == 0.5


def test_livekit_live_view_pads_a_partial_slot():
    t = LiveKitTransport()
    t._frame_len = 160
    t._place(np.ones(80, np.float32), 0)
    frames = [run(t.recv_frame()) for _ in range(LIVE_LAG_FRAMES + 1)]
    last = frames[-1]
    assert last[:80].min() == 1.0 and not last[80:].any()
