"""Optional end-to-end check against a real LiveKit server.

Skipped unless QSV_LIVEKIT_INTEGRATION=1. Start a dev server first:

    livekit-server --dev            # keys: devkey / secret
    QSV_LIVEKIT_INTEGRATION=1 pytest tests/test_livekit_integration.py

The test starts examples/livekit_mock_agent.py as a subprocess and places two real WebRTC calls.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from conftest import ROOT, run

pytestmark = pytest.mark.skipif(
    os.environ.get("QSV_LIVEKIT_INTEGRATION") != "1",
    reason="set QSV_LIVEKIT_INTEGRATION=1 with a local livekit-server",
)


def test_two_calls_over_a_local_livekit_server():
    pytest.importorskip("livekit.rtc")
    from qsv.load import LoadConfig, run_load
    from qsv.scenario import builtin_bargein_scenario
    from qsv.transports.livekit_transport import LiveKitConfig, LiveKitTransport

    url = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
    key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    agent = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "examples" / "livekit_mock_agent.py"),
            "--url",
            url,
            "--key",
            key,
            "--secret",
            secret,
            "--agent-opt",
            "response_ms=4500",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)
        cfg = LiveKitConfig(url=url, api_key=key, api_secret=secret, agent_join_timeout_s=15)
        res = run(
            run_load(
                builtin_bargein_scenario([600]),
                lambda: LiveKitTransport(cfg),
                LoadConfig(callers=2),
                run_id="it",
            )
        )
    finally:
        agent.terminate()
        agent.wait(timeout=10)
    assert all(r.ok for r in res.results), [r.error for r in res.results]
    for r in res.results:
        assert len(r.metrics.stop_latencies_ms) == 1
        assert 100 < r.metrics.stop_latencies_ms[0] < 1000
