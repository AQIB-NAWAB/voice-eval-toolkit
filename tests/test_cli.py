from __future__ import annotations

import json

import pytest

from conftest import SCENARIOS
from qsv.cli import main


def test_bargein_default_script_writes_reports(tmp_path, capsys):
    code = main(["bargein", "--transport", "loopback", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "barge-in stop" in out
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["run"]["mode"] == "bargein"
    assert data["summary"]["bargein"]["interruptions_yielded"] == 2
    assert (tmp_path / "report.md").exists()


def test_bargein_custom_offsets_agent_opts_and_audio(tmp_path):
    code = main(
        [
            "bargein",
            "--interrupt-at",
            "400,800,1200",
            "--agent-opt",
            "stop_latency_ms=200",
            "--agent-opt",
            "response_ms=5000",
            "--save-audio",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["summary"]["bargein"]["interrupt_turns"] == 3
    p95 = data["summary"]["bargein"]["stop_latency_ms"]["p95"]
    assert 250 < p95 < 450
    assert list(tmp_path.glob("*.wav"))


def test_bargein_limit_failure_gives_exit_code_1(tmp_path):
    code = main(
        [
            "bargein",
            "--agent-opt",
            "barge_in=false",
            "--agent-opt",
            "response_ms=4500",
            "--max-stop-p95-ms",
            "500",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 1
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["assertions"][0]["passed"] is False


def test_load_basic_five_callers(tmp_path, capsys):
    code = main(
        [
            "load",
            "--transport",
            "loopback",
            "--scenario",
            str(SCENARIOS / "basic.yaml"),
            "--callers",
            "5",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "calls: 5/5 ok" in capsys.readouterr().out
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["summary"]["calls_ok"] == 5
    assert data["run"]["callers"] == 5


def test_load_requires_scenario():
    with pytest.raises(SystemExit) as exc:
        main(["load", "--transport", "loopback"])
    assert exc.value.code == 2


def test_validate_command(capsys, tmp_path):
    assert main(["validate", str(SCENARIOS / "basic.yaml")]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1
    assert "bad.yaml" in capsys.readouterr().err


def test_bad_scenario_and_bad_options_return_usage_error(capsys, tmp_path):
    assert main(["load", "--scenario", str(tmp_path / "nope.yaml"), "--out", str(tmp_path)]) == 2
    assert main(["bargein", "--agent-opt", "oops", "--out", str(tmp_path)]) == 2
    assert main(["bargein", "--interrupt-at", "a,b", "--out", str(tmp_path)]) == 2
    assert main(["bargein", "--agent-opt", "unknown_thing=1", "--out", str(tmp_path / "z")]) in (1, 2)
    capsys.readouterr()


def test_sip_transport_explains_it_is_a_stub(capsys, tmp_path):
    code = main(["bargein", "--transport", "sip", "--out", str(tmp_path)])
    assert code == 2
    assert "not implemented" in capsys.readouterr().err


def test_fast_flag_rejected_for_real_transports(capsys, tmp_path):
    assert main(["bargein", "--transport", "livekit", "--fast", "--out", str(tmp_path)]) == 2


def test_livekit_without_credentials_fails_cleanly(monkeypatch, capsys, tmp_path):
    pytest.importorskip("livekit.api")
    for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    code = main(["bargein", "--transport", "livekit", "--out", str(tmp_path), "--quiet"])
    assert code == 1  # the call fails to connect and is reported, not raised
    data = json.loads((tmp_path / "report.json").read_text())
    assert "LIVEKIT_URL" in data["summary"]["errors"][0]


def test_python_dash_m_entrypoint(tmp_path):
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "qsv", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and "qsv" in r.stdout
