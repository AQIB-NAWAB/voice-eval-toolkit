from __future__ import annotations

import numpy as np
import pytest
import yaml

from conftest import SCENARIOS
from qsv.audio import rms_db, synth_speech, write_wav
from qsv.scenario import (
    ScenarioError,
    builtin_bargein_scenario,
    load_scenario,
    parse_scenario,
)

MINIMAL = {"name": "t", "turns": [{"say": "hello there friend"}]}


def test_all_example_scenarios_load():
    files = sorted(SCENARIOS.glob("*.yaml"))
    assert len(files) >= 4
    for f in files:
        sc = load_scenario(f)
        assert sc.turns and sc.name


def test_parse_defaults_and_duration_estimate():
    sc = parse_scenario(MINIMAL)
    assert sc.sample_rate == 8000
    t = sc.turns[0]
    assert t.pause_ms == 400 and t.interrupt_after_ms is None
    assert t.speech_duration_ms() == 1200  # 3 words at 150 wpm
    assert t.reference_text == "hello there friend"


def test_full_turn_fields():
    sc = parse_scenario(
        {
            "name": "x",
            "wait_for_greeting_ms": 1500,
            "assertions": {"max_wer": 0.2, "max_false_interruptions": 0},
            "agent": {"stop_latency_ms": 90},
            "turns": [
                {
                    "say": "hi",
                    "duration_ms": 900,
                    "expected_transcript": "hi",
                    "interrupt_after_ms": 400,
                    "pause_ms": 100,
                    "hesitation": {"at_ms": 300, "duration_ms": 500},
                    "level_db": -25,
                }
            ],
        }
    )
    t = sc.turns[0]
    assert t.interrupt_after_ms == 400 and t.hesitation.duration_ms == 500 and t.level_db == -25
    assert sc.assertions.as_dict() == {"max_wer": 0.2, "max_false_interruptions": 0}
    assert sc.agent == {"stop_latency_ms": 90}
    assert sc.wait_for_greeting_ms == 1500


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "mapping"),
        ({"turns": [{"say": "x"}]}, "name"),
        ({"name": "a"}, "turns"),
        ({"name": "a", "turns": []}, "turns"),
        ({"name": "a", "turns": [{"say": "x"}], "bogus": 1}, "unknown scenario field"),
        ({"name": "a", "turns": [{"say": "x", "wat": 1}]}, "unknown field"),
        ({"name": "a", "turns": [{"pause_ms": 5}]}, "at least one of"),
        ({"name": "a", "turns": [{"say": "x", "pause_ms": -1}]}, ">= 0"),
        ({"name": "a", "turns": [{"say": "x", "interrupt_after_ms": "soon"}]}, "must be a number"),
        ({"name": "a", "turns": [{"say": "x", "hesitation": {"at_ms": 1}}]}, "hesitation"),
        ({"name": "a", "turns": [{"say": "x"}], "assertions": {"max_vibes": 1}}, "unknown assertion"),
        ({"name": "a", "turns": [{"say": "x"}], "sample_rate": 100}, ">= 8000"),
    ],
)
def test_invalid_scenarios_raise_clear_errors(data, message):
    with pytest.raises(ScenarioError, match=message):
        parse_scenario(data)


def test_load_scenario_reports_path_and_yaml_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ScenarioError, match="bad.yaml"):
        load_scenario(bad)
    with pytest.raises(ScenarioError, match="cannot read"):
        load_scenario(tmp_path / "missing.yaml")
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(yaml.safe_dump({"name": "z"}), encoding="utf-8")
    with pytest.raises(ScenarioError, match="wrong.yaml"):
        load_scenario(wrong)


def test_turn_audio_synthetic_duration_and_level():
    sc = parse_scenario({"name": "a", "turns": [{"say": "x", "duration_ms": 1000, "level_db": -24}]})
    x = sc.build_turn_audio(0, seed=1)
    assert len(x) == 8000
    assert rms_db(x) == pytest.approx(-24, abs=0.5)


def test_hesitation_inserts_silence_in_the_middle():
    sc = parse_scenario(
        {
            "name": "a",
            "turns": [{"say": "x", "duration_ms": 2000, "hesitation": {"at_ms": 1000, "duration_ms": 500}}],
        }
    )
    x = sc.build_turn_audio(0)
    assert len(x) == 2500 * 8
    assert not x[8000:12000].any()
    assert x[:8000].any() and x[12000:].any()


def test_audio_file_turn_is_resampled_and_relative_paths_resolve(tmp_path):
    wav = tmp_path / "caller.wav"
    write_wav(wav, synth_speech(500, 16000, seed=1), 16000)
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"name": "f", "turns": [{"audio": "caller.wav"}]}), encoding="utf-8")
    sc = load_scenario(path)
    x = sc.build_turn_audio(0)
    assert len(x) == 4000
    assert np.isfinite(x).all() and x.any()


def test_builtin_bargein_scenario_shape():
    sc = builtin_bargein_scenario([300, 700, 1100])
    assert len(sc.turns) == 4
    assert [t.interrupt_after_ms for t in sc.turns] == [None, 300, 700, 1100]
