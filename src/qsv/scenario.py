"""YAML scenario format and caller audio construction.

A scenario is a script for a simulated caller::

    name: refund-interrupt
    sample_rate: 8000
    wait_for_greeting_ms: 2000
    turns:
      - say: "Hi, I need a refund for order 1234"
        expected_transcript: "hi i need a refund for order 1234"
      - say: "Sorry, actually it was order 1243"
        interrupt_after_ms: 900      # start talking 900 ms into the agent's reply
        pause_ms: 300

Turn fields
-----------
say                  Text the caller speaks. Used for timing and as the WER reference when
                     ``expected_transcript`` is absent.
audio                Path to a WAV file to play instead of synthetic speech (resampled).
duration_ms          Speaking time for synthetic speech. Estimated from ``say`` if omitted.
expected_transcript  WER reference. Falls back to ``say``.
interrupt_after_ms   Barge in: start this turn that many ms after the agent's current or
                     latest reply begins. Without it the caller waits politely.
pause_ms             Silence before the turn starts once the agent has finished (polite
                     turns only). Default 400.
hesitation           ``{at_ms, duration_ms}``: go quiet in the middle of the turn, like a
                     caller saying "um". Agents with aggressive endpointing jump in.
level_db             Speech level in dBFS RMS. Default -20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qsv.audio import (
    DEFAULT_SAMPLE_RATE,
    estimate_speech_duration_ms,
    read_wav,
    resample,
    silence,
    synth_speech,
)


class ScenarioError(ValueError):
    """Raised when a scenario file is malformed."""


@dataclass(frozen=True)
class Hesitation:
    at_ms: int
    duration_ms: int


@dataclass(frozen=True)
class Turn:
    say: str | None = None
    audio: Path | None = None
    duration_ms: int | None = None
    expected_transcript: str | None = None
    interrupt_after_ms: int | None = None
    pause_ms: int = 400
    hesitation: Hesitation | None = None
    level_db: float = -20.0

    @property
    def reference_text(self) -> str | None:
        return self.expected_transcript if self.expected_transcript is not None else self.say

    def speech_duration_ms(self) -> int:
        if self.duration_ms is not None:
            return self.duration_ms
        if self.say:
            return estimate_speech_duration_ms(self.say)
        return 2000


@dataclass(frozen=True)
class Assertions:
    """Optional pass/fail limits checked against the run summary."""

    max_bargein_stop_p95_ms: float | None = None
    max_turn_gap_p95_ms: float | None = None
    max_false_interruptions: int | None = None
    max_wer: float | None = None
    min_success_rate: float | None = None

    def as_dict(self) -> dict[str, float | int]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: list[Turn]
    description: str = ""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    wait_for_greeting_ms: int = 0
    initial_silence_ms: int = 500
    response_timeout_ms: int = 6000
    end_silence_ms: int = 700
    tail_ms: int = 800
    max_call_s: int = 120
    noise_db: float = -70.0
    agent: dict[str, Any] = field(default_factory=dict)
    assertions: Assertions = field(default_factory=Assertions)
    base_dir: Path | None = None

    def build_turn_audio(self, index: int, seed: int = 0) -> np.ndarray:
        """Return the caller waveform for turn ``index`` at the scenario sample rate."""
        turn = self.turns[index]
        sr = self.sample_rate
        if turn.audio is not None:
            path = turn.audio
            if not path.is_absolute() and self.base_dir is not None:
                path = self.base_dir / path
            samples, rate = read_wav(path)
            speech = resample(samples, rate, sr)
        else:
            speech = synth_speech(
                turn.speech_duration_ms(), sr, seed=seed * 1009 + index, level_db=turn.level_db
            )
        if turn.hesitation is None:
            return speech
        cut = min(len(speech), int(turn.hesitation.at_ms * sr / 1000))
        gap = silence(turn.hesitation.duration_ms, sr)
        return np.concatenate([speech[:cut], gap, speech[cut:]]).astype(np.float32)


_TURN_KEYS = {
    "say",
    "audio",
    "duration_ms",
    "expected_transcript",
    "interrupt_after_ms",
    "pause_ms",
    "hesitation",
    "level_db",
}
_SCENARIO_KEYS = {
    "name",
    "description",
    "sample_rate",
    "wait_for_greeting_ms",
    "initial_silence_ms",
    "response_timeout_ms",
    "end_silence_ms",
    "tail_ms",
    "max_call_s",
    "noise_db",
    "agent",
    "assertions",
    "turns",
}


def _int(value: Any, field_name: str, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{where}: '{field_name}' must be a number, got {value!r}")
    if value < minimum:
        raise ScenarioError(f"{where}: '{field_name}' must be >= {minimum}, got {value}")
    return int(value)


def _parse_turn(raw: Any, index: int) -> Turn:
    where = f"turns[{index}]"
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: each turn must be a mapping")
    unknown = set(raw) - _TURN_KEYS
    if unknown:
        raise ScenarioError(f"{where}: unknown field(s) {sorted(unknown)}")
    if raw.get("say") is None and raw.get("audio") is None and raw.get("duration_ms") is None:
        raise ScenarioError(f"{where}: give at least one of 'say', 'audio' or 'duration_ms'")
    hes = raw.get("hesitation")
    hesitation = None
    if hes is not None:
        if not isinstance(hes, dict) or set(hes) != {"at_ms", "duration_ms"}:
            raise ScenarioError(f"{where}: 'hesitation' needs exactly 'at_ms' and 'duration_ms'")
        hesitation = Hesitation(
            _int(hes["at_ms"], "hesitation.at_ms", where),
            _int(hes["duration_ms"], "hesitation.duration_ms", where, 1),
        )
    interrupt = raw.get("interrupt_after_ms")
    if interrupt is not None:
        interrupt = _int(interrupt, "interrupt_after_ms", where)
    duration = raw.get("duration_ms")
    if duration is not None:
        duration = _int(duration, "duration_ms", where, 1)
    return Turn(
        say=str(raw["say"]) if raw.get("say") is not None else None,
        audio=Path(raw["audio"]) if raw.get("audio") is not None else None,
        duration_ms=duration,
        expected_transcript=(
            str(raw["expected_transcript"]) if raw.get("expected_transcript") is not None else None
        ),
        interrupt_after_ms=interrupt,
        pause_ms=_int(raw.get("pause_ms", 400), "pause_ms", where),
        hesitation=hesitation,
        level_db=float(raw.get("level_db", -20.0)),
    )


def parse_scenario(data: Any, base_dir: Path | None = None) -> Scenario:
    """Validate a parsed YAML document and build a ``Scenario``."""
    if not isinstance(data, dict):
        raise ScenarioError("scenario must be a YAML mapping")
    unknown = set(data) - _SCENARIO_KEYS
    if unknown:
        raise ScenarioError(f"unknown scenario field(s) {sorted(unknown)}")
    if not data.get("name"):
        raise ScenarioError("scenario needs a 'name'")
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ScenarioError("scenario needs a non-empty 'turns' list")
    turns = [_parse_turn(t, i) for i, t in enumerate(raw_turns)]
    sample_rate = _int(data.get("sample_rate", DEFAULT_SAMPLE_RATE), "sample_rate", "scenario", 8000)
    agent = data.get("agent") or {}
    if not isinstance(agent, dict):
        raise ScenarioError("'agent' must be a mapping of loopback agent options")
    raw_assert = data.get("assertions") or {}
    if not isinstance(raw_assert, dict):
        raise ScenarioError("'assertions' must be a mapping")
    allowed = set(Assertions.__dataclass_fields__)
    bad = set(raw_assert) - allowed
    if bad:
        raise ScenarioError(f"unknown assertion(s) {sorted(bad)}; allowed: {sorted(allowed)}")
    return Scenario(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        turns=turns,
        sample_rate=sample_rate,
        wait_for_greeting_ms=_int(data.get("wait_for_greeting_ms", 0), "wait_for_greeting_ms", "scenario"),
        initial_silence_ms=_int(data.get("initial_silence_ms", 500), "initial_silence_ms", "scenario"),
        response_timeout_ms=_int(data.get("response_timeout_ms", 6000), "response_timeout_ms", "scenario", 1),
        end_silence_ms=_int(data.get("end_silence_ms", 700), "end_silence_ms", "scenario", 1),
        tail_ms=_int(data.get("tail_ms", 800), "tail_ms", "scenario"),
        max_call_s=_int(data.get("max_call_s", 120), "max_call_s", "scenario", 1),
        noise_db=float(data.get("noise_db", -70.0)),
        agent=dict(agent),
        assertions=Assertions(**raw_assert),
        base_dir=base_dir,
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario YAML file."""
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{p}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ScenarioError(f"cannot read scenario {p}: {exc}") from exc
    try:
        return parse_scenario(data, base_dir=p.parent)
    except ScenarioError as exc:
        raise ScenarioError(f"{p}: {exc}") from exc


def builtin_bargein_scenario(
    interrupt_offsets_ms: list[int] | None = None,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Scenario:
    """A ready-made scenario for ``qsv bargein``: one question, then one interruption per offset."""
    offsets = interrupt_offsets_ms or [600, 1200]
    turns = [Turn(say="Hi, I would like to move my appointment to next week", duration_ms=2600)]
    for offset in offsets:
        turns.append(
            Turn(
                say="Sorry, actually make it Thursday morning",
                duration_ms=1800,
                interrupt_after_ms=offset,
                pause_ms=300,
            )
        )
    return Scenario(
        name="builtin-bargein",
        description="One opening request followed by interruptions at set offsets.",
        turns=turns,
        sample_rate=sample_rate,
        agent={"response_ms": [4500]},
    )
