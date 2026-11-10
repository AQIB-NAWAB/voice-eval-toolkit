# Contributing

Thanks for taking a look. This is a small project and the bar for contributions is simple: keep the measurements honest and the tests green.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,vad]"
ruff check .
pytest
```

Python 3.10 or newer. Everything except the LiveKit integration test runs without a network.

## Ground rules

- **Measure from audio.** Metrics are derived from the two recorded channels, never from agent internals, so they stay comparable across stacks. If a metric needs agent cooperation (like transcripts for WER), make that explicit in the docs.
- **Define it precisely.** A new metric needs an entry in `docs/metrics.md` with a definition, edge cases and known limits, plus tests on synthetic audio where the right answer is known.
- **Keep the loopback agent honest.** It is a mock. Changes to it should make test scenarios easier to control, and docs should never present its numbers as real agent performance.
- **Transports are pluggable.** A new transport subclasses `qsv.transports.base.Transport`, is import-guarded if it needs an optional dependency, and gets a line in `get_transport_factory`. See `docs/transports.md`.
- **No claims about production use.** QuickStart AI is in development. Write docs accordingly.

## Tests

- `pytest` runs the full suite (VAD, metric math, barge-in on synthetic audio, scenario parsing, the load runner, reports, CLI, OpenTelemetry).
- `QSV_LIVEKIT_INTEGRATION=1 pytest tests/test_livekit_integration.py` runs two real WebRTC calls against a local `livekit-server --dev` and the example mock agent.

## Style

`ruff check .` must be clean. `ruff format .` is welcome. Write docs in plain, direct language.

## Pull requests

Describe what you measured and how you checked it. If a change moves a number in the README, update the README output in the same PR.
