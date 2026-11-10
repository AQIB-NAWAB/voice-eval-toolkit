from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from qsv.audio import silence, synth_speech

SR = 8000
ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "examples" / "scenarios"


def run(coro):
    return asyncio.run(coro)


def timeline(total_s: float, bursts: list[tuple[float, float]], sr: int = SR, level_db: float = -20.0):
    """Silence of ``total_s`` with speech-like bursts at the given (start, end) seconds."""
    out = silence(total_s * 1000, sr)
    for i, (a, b) in enumerate(bursts):
        seg = synth_speech((b - a) * 1000, sr, seed=i + 3, level_db=level_db)
        i0 = int(round(a * sr))
        out[i0 : i0 + len(seg)] = seg
    return out


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
