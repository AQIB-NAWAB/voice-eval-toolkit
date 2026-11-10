"""Run one scripted call over a transport and score it."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from qsv.audio import DEFAULT_FRAME_MS
from qsv.caller import CallerSession
from qsv.metrics import AnalysisConfig, CallMetrics, TurnPlan, analyze_call
from qsv.scenario import Scenario
from qsv.transports.base import CallContext, Transport
from qsv.vad import SpeechGate, make_vad


@dataclass
class CallResult:
    call_id: str
    ok: bool
    metrics: CallMetrics | None = None
    error: str | None = None
    wall_time_s: float = 0.0
    started_unix_ns: int = 0
    connect_time_s: float = 0.0
    plans: list[TurnPlan] = field(default_factory=list)
    skipped_interrupts: list[int] = field(default_factory=list)
    timed_out_turns: list[int] = field(default_factory=list)
    caller_audio: np.ndarray | None = None
    agent_audio: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "ok": self.ok,
            "error": self.error,
            "wall_time_s": round(self.wall_time_s, 4),
            "connect_time_s": round(self.connect_time_s, 4),
            "skipped_interrupts": self.skipped_interrupts,
            "timed_out_turns": self.timed_out_turns,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


async def run_call(
    scenario: Scenario,
    transport: Transport,
    *,
    call_id: str,
    seed: int = 0,
    analysis: AnalysisConfig | None = None,
    agent_options: dict | None = None,
    realtime: bool | None = None,
    keep_audio: bool = False,
) -> CallResult:
    """Place one call, drive the scripted caller until it finishes, and analyse the recording.

    Failures (connect errors, transport exceptions) are returned as ``ok=False`` results
    rather than raised, so a load run keeps going and reports them. ``NotImplementedError``
    is re-raised: it means the transport itself is a stub and no call can succeed.
    """
    cfg = analysis or AnalysisConfig()
    frame_ms = cfg.frame_ms or DEFAULT_FRAME_MS
    pace = transport.realtime if realtime is None else realtime
    sr = scenario.sample_rate
    result = CallResult(call_id=call_id, ok=False, started_unix_ns=time.time_ns())
    wall_start = time.perf_counter()

    caller = CallerSession(scenario, seed=seed, frame_ms=frame_ms, reaction_lag_ms=transport.live_lag_ms)
    options = {**scenario.agent, **(agent_options or {})}
    ctx = CallContext(
        call_id=call_id,
        scenario_name=scenario.name,
        sample_rate=sr,
        frame_ms=frame_ms,
        seed=seed,
        reference_texts=[t.reference_text for t in scenario.turns],
        current_turn=lambda: caller.current_turn,
        agent_options=options,
    )
    try:
        try:
            await transport.connect(ctx)
        except NotImplementedError:
            raise
        except Exception as exc:
            result.error = f"connect failed: {type(exc).__name__}: {exc}"
            return result
        result.connect_time_s = time.perf_counter() - wall_start
        result.started_unix_ns = time.time_ns()

        watcher = SpeechGate(
            make_vad("energy", sr, threshold_db=cfg.vad_threshold_db),
            sr,
            frame_ms=frame_ms,
            onset_ms=cfg.vad_onset_ms,
            hangover_ms=60,
        )
        caller_frames: list[np.ndarray] = []
        agent_frames: list[np.ndarray] = []
        clock0 = time.perf_counter()
        frame_s = frame_ms / 1000.0
        i = 0
        while not caller.done:
            frame = caller.next_frame(watcher.active)
            await transport.send_frame(frame)
            heard = await transport.recv_frame()
            watcher.update(heard)
            caller_frames.append(frame)
            agent_frames.append(heard)
            i += 1
            if pace:
                delay = clock0 + i * frame_s - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            elif i % 50 == 0:
                await asyncio.sleep(0)  # let other simulated calls run

        caller_audio = np.concatenate(caller_frames) if caller_frames else np.zeros(0, np.float32)
        agent_audio = np.concatenate(agent_frames) if agent_frames else np.zeros(0, np.float32)
        recording = transport.agent_recording()
        if recording is not None:
            agent_audio = np.zeros(len(caller_audio), dtype=np.float32)
            keep = min(len(recording), len(agent_audio))
            agent_audio[:keep] = recording[:keep]
        result.plans = list(caller.plans)
        result.skipped_interrupts = list(caller.skipped_interrupts)
        result.timed_out_turns = list(caller.timed_out_turns)
        result.metrics = analyze_call(
            caller_audio,
            agent_audio,
            sr,
            caller.plans,
            transcripts=list(transport.transcripts),
            config=cfg,
        )
        result.ok = True
        if keep_audio:
            result.caller_audio, result.agent_audio = caller_audio, agent_audio
    except NotImplementedError:
        raise
    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            await transport.close()
        finally:
            result.wall_time_s = time.perf_counter() - wall_start
    return result
