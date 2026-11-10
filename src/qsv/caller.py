"""The scripted caller: a tick-driven state machine.

Each tick is one audio frame (20 ms by default). ``next_frame`` is given whether the
agent is currently audible and returns the frame the caller sends. The caller reacts to
the agent only through that single boolean, the same way a person on a phone would.

Phases::

    LEAD_IN -> [COUNTDOWN | AWAIT_END | SPEAKING]
    SPEAKING -> AWAIT_RESPONSE -> COUNTDOWN (next turn interrupts)
                               -> AWAIT_END  -> PAUSE -> SPEAKING
                               -> TAIL (last turn)  -> DONE

COUNTDOWN is how a barge-in is scheduled: when the agent starts a reply, the caller
waits ``interrupt_after_ms`` and then speaks whether or not the agent is still talking.
"""

from __future__ import annotations

import enum
import math

import numpy as np

from qsv.audio import DEFAULT_FRAME_MS, frame_samples, silence
from qsv.metrics import TurnPlan
from qsv.scenario import Scenario


class Phase(enum.Enum):
    LEAD_IN = "lead_in"
    SPEAKING = "speaking"
    AWAIT_RESPONSE = "await_response"
    COUNTDOWN = "countdown"
    AWAIT_END = "await_end"
    PAUSE = "pause"
    TAIL = "tail"
    DONE = "done"


class CallerSession:
    def __init__(
        self,
        scenario: Scenario,
        *,
        seed: int = 0,
        frame_ms: int = DEFAULT_FRAME_MS,
        reaction_lag_ms: float = 0.0,
    ) -> None:
        self.scenario = scenario
        self.frame_ms = frame_ms
        self.reaction_lag_ms = reaction_lag_ms
        self.frame_len = frame_samples(scenario.sample_rate, frame_ms)
        self._noise_rng = np.random.default_rng(seed + 7919)
        self._audio = [scenario.build_turn_audio(i, seed) for i in range(len(scenario.turns))]
        self.plans: list[TurnPlan] = []
        self.skipped_interrupts: list[int] = []
        self.timed_out_turns: list[int] = []

        self.tick = 0
        self.phase = Phase.LEAD_IN
        self._phase_start = 0
        self._next_turn = 0
        self._cur_turn = -1
        self._cur_pos = 0
        self._countdown_target = 0
        self._interrupt_pending = False
        self._agent_idle = 0
        self._prev_agent = False
        self._turn_started_by_interrupt = False

    # ------------------------------------------------------------------ helpers

    def _ticks(self, ms: float) -> int:
        return int(math.ceil(ms / self.frame_ms))

    def _interrupt_ticks(self, ms: float) -> int:
        """Countdown length for a barge-in, shortened by how late the caller sees the agent."""
        return max(0, self._ticks(ms - self.reaction_lag_ms))

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_start = self.tick

    @property
    def current_turn(self) -> int | None:
        """Index of the turn being spoken right now, else the most recent one."""
        return self._cur_turn if self._cur_turn >= 0 else None

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    @property
    def elapsed_s(self) -> float:
        return self.tick * self.frame_ms / 1000.0

    def _has_next(self) -> bool:
        return self._next_turn < len(self.scenario.turns)

    def _start_next_turn(self, via_interrupt: bool) -> None:
        self._cur_turn = self._next_turn
        self._next_turn += 1
        self._cur_pos = 0
        self._turn_started_by_interrupt = via_interrupt
        self._enter(Phase.SPEAKING)

    def _after_response_decision(self, responded: bool) -> None:
        """Pick the next phase once the agent has answered or the wait timed out."""
        if not self._has_next():
            self._enter(Phase.AWAIT_END if responded else Phase.TAIL)
            return
        nxt = self.scenario.turns[self._next_turn]
        if responded and nxt.interrupt_after_ms is not None:
            self._countdown_target = self.tick + self._interrupt_ticks(nxt.interrupt_after_ms)
            self._enter(Phase.COUNTDOWN)
        elif responded:
            self._enter(Phase.AWAIT_END)
        else:
            if nxt.interrupt_after_ms is not None:
                self.skipped_interrupts.append(self._next_turn)
            self._enter(Phase.PAUSE)

    # ------------------------------------------------------------------ main step

    def next_frame(self, agent_active: bool) -> np.ndarray:
        sc = self.scenario
        agent_onset = agent_active and not self._prev_agent
        self._prev_agent = agent_active
        self._agent_idle = 0 if agent_active else self._agent_idle + 1
        elapsed = self.tick - self._phase_start

        if self.tick >= self._ticks(sc.max_call_s * 1000) and self.phase is not Phase.DONE:
            self._enter(Phase.DONE)

        if self.phase is Phase.LEAD_IN:
            nxt = sc.turns[0]
            greeting_wait = self._ticks(sc.wait_for_greeting_ms)
            if agent_onset or agent_active:
                if nxt.interrupt_after_ms is not None:
                    self._countdown_target = self.tick + self._interrupt_ticks(nxt.interrupt_after_ms)
                    self._enter(Phase.COUNTDOWN)
                else:
                    self._enter(Phase.AWAIT_END)
            elif elapsed >= max(self._ticks(sc.initial_silence_ms), greeting_wait):
                if nxt.interrupt_after_ms is not None:
                    self.skipped_interrupts.append(0)
                self._start_next_turn(via_interrupt=False)

        elif self.phase is Phase.AWAIT_RESPONSE:
            if agent_active:
                self._after_response_decision(True)
            elif elapsed >= self._ticks(sc.response_timeout_ms):
                self.timed_out_turns.append(self._cur_turn)
                self._after_response_decision(False)

        elif self.phase is Phase.COUNTDOWN:
            if self.tick >= self._countdown_target:
                self._start_next_turn(via_interrupt=True)

        elif self.phase is Phase.AWAIT_END:
            limit = self._ticks(sc.response_timeout_ms * 5)
            if self._agent_idle >= self._ticks(sc.end_silence_ms) or elapsed >= limit:
                self._enter(Phase.PAUSE if self._has_next() else Phase.TAIL)

        elif self.phase is Phase.PAUSE:
            pause = sc.turns[self._next_turn].pause_ms if self._has_next() else 0
            if not self._has_next():
                self._enter(Phase.TAIL)
            elif elapsed >= self._ticks(pause):
                self._start_next_turn(via_interrupt=False)

        elif self.phase is Phase.TAIL and elapsed >= self._ticks(sc.tail_ms):
            self._enter(Phase.DONE)

        frame = self._emit()
        self.tick += 1
        return frame

    def _emit(self) -> np.ndarray:
        if self.phase is Phase.SPEAKING:
            audio = self._audio[self._cur_turn]
            if self._cur_pos == 0:
                self.plans.append(
                    TurnPlan(
                        index=self._cur_turn,
                        start_s=self.tick * self.frame_ms / 1000.0,
                        end_s=(self.tick * self.frame_ms / 1000.0) + len(audio) / self.scenario.sample_rate,
                        interrupt=self._turn_started_by_interrupt,
                        reference_text=self.scenario.turns[self._cur_turn].reference_text,
                    )
                )
            chunk = audio[self._cur_pos : self._cur_pos + self.frame_len]
            self._cur_pos += self.frame_len
            if len(chunk) < self.frame_len:
                chunk = np.concatenate([chunk, np.zeros(self.frame_len - len(chunk), dtype=np.float32)])
            if self._cur_pos >= len(audio):
                self._enter_after_speaking()
            return chunk.astype(np.float32, copy=False)
        return silence(self.frame_ms, self.scenario.sample_rate, self.scenario.noise_db, self._noise_rng)

    def _enter_after_speaking(self) -> None:
        # The phase change takes effect on the next tick; the agent-activity edge detector
        # compares against the agent state at that tick, so an agent that never stopped
        # counts as an immediate response.
        self.phase = Phase.AWAIT_RESPONSE
        self._phase_start = self.tick + 1
