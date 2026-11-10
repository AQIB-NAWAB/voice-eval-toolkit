"""SIP transport (stub).

SIP dialing is not implemented yet. The class exists so the interface, options and CLI
flags are settled and so the failure message tells you what to do instead.

Planned approaches
------------------
1. **LiveKit SIP outbound (shortest path).** If the agent under test is reachable by phone
   number, create a LiveKit room and ask LiveKit's SIP service to dial it into that room
   with ``LiveKitAPI().sip.create_sip_participant(...)`` (needs an outbound SIP trunk id
   and ``wait_until_answered``). Our caller then joins the same room and the existing
   ``LiveKitTransport`` publishes caller audio and records the SIP participant's track.
   The only new code is the dial step and hang-up handling.
2. **PJSUA2 (direct SIP/RTP).** Register an account, place the call, and bridge
   ``AudioMediaPort`` callbacks to ``send_frame`` and ``recv_frame``. This gives wire-level
   control (codecs, DTMF, RTP stats) and is the route for impairment tests with tc netem.
   Expect 8 kHz G.711 (PCMU or PCMA), so analysis should run at 8 kHz.

Whichever is chosen, the contract is the same as every transport: ``connect`` places the
call, ``send_frame`` and ``recv_frame`` exchange one frame per tick in real time, and
``close`` hangs up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qsv.transports.base import CallContext, Transport

GUIDANCE = (
    "The SIP transport is not implemented yet. Options today: (1) run the agent behind "
    "LiveKit and use --transport livekit; (2) dial through a LiveKit SIP outbound trunk and "
    "reuse the LiveKit transport for audio (see docs/transports.md); (3) contribute a PJSUA2 "
    "implementation of qsv.transports.sip.SipTransport."
)


@dataclass
class SipConfig:
    uri: str = ""
    username: str | None = None
    password: str | None = None
    codec: str = "PCMU"
    trunk_id: str | None = None


class SipTransport(Transport):
    name = "sip"
    realtime = True

    def __init__(self, config: SipConfig | None = None) -> None:
        super().__init__()
        self.config = config or SipConfig()

    async def connect(self, ctx: CallContext) -> None:
        raise NotImplementedError(GUIDANCE)

    async def send_frame(self, frame: np.ndarray) -> None:
        raise NotImplementedError(GUIDANCE)

    async def recv_frame(self) -> np.ndarray:
        raise NotImplementedError(GUIDANCE)

    async def close(self) -> None:
        return None
