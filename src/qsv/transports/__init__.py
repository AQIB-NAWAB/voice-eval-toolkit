"""Pluggable transports. ``get_transport_factory`` maps a name to a per-call constructor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qsv.transports.base import CallContext, Transport

TRANSPORT_NAMES = ("loopback", "livekit", "sip")

TransportFactory = Callable[[], Transport]


def get_transport_factory(name: str, options: dict[str, Any] | None = None) -> TransportFactory:
    """Return a callable that builds a fresh transport for each call.

    ``options`` are transport specific. For ``loopback`` they override the mock agent
    settings. For ``livekit`` they are ``LiveKitConfig`` fields. For ``sip`` they are
    ``SipConfig`` fields.
    """
    options = dict(options or {})
    name = name.lower()
    if name == "loopback":
        from qsv.transports.loopback import LoopbackTransport

        return LoopbackTransport  # configured per call from the scenario and context
    if name == "livekit":
        from qsv.transports.livekit_transport import LiveKitConfig, LiveKitTransport

        config = LiveKitConfig(**{k: v for k, v in options.items() if v is not None})
        return lambda: LiveKitTransport(config)
    if name == "sip":
        from qsv.transports.sip import SipConfig, SipTransport

        sip_config = SipConfig(**{k: v for k, v in options.items() if v is not None})
        return lambda: SipTransport(sip_config)
    raise ValueError(f"unknown transport '{name}'. Choose one of: {', '.join(TRANSPORT_NAMES)}")


__all__ = ["CallContext", "Transport", "TransportFactory", "TRANSPORT_NAMES", "get_transport_factory"]
