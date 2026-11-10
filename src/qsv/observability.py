"""OpenTelemetry export (optional, off unless configured).

``Telemetry.disabled()`` is a no-op and never imports OpenTelemetry. When an endpoint is
configured, each finished call becomes one trace:

* a root span ``qsv.call`` with the call's summary metrics as attributes, and
* one child span ``qsv.turn`` per scripted caller turn, timed from the caller starting to
  speak until the agent's reply ends (or the caller's turn ends if there was no reply),
  with ``qsv.turn.gap_ms``, ``qsv.bargein.stop_latency_ms``, ``qsv.turn.false_interruption``,
  ``qsv.turn.overlap_ms`` and ``qsv.turn.wer`` where they apply.

Spans are emitted after the call finishes, using the measured call timeline for start and end
times. Optionally (``metrics=True``) histograms for turn gap, stop latency and overlap are
exported as OTLP metrics too. Langfuse and Phoenix ingest traces only, so metrics are for
collectors such as Grafana Alloy or the OpenTelemetry Collector.

Backends
--------
Langfuse: OTLP/HTTP at ``<host>/api/public/otel`` with Basic auth (public:secret key) and the
``x-langfuse-ingestion-version: 4`` header. Arize Phoenix: OTLP/HTTP at ``<host>/v1/traces``
(port 6006 locally) with an optional ``Authorization: Bearer`` key. See docs/architecture.md.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any

from qsv.session import CallResult

LANGFUSE_DEFAULT_HOST = "https://cloud.langfuse.com"
PHOENIX_DEFAULT_ENDPOINT = "http://localhost:6006"


@dataclass
class TelemetryConfig:
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    service_name: str = "qsv"
    project: str = "qsv"
    metrics: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)


def traces_endpoint(endpoint: str) -> str:
    """Normalise a base URL or full URL to the OTLP/HTTP traces URL."""
    e = endpoint.rstrip("/")
    return e if e.endswith("/v1/traces") else e + "/v1/traces"


def metrics_endpoint(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    if e.endswith("/v1/traces"):
        e = e[: -len("/v1/traces")]
    return e if e.endswith("/v1/metrics") else e + "/v1/metrics"


def parse_headers(pairs: list[str] | str | None) -> dict[str, str]:
    """Parse ``k=v`` items (list, or one comma separated string as in OTEL_EXPORTER_OTLP_HEADERS)."""
    if not pairs:
        return {}
    items = pairs.split(",") if isinstance(pairs, str) else pairs
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"header '{item}' must look like key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def langfuse_config(
    public_key: str | None = None, secret_key: str | None = None, host: str | None = None
) -> TelemetryConfig:
    """Langfuse (cloud or self-hosted) from arguments or LANGFUSE_* environment variables."""
    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    host = (
        host
        or os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL")
        or LANGFUSE_DEFAULT_HOST
    )
    if not public_key or not secret_key:
        raise ValueError("Langfuse export needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return TelemetryConfig(
        endpoint=host.rstrip("/") + "/api/public/otel",
        headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
    )


def phoenix_config(endpoint: str | None = None, api_key: str | None = None) -> TelemetryConfig:
    """Arize Phoenix (local or cloud) from arguments or PHOENIX_* environment variables."""
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or PHOENIX_DEFAULT_ENDPOINT
    api_key = api_key or os.environ.get("PHOENIX_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return TelemetryConfig(endpoint=endpoint, headers=headers)


def config_from_env() -> TelemetryConfig:
    """Standard OTLP environment variables. Disabled when no endpoint is set."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    headers = parse_headers(
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS") or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    )
    return TelemetryConfig(endpoint=endpoint, headers=headers)


def _ns(call_start_ns: int, seconds: float) -> int:
    return call_start_ns + int(seconds * 1e9)


class Telemetry:
    """Emit call and turn spans. Construct with ``Telemetry.disabled()`` or ``Telemetry.start``."""

    def __init__(self) -> None:
        self.enabled = False
        self._tracer: Any = None
        self._provider: Any = None
        self._meter_provider: Any = None
        self._hist: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}

    @classmethod
    def disabled(cls) -> Telemetry:
        return cls()

    @classmethod
    def start(
        cls,
        config: TelemetryConfig,
        *,
        span_exporter: Any = None,
        metric_reader: Any = None,
    ) -> Telemetry:
        """Set up tracing. ``span_exporter`` and ``metric_reader`` let tests inject in-memory ones."""
        self = cls()
        if not config.enabled and span_exporter is None:
            return self
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry export needs the otel extra: pip install voice-eval-toolkit[otel]"
            ) from exc

        resource = Resource.create(
            {
                "service.name": config.service_name,
                "openinference.project.name": config.project,
            }
        )
        provider = TracerProvider(resource=resource)
        if span_exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=traces_endpoint(config.endpoint or ""), headers=config.headers or None
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        self._provider = provider
        self._tracer = provider.get_tracer("qsv", "0.1.0")
        self.enabled = True

        if config.metrics or metric_reader is not None:
            from opentelemetry.sdk.metrics import MeterProvider

            if metric_reader is None:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=metrics_endpoint(config.endpoint or ""), headers=config.headers or None
                    )
                )
            self._meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            meter = self._meter_provider.get_meter("qsv", "0.1.0")
            self._hist = {
                "turn_gap": meter.create_histogram(
                    "qsv.turn.gap", unit="ms", description="Caller end to agent first audio"
                ),
                "stop": meter.create_histogram(
                    "qsv.bargein.stop_latency", unit="ms", description="Caller onset to agent audio stop"
                ),
                "overlap": meter.create_histogram(
                    "qsv.call.overlap", unit="ms", description="Caller and agent speaking together"
                ),
            }
            self._counters = {
                "calls": meter.create_counter("qsv.calls", description="Calls placed"),
                "false": meter.create_counter(
                    "qsv.false_interruptions", description="Agent cut the caller off"
                ),
            }
        return self

    # ------------------------------------------------------------------ recording

    def record_call(
        self, result: CallResult, *, scenario: str, transport: str, run_id: str, mode: str = "load"
    ) -> None:
        if not self.enabled:
            return
        from opentelemetry import trace

        common = {
            "session.id": run_id,
            "langfuse.session.id": run_id,
            "langfuse.trace.name": f"qsv:{scenario}",
            "qsv.run_id": run_id,
            "qsv.scenario": scenario,
            "qsv.transport": transport,
        }
        start_ns = result.started_unix_ns
        end_ns = start_ns + int(result.wall_time_s * 1e9)
        m = result.metrics
        if m is not None:
            end_ns = max(end_ns, _ns(start_ns, m.duration_s))
        root = self._tracer.start_span(
            "qsv.call",
            start_time=start_ns,
            attributes={
                **common,
                "openinference.span.kind": "CHAIN",
                "qsv.mode": mode,
                "qsv.call_id": result.call_id,
                "qsv.ok": result.ok,
            },
        )
        ctx = trace.set_span_in_context(root)
        if result.error:
            root.set_attribute("qsv.error", result.error)
            root.set_status(trace.Status(trace.StatusCode.ERROR, result.error))
        if m is not None:
            self._set_call_attrs(root, m)
            for t in m.turns:
                end_s = t.agent_end_s if t.agent_end_s is not None else t.caller_end_s
                span = self._tracer.start_span(
                    "qsv.turn",
                    context=ctx,
                    start_time=_ns(start_ns, t.caller_start_s),
                    attributes={
                        **common,
                        "openinference.span.kind": "CHAIN",
                        "qsv.call_id": result.call_id,
                        "qsv.turn.index": t.index,
                        "qsv.turn.interrupt": t.interrupt,
                        "qsv.turn.false_interruption": t.false_interruption,
                        "qsv.turn.overlap_ms": round(t.overlap_ms, 1),
                    },
                )
                if t.turn_gap_ms is not None:
                    span.set_attribute("qsv.turn.gap_ms", round(t.turn_gap_ms, 1))
                if t.barge_in is not None:
                    span.set_attribute("qsv.bargein.agent_was_speaking", t.barge_in.agent_was_speaking)
                    span.set_attribute("qsv.bargein.yielded", t.barge_in.yielded)
                    if t.barge_in.stop_latency_ms is not None:
                        span.set_attribute(
                            "qsv.bargein.stop_latency_ms", round(t.barge_in.stop_latency_ms, 1)
                        )
                if t.wer is not None:
                    span.set_attribute("qsv.turn.wer", round(t.wer, 4))
                if t.transcript:
                    span.set_attribute("output.value", t.transcript)
                span.end(end_time=_ns(start_ns, max(end_s, t.caller_end_s)))
            self._record_metrics(m, scenario, transport)
        root.end(end_time=end_ns)

    @staticmethod
    def _set_call_attrs(span: Any, m: Any) -> None:
        span.set_attribute("qsv.turns", len(m.turns))
        span.set_attribute("qsv.false_interruptions", m.false_interruptions)
        span.set_attribute("qsv.interruptions_attempted", m.interruptions_attempted)
        span.set_attribute("qsv.interruptions_yielded", m.interruptions_yielded)
        span.set_attribute("qsv.overlap_ms", round(m.total_overlap_ms, 1))
        if m.time_to_first_audio_ms is not None:
            span.set_attribute("qsv.time_to_first_audio_ms", round(m.time_to_first_audio_ms, 1))
        gaps = m.turn_gaps_ms
        if gaps:
            span.set_attribute("qsv.turn_gap_ms.max", round(max(gaps), 1))
        if m.stop_latencies_ms:
            span.set_attribute("qsv.bargein.stop_latency_ms.max", round(max(m.stop_latencies_ms), 1))
        if m.wer is not None:
            span.set_attribute("qsv.wer", round(m.wer, 4))

    def _record_metrics(self, m: Any, scenario: str, transport: str) -> None:
        if not self._hist:
            return
        attrs = {"qsv.scenario": scenario, "qsv.transport": transport}
        for gap in m.turn_gaps_ms:
            self._hist["turn_gap"].record(gap, attrs)
        for stop in m.stop_latencies_ms:
            self._hist["stop"].record(stop, attrs)
        self._hist["overlap"].record(m.total_overlap_ms, attrs)
        self._counters["calls"].add(1, attrs)
        if m.false_interruptions:
            self._counters["false"].add(m.false_interruptions, attrs)

    def shutdown(self) -> None:
        """Flush and stop exporters."""
        if self._provider is not None:
            self._provider.force_flush()
            self._provider.shutdown()
        if self._meter_provider is not None:
            self._meter_provider.force_flush()
            self._meter_provider.shutdown()
        self.enabled = False
