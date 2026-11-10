from __future__ import annotations

import pytest

from conftest import SCENARIOS, run

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from qsv.load import LoadConfig, run_load  # noqa: E402
from qsv.observability import (  # noqa: E402
    Telemetry,
    TelemetryConfig,
    config_from_env,
    langfuse_config,
    metrics_endpoint,
    parse_headers,
    phoenix_config,
    traces_endpoint,
)
from qsv.scenario import load_scenario  # noqa: E402
from qsv.transports.loopback import LoopbackTransport  # noqa: E402


def results(name="barge_in.yaml", callers=2):
    sc = load_scenario(SCENARIOS / name)
    return run(run_load(sc, LoopbackTransport, LoadConfig(callers=callers), run_id="otel-test")).results


def test_disabled_telemetry_is_a_noop():
    t = Telemetry.disabled()
    assert not t.enabled
    for r in results(callers=1):
        t.record_call(r, scenario="s", transport="loopback", run_id="x")
    t.shutdown()
    assert not Telemetry.start(TelemetryConfig()).enabled


def test_spans_per_call_and_per_turn():
    exporter = InMemorySpanExporter()
    t = Telemetry.start(
        TelemetryConfig(endpoint="http://localhost:6006", project="demo"), span_exporter=exporter
    )
    rs = results(callers=2)
    for r in rs:
        t.record_call(r, scenario="barge-in", transport="loopback", run_id="otel-test")
    t.shutdown()
    spans = exporter.get_finished_spans()
    calls = [s for s in spans if s.name == "qsv.call"]
    turns = [s for s in spans if s.name == "qsv.turn"]
    assert len(calls) == 2 and len(turns) == 8
    root = calls[0]
    assert root.attributes["qsv.scenario"] == "barge-in"
    assert root.attributes["langfuse.session.id"] == "otel-test"
    assert root.attributes["session.id"] == "otel-test"
    assert root.attributes["qsv.interruptions_yielded"] == 3
    assert root.resource.attributes["openinference.project.name"] == "demo"
    child = next(s for s in turns if s.parent is not None and s.parent.span_id == root.context.span_id)
    assert child.context.trace_id == root.context.trace_id
    stops = [
        s.attributes["qsv.bargein.stop_latency_ms"]
        for s in turns
        if "qsv.bargein.stop_latency_ms" in s.attributes
    ]
    assert len(stops) == 6 and all(100 < v < 400 for v in stops)
    assert all(s.end_time >= s.start_time for s in spans)


def test_failed_call_sets_error_status():
    from qsv.session import CallResult

    exporter = InMemorySpanExporter()
    t = Telemetry.start(TelemetryConfig(endpoint="http://x"), span_exporter=exporter)
    t.record_call(
        CallResult(call_id="bad", ok=False, error="connect failed"),
        scenario="s",
        transport="livekit",
        run_id="r",
    )
    t.shutdown()
    (span,) = exporter.get_finished_spans()
    assert span.attributes["qsv.error"] == "connect failed"
    assert span.status.status_code.name == "ERROR"


def test_metrics_histograms_when_enabled():
    reader = InMemoryMetricReader()
    exporter = InMemorySpanExporter()
    t = Telemetry.start(
        TelemetryConfig(endpoint="http://x", metrics=True), span_exporter=exporter, metric_reader=reader
    )
    for r in results(callers=2):
        t.record_call(r, scenario="barge-in", transport="loopback", run_id="m")
    data = reader.get_metrics_data()
    names = {m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
    assert {"qsv.turn.gap", "qsv.bargein.stop_latency", "qsv.call.overlap", "qsv.calls"} <= names
    t.shutdown()


def test_endpoint_normalisation():
    assert traces_endpoint("http://localhost:6006") == "http://localhost:6006/v1/traces"
    assert traces_endpoint("http://h/v1/traces/") == "http://h/v1/traces"
    assert (
        traces_endpoint("https://cloud.langfuse.com/api/public/otel")
        == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    )
    assert metrics_endpoint("http://h:4318/v1/traces") == "http://h:4318/v1/metrics"


def test_parse_headers():
    assert parse_headers("a=1, b=2=3") == {"a": "1", "b": "2=3"}
    assert parse_headers(["x=y"]) == {"x": "y"}
    assert parse_headers(None) == {}
    with pytest.raises(ValueError):
        parse_headers(["nope"])


def test_langfuse_preset(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    cfg = langfuse_config()
    assert cfg.endpoint == "https://cloud.langfuse.com/api/public/otel"
    import base64

    assert cfg.headers["Authorization"] == "Basic " + base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    assert cfg.headers["x-langfuse-ingestion-version"] == "4"
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000/")
    assert langfuse_config().endpoint == "http://localhost:3000/api/public/otel"
    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    with pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"):
        langfuse_config()


def test_phoenix_preset(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    cfg = phoenix_config()
    assert cfg.endpoint == "http://localhost:6006" and cfg.headers == {}
    monkeypatch.setenv("PHOENIX_API_KEY", "abc")
    assert phoenix_config("https://phoenix.example.com").headers == {"Authorization": "Bearer abc"}


def test_config_from_env(monkeypatch):
    for v in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ):
        monkeypatch.delenv(v, raising=False)
    assert not config_from_env().enabled
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "k=v")
    cfg = config_from_env()
    assert cfg.enabled and cfg.headers == {"k": "v"}


def test_real_otlp_exporter_can_be_constructed_without_sending():
    t = Telemetry.start(TelemetryConfig(endpoint="http://127.0.0.1:9", headers={"a": "b"}))
    assert t.enabled
    t._provider.shutdown()


def test_langfuse_style_export_reaches_an_otlp_http_receiver(monkeypatch):
    """Send real OTLP/HTTP to a local receiver and check path and auth headers."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append({"path": self.path, "headers": dict(self.headers), "size": len(body)})
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-y")
        monkeypatch.setenv("LANGFUSE_HOST", f"http://127.0.0.1:{server.server_port}")
        telemetry = Telemetry.start(langfuse_config())
        for r in results(callers=1):
            telemetry.record_call(r, scenario="barge-in", transport="loopback", run_id="rx")
        telemetry.shutdown()
    finally:
        server.shutdown()
    assert seen, "no OTLP request arrived"
    assert seen[0]["path"] == "/api/public/otel/v1/traces"
    headers = {k.lower(): v for k, v in seen[0]["headers"].items()}
    assert headers["authorization"].startswith("Basic ")
    assert headers["x-langfuse-ingestion-version"] == "4"
    assert seen[0]["size"] > 200
