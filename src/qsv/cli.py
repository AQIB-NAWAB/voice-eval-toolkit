"""Command line interface: ``qsv bargein``, ``qsv load`` and ``qsv validate``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

from qsv import __version__
from qsv.audio import write_stereo_wav
from qsv.load import LoadConfig, LoadResult, evaluate_assertions, run_load, summarize
from qsv.metrics import AnalysisConfig
from qsv.observability import (
    Telemetry,
    TelemetryConfig,
    config_from_env,
    langfuse_config,
    parse_headers,
    phoenix_config,
)
from qsv.report import build_report, write_reports
from qsv.scenario import Assertions, Scenario, ScenarioError, builtin_bargein_scenario, load_scenario
from qsv.transports import TRANSPORT_NAMES, get_transport_factory

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("transport")
    g.add_argument(
        "--transport",
        choices=TRANSPORT_NAMES,
        default="loopback",
        help="audio path to the agent (default: loopback mock agent)",
    )
    g.add_argument(
        "--agent-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="loopback mock agent option, repeatable (e.g. stop_latency_ms=150)",
    )
    g.add_argument("--livekit-url", help="LiveKit server URL (or LIVEKIT_URL)")
    g.add_argument("--livekit-room-prefix", default="qsv", help="rooms are named <prefix>-<call id>")
    g.add_argument("--livekit-agent-name", help="explicit agent dispatch name (omit for automatic dispatch)")
    g.add_argument(
        "--livekit-join-timeout", type=float, default=20.0, help="seconds to wait for the agent audio track"
    )
    g.add_argument("--sip-uri", help="SIP URI to dial (SIP transport is a stub)")
    t = p.add_argument_group("analysis")
    t.add_argument(
        "--vad",
        choices=["energy", "webrtc"],
        default="energy",
        help="voice activity detector (webrtc needs the vad extra)",
    )
    t.add_argument("--vad-threshold-db", type=float, default=-45.0, help="energy VAD floor in dBFS")
    t.add_argument(
        "--min-gap-ms",
        type=float,
        default=300.0,
        help="agent audio starting less than this after the caller stops counts as a cut-off",
    )
    t.add_argument(
        "--stop-timeout-ms", type=float, default=2000.0, help="slower than this is a failure to yield"
    )
    r = p.add_argument_group("run")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", help="report directory (default: qsv-report/<command>)")
    r.add_argument(
        "--save-audio",
        action="store_true",
        help="write stereo WAVs (caller left, agent right) for the first call",
    )
    r.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        default=None,
        help="pace audio in real time even for loopback",
    )
    r.add_argument(
        "--fast", dest="realtime", action="store_false", help="never pace (only valid for loopback)"
    )
    r.add_argument("--quiet", action="store_true")
    a = p.add_argument_group("pass/fail limits (override the scenario's assertions)")
    a.add_argument("--max-stop-p95-ms", type=float)
    a.add_argument("--max-turn-gap-p95-ms", type=float)
    a.add_argument("--max-false-interruptions", type=int)
    a.add_argument("--max-wer", type=float, help="corpus WER as a fraction, e.g. 0.1")
    a.add_argument("--min-success-rate", type=float)
    o = p.add_argument_group("OpenTelemetry (off unless configured)")
    o.add_argument(
        "--otel-preset",
        choices=["langfuse", "phoenix"],
        help="use LANGFUSE_* or PHOENIX_* environment variables",
    )
    o.add_argument("--otel-endpoint", help="OTLP/HTTP endpoint (base URL or .../v1/traces)")
    o.add_argument("--otel-header", action="append", default=[], metavar="KEY=VALUE")
    o.add_argument("--otel-project", default="qsv", help="Phoenix project name")
    o.add_argument(
        "--otel-metrics", action="store_true", help="also export histograms to <endpoint>/v1/metrics"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsv", description="Evaluate voice agents: barge-in, turn-taking and simulated-caller load."
    )
    parser.add_argument("--version", action="version", version=f"qsv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser(
        "bargein",
        help="barge-in and turn-taking harness",
        description=(
            "Play a scripted caller with interruptions and measure stop latency, "
            "cut-offs, turn gaps and overlap."
        ),
    )
    b.add_argument("--scenario", help="scenario YAML (default: a built-in interruption script)")
    b.add_argument(
        "--interrupt-at",
        default="600,1200",
        help=(
            "for the built-in script: comma separated offsets in ms after the agent starts "
            "replying (default 600,1200)"
        ),
    )
    b.add_argument("--repeat", type=int, default=1, help="how many calls to place, one after another")
    _add_common(b)

    ld = sub.add_parser(
        "load",
        help="simulated-caller load and eval run",
        description=(
            "Run N concurrent scripted callers and report latency percentiles, "
            "barge-in metrics and optional WER."
        ),
    )
    ld.add_argument("--scenario", required=True, help="scenario YAML")
    ld.add_argument("--callers", type=int, default=5, help="number of calls")
    ld.add_argument("--concurrency", type=int, help="max simultaneous calls (default: all)")
    ld.add_argument("--ramp-up", type=float, default=0.0, help="spread call starts over this many seconds")
    _add_common(ld)

    v = sub.add_parser("validate", help="check scenario files")
    v.add_argument("files", nargs="+")
    return parser


# --------------------------------------------------------------------------- helpers


def _agent_options(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--agent-opt '{item}' must look like KEY=VALUE")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _transport_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.transport == "livekit":
        return {
            "url": args.livekit_url,
            "room_prefix": args.livekit_room_prefix,
            "agent_name": args.livekit_agent_name,
            "agent_join_timeout_s": args.livekit_join_timeout,
        }
    if args.transport == "sip":
        return {"uri": args.sip_uri or ""}
    return {}


def _telemetry(args: argparse.Namespace) -> Telemetry:
    if args.otel_preset == "langfuse":
        cfg = langfuse_config()
    elif args.otel_preset == "phoenix":
        cfg = phoenix_config(args.otel_endpoint)
    elif args.otel_endpoint:
        cfg = TelemetryConfig(endpoint=args.otel_endpoint)
    else:
        cfg = config_from_env()
    if not cfg.enabled:
        return Telemetry.disabled()
    cfg.headers.update(parse_headers(args.otel_header))
    cfg.project = args.otel_project
    cfg.metrics = args.otel_metrics
    return Telemetry.start(cfg)


def _assertions(base: Assertions, args: argparse.Namespace) -> Assertions:
    overrides = {
        "max_bargein_stop_p95_ms": args.max_stop_p95_ms,
        "max_turn_gap_p95_ms": args.max_turn_gap_p95_ms,
        "max_false_interruptions": args.max_false_interruptions,
        "max_wer": args.max_wer,
        "min_success_rate": args.min_success_rate,
    }
    return dataclasses.replace(base, **{k: v for k, v in overrides.items() if v is not None})


def _fmt(v: float | None, suffix: str = " ms") -> str:
    return "n/a" if v is None else f"{v:.0f}{suffix}"


def _print_summary(
    summary: dict[str, Any], checks: list, json_path: Path, md_path: Path, wall: float
) -> None:
    g = summary["turn_gap_ms"]
    b = summary["bargein"]["stop_latency_ms"]
    print(
        f"calls: {summary['calls_ok']}/{summary['calls']} ok in {wall:.1f}s, {summary['turns']} turns scored"
    )
    print(f"turn gap       p50 {_fmt(g['p50'])}  p95 {_fmt(g['p95'])}  p99 {_fmt(g['p99'])}")
    yr = summary["bargein"]["yield_rate"]
    print(
        f"barge-in stop  p50 {_fmt(b['p50'])}  p95 {_fmt(b['p95'])}  p99 {_fmt(b['p99'])}  "
        f"(yielded {summary['bargein']['interruptions_yielded']}"
        f"/{summary['bargein']['interruptions_attempted']}"
        f"{'' if yr is None else f', {yr * 100:.0f}%'})"
    )
    print(f"false interruptions: {summary['false_interruptions']['count']}")
    if summary["wer"]["corpus"] is not None:
        print(f"WER (corpus): {summary['wer']['corpus'] * 100:.1f}% over {summary['wer']['words']} words")
    for c in checks:
        actual = "n/a" if c.actual is None else f"{c.actual:g}"
        print(f"check {c.name}: {'pass' if c.passed else 'FAIL'} (limit {c.limit:g}, actual {actual})")
    for e in summary["errors"]:
        print(f"error: {e}", file=sys.stderr)
    print(f"report: {json_path}")
    print(f"        {md_path}")


def _run(args: argparse.Namespace, scenario: Scenario, *, mode: str, load_cfg: LoadConfig) -> int:
    if args.realtime is False and args.transport != "loopback":
        print("error: --fast is only valid with the loopback transport", file=sys.stderr)
        return EXIT_USAGE
    load_cfg.realtime = args.realtime
    load_cfg.agent_options = _agent_options(args.agent_opt)
    load_cfg.keep_audio_for = 1 if args.save_audio else 0
    factory = get_transport_factory(args.transport, _transport_options(args))
    analysis = AnalysisConfig(
        min_silence_gap_ms=args.min_gap_ms,
        stop_timeout_ms=args.stop_timeout_ms,
        vad=args.vad,
        vad_threshold_db=args.vad_threshold_db,
    )
    telemetry = _telemetry(args)
    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.out or f"qsv-report/{mode}")

    def on_result(res: Any) -> None:
        telemetry.record_call(res, scenario=scenario.name, transport=args.transport, run_id=run_id, mode=mode)

    try:
        result: LoadResult = asyncio.run(
            run_load(scenario, factory, load_cfg, analysis=analysis, on_result=on_result, run_id=run_id)
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        telemetry.shutdown()

    summary = summarize(result.results)
    checks = evaluate_assertions(_assertions(scenario.assertions, args), summary)
    report = build_report(
        result, mode=mode, transport=args.transport, run_id=run_id, assertion_results=checks
    )
    json_path, md_path = write_reports(report, out_dir)

    if args.save_audio:
        first = next((r for r in result.results if r.caller_audio is not None), None)
        if first is not None and first.agent_audio is not None:
            wav = out_dir / f"{first.call_id}.wav"
            write_stereo_wav(wav, first.caller_audio, first.agent_audio, scenario.sample_rate)
            if not args.quiet:
                print(f"audio: {wav}")
    if not args.quiet:
        _print_summary(summary, checks, json_path, md_path, result.wall_time_s)
    failed = summary["calls_ok"] == 0 or any(not c.passed for c in checks)
    return EXIT_FAILED if failed else EXIT_OK


def _cmd_bargein(args: argparse.Namespace) -> int:
    if args.scenario:
        scenario = load_scenario(args.scenario)
    else:
        try:
            offsets = [int(x) for x in args.interrupt_at.split(",") if x.strip()]
        except ValueError:
            print("error: --interrupt-at must be comma separated integers (ms)", file=sys.stderr)
            return EXIT_USAGE
        scenario = builtin_bargein_scenario(offsets)
    if args.repeat < 1:
        print("error: --repeat must be at least 1", file=sys.stderr)
        return EXIT_USAGE
    return _run(
        args,
        scenario,
        mode="bargein",
        load_cfg=LoadConfig(callers=args.repeat, concurrency=1, seed=args.seed),
    )


def _cmd_load(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    if args.callers < 1:
        print("error: --callers must be at least 1", file=sys.stderr)
        return EXIT_USAGE
    cfg = LoadConfig(
        callers=args.callers, concurrency=args.concurrency, ramp_up_s=args.ramp_up, seed=args.seed
    )
    return _run(args, scenario, mode="load", load_cfg=cfg)


def _cmd_validate(args: argparse.Namespace) -> int:
    bad = 0
    for f in args.files:
        try:
            sc = load_scenario(f)
            print(f"ok   {f}: {sc.name} ({len(sc.turns)} turns, {sc.sample_rate} Hz)")
        except ScenarioError as exc:
            bad += 1
            print(f"FAIL {exc}", file=sys.stderr)
    return EXIT_FAILED if bad else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "bargein":
            return _cmd_bargein(args)
        if args.command == "load":
            return _cmd_load(args)
        return _cmd_validate(args)
    except (ScenarioError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
