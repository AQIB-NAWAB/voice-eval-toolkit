"""Regenerate the images in docs/images from real qsv runs.

Needs matplotlib and pillow (pip install matplotlib pillow). Run from the repo root:

    python scripts/make_images.py

Everything shown comes from the loopback transport, i.e. the bundled mock agent.
Nothing here is typed in by hand: the terminal images are the real CLI output, and the charts
are drawn from the recorded audio and metrics of real runs.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from qsv.load import LoadConfig, run_load, summarize  # noqa: E402
from qsv.metrics import AnalysisConfig  # noqa: E402
from qsv.scenario import load_scenario  # noqa: E402
from qsv.transports import get_transport_factory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
INK = "#1a1a1a"
GRAY = "#6b6b6b"
LIGHT = "#d9d9d9"
ACCENT = "#1f5fa8"
SOFT = "#c9d9ec"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save_png(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    img = Image.open(path).convert("RGB").quantize(colors=64)
    img.save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------- terminal images


def run_cli(args: list[str]) -> str:
    cmd = [sys.executable, "-m", "qsv", *args]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.rstrip()


def terminal_png(name: str, command: str, output: str) -> None:
    mono = font_manager.findfont("DejaVu Sans Mono")
    font = ImageFont.truetype(mono, 17)
    cmd_lines = command.split("\n")
    lines = [("$ " + cmd_lines[0], True)] + [("  " + c, True) for c in cmd_lines[1:]]
    lines += [(line, False) for line in output.splitlines()]
    pad, lh = 22, 26
    width = max(int(font.getlength(t)) for t, _ in lines) + pad * 2
    height = lh * len(lines) + pad * 2
    img = Image.new("RGB", (max(width, 760), height), "#1c1c1c")
    d = ImageDraw.Draw(img)
    for i, (text, is_cmd) in enumerate(lines):
        y = pad + i * lh
        if is_cmd:
            if text.startswith("$ "):
                d.text((pad, y), "$ ", font=font, fill="#8fbf8f")
            d.text((pad + font.getlength("$ "), y), text[2:], font=font, fill="#f2f2f2")
        else:
            d.text((pad, y), text, font=font, fill="#cfcfcf")
    path = OUT / name
    img.quantize(colors=32).save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------- real runs


def run_calls(scenario_file: str, callers: int, agent_opts: dict[str, str], keep: int = 0):
    scenario = load_scenario(ROOT / scenario_file)
    cfg = LoadConfig(callers=callers, seed=1, agent_options=agent_opts, keep_audio_for=keep)
    factory = get_transport_factory("loopback", {})
    return asyncio.run(run_load(scenario, factory, cfg, analysis=AnalysisConfig()))


def envelope(x: np.ndarray, sr: int, win_ms: int = 20) -> tuple[np.ndarray, np.ndarray]:
    n = int(sr * win_ms / 1000)
    frames = len(x) // n
    rms = np.sqrt((x[: frames * n].reshape(frames, n) ** 2).mean(axis=1))
    return np.arange(frames) * n / sr, rms


# ---------------------------------------------------------------------------- charts


def timeline_png() -> None:
    res = run_calls("examples/scenarios/barge_in.yaml", 1, {}, keep=1)
    call = res.results[0]
    sr = res.scenario.sample_rate
    caller, agent = call.caller_audio, call.agent_audio
    cuts = [
        t
        for t in call.metrics.turns
        if t.interrupt and t.barge_in is not None and t.barge_in.stop_latency_ms is not None
    ]
    first = cuts[0]

    fig, (top, zoom) = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [1, 1.25]})

    def lanes(ax, x0, x1):
        for y0, audio, color in ((1.0, caller, ACCENT), (0.0, agent, INK)):
            t, e = envelope(audio, sr)
            e = e / max(e.max(), 1e-9) * 0.36
            keep = (t >= x0 - 0.1) & (t <= x1 + 0.1)
            ax.fill_between(
                t[keep], y0 - e[keep], y0 + e[keep], color=color, alpha=0.85, linewidth=0, step="mid"
            )
            ax.axhline(y0, color=LIGHT, linewidth=0.8, zorder=0)
        ax.set_yticks([0.0, 1.0], ["Agent", "Caller"], fontsize=12)
        ax.set_xlim(x0, x1)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    total = len(caller) / sr
    lanes(top, 0, total)
    top.set_ylim(-0.6, 1.5)
    for i, t in enumerate(cuts, 1):
        top.axvline(t.caller_start_s, color=ACCENT, linestyle=(0, (4, 3)), linewidth=1.2)
        top.text(t.caller_start_s + 0.12, 1.42, f"interruption {i}", fontsize=9, color=ACCENT, va="top")
    top.set_xlabel("seconds into the call")
    top.set_title(
        "A whole loopback call: the caller interrupts the agent three times", loc="left", fontsize=12
    )

    start = first.caller_start_s
    stop = start + first.barge_in.stop_latency_ms / 1000
    lanes(zoom, start - 1.0, start + 1.0)
    zoom.set_ylim(-0.75, 1.6)
    zoom.axvline(start, color=ACCENT, linestyle=(0, (4, 3)), linewidth=1.4)
    zoom.axvline(stop, color=INK, linestyle=(0, (4, 3)), linewidth=1.4)
    zoom.annotate(
        "",
        xy=(stop, 0.62),
        xytext=(start, 0.62),
        arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 1.3, "shrinkA": 0, "shrinkB": 0},
    )
    zoom.text(stop + 0.04, 0.62, f"{first.barge_in.stop_latency_ms:.0f} ms to stop", va="center", fontsize=11)
    zoom.text(
        start - 0.04, -0.55, "caller starts talking", ha="right", va="center", fontsize=10, color=ACCENT
    )
    zoom.text(stop + 0.04, -0.55, "agent goes quiet", ha="left", va="center", fontsize=10, color=INK)
    zoom.set_xlabel("seconds into the call (zoomed on the first interruption)")
    fig.tight_layout()
    save_png(fig, "bargein-timeline.png")


def latency_png() -> dict:
    opts = {"jitter_ms": "200", "stop_jitter_ms": "60"}
    res = run_calls("examples/scenarios/barge_in.yaml", 20, opts)
    gaps, stops = [], []
    for r in res.results:
        for t in r.metrics.turns:
            if t.turn_gap_ms is not None:
                gaps.append(t.turn_gap_ms)
            if t.barge_in and t.barge_in.stop_latency_ms is not None:
                stops.append(t.barge_in.stop_latency_ms)
    summary = summarize(res.results)
    print("latency run:", len(res.results), "calls,", len(gaps), "gaps,", len(stops), "stops")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title, color in (
        (axes[0], gaps, "Turn gap (caller stops, agent starts)", INK),
        (axes[1], stops, "Barge-in stop latency", ACCENT),
    ):
        lo, hi = min(data), max(data)
        bins = np.arange(np.floor(lo / 20) * 20, hi + 40, 20)
        ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        p50, p95 = np.percentile(data, [50, 95])
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.28)
        for val, label, ls in ((p50, "p50", "-"), (p95, "p95", (0, (4, 3)))):
            ax.axvline(val, color=GRAY, linestyle=ls, linewidth=1.3)
            ax.text(val, top * 1.2, f" {label} {val:.0f} ms", ha="left", va="center", fontsize=10, color=INK)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("milliseconds")
        ax.set_ylabel("turns")
    fig.text(
        0.01,
        0.005,
        f"20 loopback calls, {len(gaps)} turn gaps, {len(stops)} interruptions. "
        "Mock agent with random jitter added, so this shows the charts, not a real agent.",
        fontsize=9,
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_png(fig, "latency-distribution.png")
    return summary


def architecture_png() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fill="white", edge=INK, bold=False):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=edge, lw=1.4)
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold" if bold else "normal",
        )

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                color=INK,
                lw=1.3,
                shrinkA=0,
                shrinkB=0,
            )
        )
        if label:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center", va="bottom", fontsize=9, color=GRAY
            )

    ax.text(0.1, 4.95, "Sending side: what the caller does", fontsize=11, color=GRAY, va="center")
    box(0.1, 3.55, 2.1, 1.0, "Scenario YAML\nturns, pauses,\ninterruptions", fill="#f4f4f4")
    box(3.0, 3.55, 2.1, 1.0, "Simulated\ncallers\n(N at once)", fill=SOFT, edge=ACCENT)
    box(5.9, 3.55, 2.2, 1.0, "Transport", bold=True)
    box(8.9, 3.55, 2.0, 1.0, "Your voice\nagent", fill="#f4f4f4")
    arrow(2.2, 4.05, 3.0, 4.05)
    arrow(5.1, 4.05, 5.9, 4.05, "caller audio")
    arrow(8.1, 4.05, 8.9, 4.05)
    ax.text(
        7.0, 4.72, "loopback (mock agent)  |  LiveKit  |  SIP (stub)", ha="center", fontsize=9.5, color=GRAY
    )

    # the agent's audio comes back
    ax.add_patch(
        FancyArrowPatch((9.9, 3.55), (9.9, 2.65), arrowstyle="-", color=INK, lw=1.3, shrinkA=0, shrinkB=0)
    )
    ax.add_patch(
        FancyArrowPatch((9.9, 2.65), (7.0, 2.65), arrowstyle="-", color=INK, lw=1.3, shrinkA=0, shrinkB=0)
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.0, 2.65),
            (7.0, 3.55),
            arrowstyle="-|>",
            mutation_scale=14,
            color=INK,
            lw=1.3,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(8.45, 2.52, "agent audio", ha="center", va="top", fontsize=9, color=GRAY)

    ax.text(0.1, 2.5, "Scoring side: what the audio says", fontsize=11, color=GRAY, va="center")
    box(0.1, 0.55, 2.1, 1.0, "Both audio\nchannels\n(recorded)", fill="#f4f4f4")
    box(3.0, 0.55, 2.1, 1.0, "VAD\nspeech segments", bold=True)
    box(5.9, 0.55, 2.2, 1.0, "Metrics\nstop latency, gaps,\nfalse interruptions", fill=SOFT, edge=ACCENT)
    box(8.9, 1.1, 2.0, 0.65, "report.json\nreport.md")
    box(8.9, 0.15, 2.0, 0.65, "OpenTelemetry\nLangfuse / Phoenix")
    arrow(2.2, 1.05, 3.0, 1.05)
    arrow(5.1, 1.05, 5.9, 1.05)
    arrow(8.1, 1.2, 8.9, 1.42)
    arrow(8.1, 0.9, 8.9, 0.48)
    line = {"arrowstyle": "-", "color": GRAY, "lw": 1.1, "linestyle": (0, (3, 3)), "shrinkA": 0, "shrinkB": 0}
    ax.add_patch(FancyArrowPatch((6.3, 3.55), (6.3, 1.95), **line))
    ax.add_patch(FancyArrowPatch((6.3, 1.95), (1.15, 1.95), **line))
    ax.add_patch(
        FancyArrowPatch(
            (1.15, 1.95),
            (1.15, 1.55),
            arrowstyle="-|>",
            mutation_scale=14,
            color=GRAY,
            lw=1.1,
            linestyle=(0, (3, 3)),
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(3.7, 2.05, "both audio streams are recorded", fontsize=9, color=GRAY, va="bottom", ha="center")
    fig.tight_layout(pad=0.4)
    save_png(fig, "architecture.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd1 = "qsv bargein --transport loopback"
    terminal_png("terminal-bargein.png", cmd1, run_cli(cmd1.split()[1:] + ["--out", "qsv-report/bargein"]))
    cmd2 = (
        "qsv load --transport loopback --scenario examples/scenarios/barge_in.yaml \\\n"
        "         --callers 20 --agent-opt jitter_ms=200 --agent-opt stop_jitter_ms=60"
    )
    real2 = cmd2.replace("\\\n", " ").split()[1:]
    terminal_png("terminal-load.png", cmd2, run_cli(real2 + ["--out", "qsv-report/load"]))
    timeline_png()
    latency_png()
    architecture_png()
    site_images = ROOT / "site" / "images"
    site_images.mkdir(parents=True, exist_ok=True)
    for png in OUT.glob("*.png"):
        shutil.copy2(png, site_images / png.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
