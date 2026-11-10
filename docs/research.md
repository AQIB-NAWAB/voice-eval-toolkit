# Research: measuring what a voice agent caller experiences

Background for `voice-eval-toolkit`. Star counts and project details were collected on 10 November 2026 and will drift. Where a project's repository could not be confirmed, it is cited by its package or site only.

## The problem

A cascaded voice agent (STT, then an LLM, then TTS) can look fine on every dashboard and still feel wrong on the phone. Three things decide that feeling, and they are all about timing on the audio line, not about API latency:

1. **Does it stop when I talk over it?** Barge-in is the hardest timing problem in a cascaded agent. The agent has to notice speech, decide it is a real interruption and not a cough or "mm-hm", flush audio that is already queued in TTS and in the transport buffer, and stay quiet. Every one of those steps adds to what the caller hears as "it kept talking for a second after I interrupted".
2. **Does it jump in too early?** Aggressive end-of-turn detection shortens response time but cuts people off mid-sentence, especially when they pause to think. A system tuned only for low latency will score well on turn latency and badly on the experience.
3. **How long do I wait?** The gap between the caller finishing and the agent's first audio is what people perceive as responsiveness. It includes endpointing, STT finalisation, LLM first token, TTS first byte and the network, and it varies call to call, so the tail (p95, p99) matters more than the mean.

None of these show up in a stage timer. They show up in the audio. A useful measurement therefore listens to both channels of the call, finds speech with a VAD, and reasons about when each side started and stopped.

A second problem sits on top: one clean call proves little. Voice agents tend to break under concurrency (shared GPU or API rate limits, slow worker start-up, noisy neighbours), so a useful harness also runs many scripted calls at once and reports distributions.

## What already exists

### Frameworks that emit their own telemetry

- **[Pipecat](https://github.com/pipecat-ai/pipecat)** (about 17.2k stars, BSD-2-Clause). Built-in metrics for TTFB, time to first audio, time to first answer token and processing time, plus observers for user-to-bot latency and turn tracking, and OpenTelemetry traces covering conversation, turn, STT, LLM and TTS spans. Traces can go to SigNoz, Langfuse, Phoenix or any OTLP backend.
- **[LiveKit Agents](https://github.com/livekit/agents)** (about 15.2k stars, Apache-2.0). OpenTelemetry spans for STT, end of turn, LLM, TTS and tools; end-to-end turn latency, STT delay, end-of-turn delay, LLM time to first token and TTS time to first byte; export to Langfuse, Jaeger, Grafana Tempo or Honeycomb through OTLP.

Both report what the pipeline knows about itself. Neither independently verifies what a caller heard, and neither plays an adversarial caller against the agent.

### Open-source observability and evaluation

- **[Langfuse](https://github.com/langfuse/langfuse)** (about 37k stars). Tracing, sessions, prompt management and evals with an OTLP ingest endpoint. It is not audio-native: no barge-in stop time and no telephony WER out of the box.
- **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** (about 12k stars). OTEL-based observability and evaluation with documented Pipecat and LiveKit Agents tracing guides.
- **[OpenLLMetry](https://github.com/traceloop/openllmetry)** (about 7.5k stars). OTel instrumentation for LLM SDKs, including some speech APIs. Provider-SDK level, not a turn-taking layer.
- **[LangWatch Scenario](https://github.com/langwatch/scenario)** (about 1k stars, Apache-2.0). Agent testing with simulated users, including voice simulations that can inject interruptions and background noise.
- **[Hugging Face Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)**. The reference for offline ASR WER. It is clean, offline audio, not live telephony.

Other voice stacks (TEN framework, Fonoster, Vocode, Bolna, jambonz) add their own telemetry, mostly platform or call-path level rather than per-turn caller-experience measurement.

### Open-source harnesses closest to this one

This is the part that matters most for positioning, so it is stated carefully.

- **[LiveKit EOT Bench](https://github.com/livekit/eot-bench)** (about 73 stars). A reproducible benchmark of end-of-turn detectors: false cut-off rate against added latency, across 14 languages. It evaluates turn-detection models in isolation, not a full duplex call, and it does not measure barge-in stop time.
- **[voicebench](https://pypi.org/project/voicebench/)** (PyPI, Apache-2.0, v0.2.0). An audio-level harness for response latency, time to first audio, barge-in stop time, false barge-ins, talk-over, overlap and WER. It works as a black box over WebSocket with experimental LiveKit and Pipecat adapters; SIP and plain WebRTC adapters are listed as roadmap. This overlaps the barge-in half of this project, and anyone choosing between them should look at voicebench too.
- **[rtpbench](https://pypi.org/project/rtpbench/)** (PyPI, Apache-2.0, v0.4.0). Wire-level SIP and RTP tests for turn latency, barge-in and SIP conformance, including impaired networks. Early stage and Linux/Docker oriented.
- **[telnyx-onset](https://github.com/rmsdev0/telnyx-onset)** (about 5 stars). A Telnyx-specific SIP barge-in latency harness comparing VAD-triggered with transcript-triggered interruption.
- **[jambonz-livekit-load-testing-benchmarks](https://github.com/jambonz/jambonz-livekit-load-testing-benchmarks)**. Infrastructure load-test methodology for jambonz versus LiveKit. Capacity testing rather than a caller eval runner.

### Commercial voice testing and observability

Hamming, Coval, Cekura, Roark, Bluejay and Canonical sell some mix of simulated calls at scale, regression suites, audio-native scoring, production monitoring and dashboards, with OpenTelemetry in several of them. Bluejay also published MISTT, a multi-condition STT benchmark. These products establish that latency dashboards and alert thresholds are table stakes. They also show that simulation and audio-native scoring are where buyers see value, and that this space is mostly closed source.

## The gap this project aims at

The pieces exist separately. What is thin in open source is a small, readable, Python toolkit that:

1. treats the agent as a **black box over the transports people actually use** (LiveKit now, SIP next),
2. **scripts adversarial callers declaratively**: interruptions at set offsets after the agent starts speaking, mid-sentence hesitations that tempt aggressive endpointing, optional reference transcripts,
3. **scores from audio alone** with precisely defined metrics (stop latency, false interruptions, turn gap, overlap), so numbers are comparable across stacks,
4. **runs the same scripts concurrently** and reports percentiles, with pass/fail assertions suitable for CI,
5. **exports OpenTelemetry** to the backends people already run (Langfuse, Phoenix) instead of building another dashboard,
6. and **runs fully offline in CI** through a loopback mock agent with known behavior, so the harness itself is tested.

That is a modest claim. It is not a new benchmark and it does not replace voicebench or the commercial tools. It is a clean, inspectable harness that covers barge-in plus load in one package and is easy to extend.

## Design choices that follow from this

- **Both channels, one VAD, one definition.** Metrics come from segments detected on the caller and agent recordings. The caller script's own timing is used only to map segments to turns. Energy VAD is the default because it needs no model and is deterministic; `webrtcvad` is available as an option. The limits of an energy VAD (it hears any loud sound, including TTS echo and noise) are documented in `metrics.md`.
- **Frame-synchronous driver.** Each tick sends one caller frame and reads one agent frame. This makes caller behavior reactive (it can start an interruption relative to when the agent starts replying) and makes simulated transports deterministic and fast.
- **Interruption offsets are relative to the agent's reply onset.** An absolute timestamp would not mean the same thing across agents with different latency. "Interrupt 900 ms into the reply" does.
- **Hesitations as a first-class turn feature.** Cut-offs mostly happen when a caller pauses. Making that a scenario field makes the failure reproducible.
- **WER depends on the agent's transcript.** WER is only meaningful against the transcript the agent produced for the caller. The toolkit scores what the transport reports and says so, instead of running its own STT and calling that the agent's WER.
- **Honest scope.** Telephony impairments (loss, jitter, noise), a SIP transport and a semantic turn-detection comparison are roadmap items, not features.

## Sources

Repository and package pages consulted on 10 November 2026: Pipecat and LiveKit Agents documentation (metrics and OpenTelemetry tracing), Langfuse and Phoenix OpenTelemetry documentation, the LiveKit EOT Bench repository and leaderboard, the voicebench and rtpbench PyPI pages, the telnyx-onset and jambonz load-testing repositories, and vendor sites for Hamming, Coval, Cekura, Roark, Bluejay and Canonical.
