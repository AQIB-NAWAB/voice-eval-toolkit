# Metric definitions

All metrics come from the two recorded channels of a call: **caller** (what the harness sent) and **agent** (what came back). Both channels are run through the same VAD to produce speech segments. Times are in seconds from the start of the recording in the code, and in milliseconds in reports.

## Vocabulary

- **Segment**: a stretch of detected speech. It starts at the first speech frame of a run that passed the onset filter and ends at the end of the last speech frame, so hangover never makes it longer.
- **Turn**: one scripted caller utterance. The harness knows when it scheduled each turn. The *measured* extent of a turn is the caller VAD segments that begin inside the scheduled window, from the first start to the last end. This keeps the metrics tied to the audio while tolerating a frame of VAD slop.
- **Agent response to turn N**: the first agent segment that starts after the caller began turn N and before the caller began turn N+1.

## VAD

Default is `energy`: a frame is speech if its RMS level is above `max(threshold_db, noise_floor + 15 dB)`, with `threshold_db = -45 dBFS` and a noise floor tracked on non-speech frames. Frames are 20 ms. A segment needs 40 ms of consecutive speech frames to start, and ends after 200 ms of non-speech (a hangover that bridges short gaps between syllables). `--vad webrtc` swaps in `webrtcvad` for the frame decision (needs the `vad` extra).

Resolution is one frame, so every timing metric has about plus or minus 20 ms of quantisation.

## Turn gap

`turn_gap_ms = agent_response.start - caller_turn.end`

How long the caller waits in silence for the agent's first audio. A negative value means the agent started while the caller was still talking. Run-level turn-gap percentiles use only non-negative gaps; negative ones are counted as false interruptions instead. Included for every turn that got a response, including responses to interruptions.

## False interruption (cut-off)

A turn counts as cut off when the agent's response starts **before the caller's turn ended plus `min_silence_gap_ms`** (default 300 ms, `--min-gap-ms`). This catches two cases:

- the agent starts talking while the caller is still speaking (including after a mid-turn hesitation), and
- the agent starts talking unnaturally soon after the caller stops.

The second case is a judgement call, which is why the threshold is configurable. Set `--min-gap-ms 0` to count only true overlaps at the start of a response.

An agent segment that was already playing when the caller started is not a false interruption. That is the barge-in case below.

## Barge-in stop latency

Only for turns the script marked as interruptions (`interrupt_after_ms`) where the harness actually started the turn while waiting on the agent.

- The **target** is the agent segment that was playing when the caller's turn began (it started before the caller onset and ended after it). If there is none, the agent was already silent: the interruption is not scored (`agent_was_speaking = false`).
- If the target ends before the caller's turn ends, the agent **yielded** and
  `stop_latency_ms = target.end - caller_turn.start`.
- If the target is still playing when the caller finishes, the agent **did not yield**; `stop_latency_ms` is empty and `talk_over_ms` records how long the two overlapped.
- A stop slower than `stop_timeout_ms` (default 2000 ms, `--stop-timeout-ms`) counts as not yielded.

Run-level stop latency percentiles use only turns where the agent yielded, and the yield rate is reported next to them so a fast p95 cannot hide a low yield rate.

Known limits:

- **Natural end versus barge-in stop.** From audio alone, an agent that finishes its sentence just after the caller starts looks the same as one that stopped on purpose. Place interruptions well inside the agent's reply (the offsets in the examples assume replies of several seconds). To reduce one common false positive, an agent segment that ran through an earlier interruption without yielding is treated as not yielding on later ones too.
- **Echo.** If the transport leaks the caller's audio back into the agent channel, the energy VAD will hear it as agent speech. Loopback and LiveKit have no echo path. SIP and telephony-bridge tests need care here.
- **Partial yields.** An agent that stops, then resumes talking while the caller is still speaking, shows up as a false interruption on that turn because the resumption is a new agent segment inside the caller's turn.

## Overlap

`overlap_ms` per turn is the time agent speech coincided with caller speech inside the turn window. Run level `total_overlap_ms` per call sums the intersection of all caller and agent segments over the whole call. Some overlap is normal in a successful barge-in (it is at least the stop latency).

## Time to first audio

`time_to_first_audio_ms = first agent segment start` measured from the start of the call, not from the end of the caller's first utterance. For agents that greet, it is the greeting delay. For agents that wait, it is dominated by how long the first caller turn takes, so use turn gap for first-response latency.

## Word error rate

For turns with a reference (`expected_transcript`, or `say` if absent), the hypothesis is the transcript text the agent reported during that turn (events whose timestamp falls between the start of that caller turn and the start of the next). Both sides are lowercased and stripped of punctuation (apostrophes kept).

`WER = (substitutions + deletions + insertions) / reference words`

The run-level figure is **corpus WER**: total errors over total reference words, so long utterances weigh more. `jiwer` is used when installed, with an equivalent built-in scorer otherwise.

WER needs the agent to report what it heard. The loopback agent *simulates* that with a configurable error rate (so the number only validates plumbing). The LiveKit transport collects transcripts of the caller's track published on the `lk.transcription` text stream, which LiveKit Agents does by default; agents that do not publish it produce no WER.

## Percentiles

Linear-interpolated percentiles (numpy default) over all scored turns in the run: p50, p95, p99, plus mean, min and max. With few samples, p99 is close to the max, so read it as a sample maximum until you have hundreds of turns.

## Assertions

Scenarios can carry limits that the CLI checks and turns into an exit code:

| Key | Checks |
| --- | --- |
| `max_bargein_stop_p95_ms` | p95 barge-in stop latency |
| `max_turn_gap_p95_ms` | p95 turn gap |
| `max_false_interruptions` | total count |
| `max_wer` | corpus WER, as a fraction |
| `min_success_rate` | share of calls that completed |

A ceiling check with no data (for example no interruptions landed) fails, because "no measurement" should not read as "within budget".
