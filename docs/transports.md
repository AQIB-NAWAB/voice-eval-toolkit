# Transports

A transport carries audio between the simulated caller and the agent. Pick one with `--transport`.

| Name | Status | Needs |
| --- | --- | --- |
| `loopback` | Working. In-process mock agent, no network. | nothing |
| `livekit` | Working. Real WebRTC through a LiveKit server. | `livekit` extra, server URL and API key |
| `sip` | Stub. Raises `NotImplementedError` with guidance. | not implemented |

## loopback

The mock agent is configured with `--agent-opt key=value` (repeatable) or the `agent:` block in a scenario. Options:

| Option | Default | Effect |
| --- | --- | --- |
| `endpointing_ms` | 500 | silence that ends the caller's turn |
| `response_delay_ms` | 300 | wait after end of turn before speaking |
| `jitter_ms` | 0 | seeded random plus or minus on the response delay |
| `response_ms` | `2500,2000,3000` | reply durations, cycled |
| `greeting_ms` | 0 | greet first for this long (0 disables) |
| `greeting_delay_ms` | 200 | delay before the greeting |
| `barge_in` | true | stop when the caller speaks |
| `barge_in_min_speech_ms` | 120 | caller speech needed to trigger a stop |
| `stop_latency_ms` | 80 | audio that keeps playing after the stop decision |
| `stop_jitter_ms` | 0 | seeded random plus or minus on `stop_latency_ms`, so stop latencies vary between calls |
| `stt_word_error_rate` | 0 | simulated transcript errors |
| `vad_threshold_db` | -45 | the mock's own VAD floor |

A reply that gets interrupted before it starts is cancelled, which matches how real agents behave when the caller resumes quickly. The numbers it produces describe the mock, not any real agent.

## livekit

```bash
pip install -e ".[livekit]"
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
qsv bargein --transport livekit --livekit-agent-name my-agent
```

What happens per call:

1. A token is minted with `livekit-api` (`AccessToken` with `VideoGrants(room_join=True, room=...)`). With `--livekit-agent-name`, the token carries a `RoomConfiguration` with a `RoomAgentDispatch` so that agent is dispatched when the caller joins. Without it, automatic dispatch applies.
2. The caller joins the room `qsv-<call id>` with `livekit.rtc`, publishes an `AudioSource` as a microphone track and waits (up to `--livekit-join-timeout`, 20 s) for the agent's audio track.
3. Each tick, caller audio goes in through `AudioSource.capture_frame`. Agent audio is read from `AudioStream` and placed on a timeline at its arrival time.
4. Caller transcripts published on the `lk.transcription` text stream (final ones, attributed to the caller's track) are collected for WER.
5. The call ends by disconnecting.

Notes:

- The first turn on a fresh connection can be slower than later turns (subscription and jitter-buffer warm-up on both sides). Give the scenario a longer `initial_silence_ms`, or have the agent greet and use `wait_for_greeting_ms`.
- Reported timings include the transport path (network, SFU, jitter buffers). That is what a caller experiences, but it means numbers from a LAN dev server and from a cloud region will differ.
- Run the caller close to the agent, or measure and report the round-trip separately, if you want to isolate agent latency.

### Trying it with no agent of your own

```bash
livekit-server --dev                                    # keys devkey / secret, ws://127.0.0.1:7880
python examples/livekit_mock_agent.py &                 # joins rooms named qsv-*
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
qsv bargein --transport livekit
```

`examples/livekit_mock_agent.py` reuses the loopback agent as a LiveKit participant. It is a development aid. The optional test `QSV_LIVEKIT_INTEGRATION=1 pytest tests/test_livekit_integration.py` does the same thing automatically.

## sip (stub)

`qsv.transports.sip.SipTransport` fixes the interface and raises `NotImplementedError` with this guidance. Two routes are planned:

1. **LiveKit SIP outbound.** If the agent under test answers a phone number, create a room, have LiveKit's SIP service dial that number into the room with `LiveKitAPI().sip.create_sip_participant(...)` (outbound trunk id, `sip_call_to`, `wait_until_answered`), and let the caller join the same room through the existing LiveKit transport. The SIP participant's track carries the agent's audio. New code: the dial step and hang-up handling. This is the shortest path and keeps the audio plumbing already in place.
2. **PJSUA2.** Register an account, place the call, and bridge an `AudioMediaPort` to `send_frame` and `recv_frame`. This gives wire-level control (codec choice, DTMF, RTP statistics) and is the right base for impairment testing with `tc netem`. Expect G.711 at 8 kHz, so scenarios should use `sample_rate: 8000`.

Either way the contract is the same: `connect` places the call, `send_frame` and `recv_frame` exchange one frame per tick in real time, `close` hangs up. Be careful about echo on telephony bridges, since the energy VAD will treat leaked caller audio on the agent channel as agent speech.

## Writing a transport

Subclass `qsv.transports.base.Transport`, implement `connect`, `send_frame`, `recv_frame` and `close`, set `realtime = True` unless you simulate time, and register a name in `qsv.transports.get_transport_factory`. If your transport can place incoming audio on a timeline better than one-frame-per-tick (as LiveKit does), also implement `agent_recording` and set `live_lag_ms` to how far the live view trails it.

If it needs an optional dependency, import it inside `connect` and raise an `ImportError` that names the extra to install. Keep one fresh transport instance per call.
