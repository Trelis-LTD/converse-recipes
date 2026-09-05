from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np

from dialt_recipes import twilio as bridge
from dialt_recipes.telephony_audio import (
    TelephonyAudioBridge,
    decode_mulaw,
    mulaw_8k_to_pcm16_16k,
    pcm16_16k_to_mulaw_8k,
)

SETTINGS = bridge.TwilioBridgeSettings(
    dialt_api_key="ck_test", twilio_auth_token="token", public_base_url="https://voice.example.com/")


def test_mulaw_round_trip_preserves_a_tone() -> None:
    t = np.arange(1600) / 8_000
    tone = (np.sin(2 * np.pi * 440 * t) * 12_000).astype("<i2")
    pcm16k = mulaw_8k_to_pcm16_16k(pcm16_16k_to_mulaw_8k(
        np.repeat(tone, 2).astype("<i2").tobytes()))
    back = np.frombuffer(pcm16k, dtype="<i2")
    assert 1000 < back.size < 3400
    assert 6_000 < np.abs(back).max() < 16_000


def test_streamed_outbound_matches_one_shot_across_odd_byte_chunks() -> None:
    pcm = (np.random.default_rng(1).normal(0, 3000, 6400)).astype("<i2").tobytes()
    one_shot = pcm16_16k_to_mulaw_8k(pcm)
    streamed = TelephonyAudioBridge()
    parts = [pcm[:333], pcm[333:2000], pcm[2000:]]
    out = b"".join(streamed.dialt_to_twilio(part) for part in parts)
    out += streamed.dialt_to_twilio(b"", final=True)
    assert abs(len(out) - len(one_shot)) <= 2
    assert decode_mulaw(out[100:200]).size == 100


def test_settings_urls() -> None:
    assert SETTINGS.http_url("/voice") == "https://voice.example.com/voice"
    assert SETTINGS.websocket_url("/media") == "wss://voice.example.com/media"


def test_connect_stream_twiml_carries_prelude_and_stream_urls() -> None:
    twiml = bridge.connect_stream_twiml(SETTINGS, prelude="<Play>https://x/ring.wav</Play>")
    assert twiml.index("<Play>") < twiml.index("<Connect>")
    assert 'url="wss://voice.example.com/media"' in twiml
    assert 'statusCallback="https://voice.example.com/stream-status"' in twiml
    assert twiml.endswith("</Connect><Hangup /></Response>")


def test_playback_ledger_tracks_unacknowledged_audio() -> None:
    ledger = bridge.PlaybackLedger()
    first = ledger.add(100)
    ledger.add(60)
    assert ledger.pending_ms() == 160
    ledger.acknowledge(first)
    assert ledger.pending_ms() == 60
    assert ledger.interruption(hard_clear=False) == (60, 0)
    assert ledger.interruption(hard_clear=True) == (0, 60)
    assert ledger.empty()


def _fake_socket(release_after_media: int):
    released = asyncio.Event()

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, message) -> None:
            self.sent.append(message)
            if sum(1 for m in self.sent if m.get("event") == "media") >= release_after_media:
                released.set()

        async def iter_text(self):
            acknowledged: set[str] = set()
            while True:
                pending = next((m for m in self.sent if m.get("event") == "mark"
                                and m["mark"]["name"] not in acknowledged), None)
                if pending is not None:
                    acknowledged.add(pending["mark"]["name"])
                    yield json.dumps({"event": "mark", "mark": {"name": pending["mark"]["name"]}})
                    continue
                await asyncio.sleep(0)

    return FakeWebSocket(), released


def _fake_connect(events_factory):
    class FakeSession:
        closed = False

        async def close(self):
            self.closed = True

        async def send_client_event(self, *args, **kwargs):
            return None

        async def send_tool_result(self, *args, **kwargs):
            return None

        def events(self):
            return events_factory(self)

    class FakeConnection:
        def __init__(self):
            self.session = FakeSession()

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, *args):
            return None

    holder = {}

    async def connect(*args, **kwargs):
        holder["connection"] = FakeConnection()
        return holder["connection"]

    return connect, holder


def test_run_call_bridge_paces_frames_and_marks(monkeypatch) -> None:
    websocket, released = _fake_socket(release_after_media=4)

    async def events(session):
        yield SimpleNamespace(type="audio", t_ms=20, data={}, audio=np.zeros(3200, dtype=np.float32))
        yield SimpleNamespace(type="done", t_ms=40, data={"turn_id": "turn-1"}, audio=None)
        await released.wait()

    connect, _ = _fake_connect(events)
    monkeypatch.setattr(bridge.DialtSession, "connect", connect)
    seen: list[str] = []

    async def on_event(event):
        seen.append(event.type)

    async def execute_tool(name, args):
        return {}

    async def run():
        await asyncio.wait_for(bridge.run_call_bridge(
            websocket, "MZ", "CA-paced", settings=SETTINGS, mode=bridge.DialtMode(),
            hooks=bridge.BridgeHooks(execute_tool=execute_tool, on_event=on_event)), timeout=5)

    asyncio.run(run())
    media = [m for m in websocket.sent if m.get("event") == "media"]
    marks = [m for m in websocket.sent if m.get("event") == "mark"]
    assert 5 <= len(media) <= 10          # 200 ms of audio: five frames before the final flush
    assert 1 <= len(marks) <= 3           # a mark per 100 ms, not per frame
    assert all(len(m["media"]["payload"]) <= 216 for m in media)
    assert seen == ["audio", "done"]


def test_host_end_call_closes_the_session_and_tool_failures_are_results(monkeypatch) -> None:
    websocket, _released = _fake_socket(release_after_media=10**6)
    end_call = asyncio.Event()
    results: list[tuple] = []

    async def events(session):
        yield SimpleNamespace(type="tool_call", t_ms=10, data={"id": "t1", "name": "boom", "args": {}}, audio=None)
        while not session.closed:
            await asyncio.sleep(0.01)

    connect, holder = _fake_connect(events)
    monkeypatch.setattr(bridge.DialtSession, "connect", connect)

    async def execute_tool(name, args):
        end_call.set()
        raise RuntimeError("no such tool")

    async def run():
        session_cls = None
        await asyncio.wait_for(bridge.run_call_bridge(
            websocket, "MZ", "CA-end", settings=SETTINGS, mode=bridge.DialtMode(),
            hooks=bridge.BridgeHooks(execute_tool=execute_tool, end_call=end_call)), timeout=5)

    asyncio.run(run())
    assert holder["connection"].session.closed is True


def test_receive_stream_start_rejects_bad_and_stopped_streams() -> None:
    class Socket:
        def __init__(self, messages):
            self.messages = list(messages)

        async def receive_text(self):
            return self.messages.pop(0)

    start = json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}})
    assert asyncio.run(bridge.receive_stream_start(Socket([json.dumps({"event": "connected"}), start]))) == {
        "streamSid": "MZ1", "callSid": "CA1"}
    try:
        asyncio.run(bridge.receive_stream_start(Socket([json.dumps({"event": "stop"})])))
    except ConnectionError:
        pass
    else:
        raise AssertionError("stop before start must raise ConnectionError")
    try:
        asyncio.run(bridge.receive_stream_start(Socket([json.dumps({"event": "start", "start": {}})])))
    except ValueError:
        pass
    else:
        raise AssertionError("a start without streamSid must raise ValueError")
