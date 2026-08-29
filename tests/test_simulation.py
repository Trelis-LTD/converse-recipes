import asyncio

import numpy as np
import pytest

from converse_recipes.simulation import (
    SimulationCase,
    SimulationReport,
    _TextTurnRelay,
    _VoiceTurnRelay,
    _close_voice_turn,
    _voice_tail_s,
    _fixture_result,
    evaluate_expectations,
)


def test_text_relay_waits_for_tool_work_to_settle():
    async def run():
        forwarded = []

        async def forward(text):
            forwarded.append(text)

        relay = _TextTurnRelay(forward, settle_s=0)
        relay.utterance("I'll check that now")
        relay.working(True)
        relay.done()
        await asyncio.sleep(0)
        assert forwarded == []
        relay.working(False)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await relay.close()
        return forwarded

    assert asyncio.run(run()) == ["I'll check that now"]


def test_voice_relay_silence_stops_when_receiving_turn_commits():
    class Destination:
        async def stream_audio(self, audio, **kwargs):
            self.calls.append((audio.copy(), kwargs))
            if len(self.calls) == 3:
                committed.set()

    committed = asyncio.Event()
    destination = Destination()
    destination.calls = []
    asyncio.run(_close_voice_turn(destination, committed))
    assert [len(audio) for audio, _ in destination.calls] == [1_600, 1_600, 1_600]
    assert all(kwargs == {"sr": 16_000, "chunk_ms": 100, "realtime": True}
               for _, kwargs in destination.calls)


def test_voice_relay_silence_covers_ink_hard_turn_end_fallback():
    class Destination:
        async def stream_audio(self, audio, **kwargs):
            self.calls.append((audio.copy(), kwargs))

    destination = Destination()
    destination.calls = []
    asyncio.run(_close_voice_turn(destination))
    assert sum(len(audio) for audio, _ in destination.calls) == round(16_000 * _voice_tail_s())


def test_voice_tail_treats_empty_timeout_as_broker_default(monkeypatch):
    monkeypatch.delenv("CARTESIA_TURN_END_TIMEOUT_MS", raising=False)
    assert _voice_tail_s() == 6.1
    for value in ("", "   "):
        monkeypatch.setenv("CARTESIA_TURN_END_TIMEOUT_MS", value)
        assert _voice_tail_s() == 6.1
    monkeypatch.setenv("CARTESIA_TURN_END_TIMEOUT_MS", "640")
    assert _voice_tail_s() == pytest.approx(1.14)


def test_voice_tail_ignores_malformed_timeout(monkeypatch):
    for value in ("5600ms", "5.6s", "auto"):
        monkeypatch.setenv("CARTESIA_TURN_END_TIMEOUT_MS", value)
        assert _voice_tail_s() == 6.1


def test_voice_tail_env_is_read_at_runtime(monkeypatch):
    """The CLI loads .env after import, so the tail must not be frozen at import time."""
    monkeypatch.setenv("CARTESIA_TURN_END_TIMEOUT_MS", "100")

    class Destination:
        async def stream_audio(self, audio, **kwargs):
            self.calls.append(audio.copy())

    destination = Destination()
    destination.calls = []
    asyncio.run(_close_voice_turn(destination))
    assert sum(len(audio) for audio in destination.calls) == round(16_000 * 0.6)


def test_voice_relay_streams_live_and_defers_commit_through_tool_work():
    async def run():
        class Destination:
            def __init__(self):
                self.calls = []

            async def send_audio(self, audio):
                self.calls.append(("audio", audio.copy()))

            async def stream_audio(self, audio, **kwargs):
                self.calls.append(("tail", audio.copy()))
                relay.input_committed()

        destination = Destination()
        relay = _VoiceTurnRelay(destination, settle_s=0)
        await relay.audio(np.ones(800, dtype=np.float32))
        relay.done()
        await asyncio.sleep(0)
        relay.working(True)
        await asyncio.sleep(0)
        assert [(kind, len(audio)) for kind, audio in destination.calls] == [("audio", 800)]

        await relay.audio(np.full(400, 2.0, dtype=np.float32))
        relay.done()
        relay.working(False)
        assert relay._finish_task is not None
        await asyncio.wait_for(relay._finish_task, 1)
        await relay.close()
        return destination.calls

    calls = asyncio.run(run())
    assert [(kind, len(audio)) for kind, audio in calls] == [
        ("audio", 800), ("audio", 400), ("tail", 1_600),
    ]
    assert np.all(calls[0][1] == 1.0)
    assert np.all(calls[1][1] == 2.0)


def test_voice_relay_serializes_tail_stop_before_continuation_audio():
    async def run():
        class Destination:
            def __init__(self):
                self.calls = []
                self.writing = False
                self.tail_started = asyncio.Event()
                self.release_tail = asyncio.Event()

            async def send_audio(self, audio):
                assert not self.writing
                self.writing = True
                self.calls.append(("audio", audio.copy()))
                self.writing = False

            async def stream_audio(self, audio, **_kwargs):
                assert not self.writing
                self.writing = True
                self.tail_started.set()
                await self.release_tail.wait()
                self.calls.append(("tail", audio.copy()))
                self.writing = False

        destination = Destination()
        relay = _VoiceTurnRelay(destination, settle_s=0)
        await relay.audio(np.ones(800, dtype=np.float32))
        relay.done()
        await asyncio.wait_for(destination.tail_started.wait(), 1)

        continuation = asyncio.create_task(
            relay.audio(np.full(400, 2.0, dtype=np.float32)))
        await asyncio.sleep(0)
        assert not continuation.done()
        destination.release_tail.set()
        await continuation
        await relay.close()
        return destination.calls

    calls = asyncio.run(run())
    assert [(kind, len(audio)) for kind, audio in calls] == [
        ("audio", 800), ("tail", 1_600), ("audio", 400),
    ]


def test_relay_background_failures_surface_immediately():
    async def run():
        errors = []

        async def fail_text(_text):
            raise RuntimeError("text relay failed")

        text_relay = _TextTurnRelay(fail_text, settle_s=0, on_error=errors.append)
        text_relay.utterance("hello")
        text_relay.done()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        class Destination:
            async def send_audio(self, _audio):
                pass

            async def stream_audio(self, _audio, **_kwargs):
                raise RuntimeError("voice relay failed")

        voice_relay = _VoiceTurnRelay(Destination(), settle_s=0, on_error=errors.append)
        await voice_relay.audio(np.ones(800, dtype=np.float32))
        voice_relay.done()
        assert voice_relay._finish_task is not None
        failures = await asyncio.gather(
            voice_relay._finish_task, return_exceptions=True)
        await asyncio.sleep(0)

        assert [str(error) for error in errors] == [
            "text relay failed", "voice relay failed",
        ]
        assert [str(error) for error in failures] == ["voice relay failed"]
        await text_relay.close()
        await voice_relay.close()

    asyncio.run(run())


def test_fixed_and_callback_fixtures_share_the_tool_result_contract():
    async def run():
        fixed = await _fixture_result({"lookup": {"result": {"slot": "3:30"}}}, "lookup", {})
        callback = await _fixture_result({"lookup": lambda args: {"slot": args["wanted"]}},
                                         "lookup", {"wanted": "4:00"})
        missing = await _fixture_result({}, "delete", {})
        return fixed, callback, missing

    fixed, callback, missing = asyncio.run(run())
    assert fixed == ({"slot": "3:30"}, "succeeded", True)
    assert callback == ({"slot": "4:00"}, "succeeded", True)
    assert missing[1:] == ("failed", False)


def test_case_parses_guardrails_and_keeps_simulator_toolless():
    case = SimulationCase.from_dict({
        "name": "booking", "target_instructions": "Book safely.",
        "simulator_instructions": "Need an afternoon slot.", "starter": "Hello",
        "target_tools": [{"name": "lookup"}], "fixtures": {"lookup": {"ok": True}},
        "expected": {"assistant_contains": ["3:30"], "tools_called": ["lookup"]},
        "max_turns": 4, "timeout_s": 90, "silence_s": 20,
    })
    assert case.max_turns == 4 and case.timeout_s == 90 and case.silence_s == 20
    assert case.target_tools == ({"name": "lookup"},)
    assert case.expected == {
        "assistant_contains": ("3:30",), "tools_called": ("lookup",),
    }


def test_expectations_score_committed_output_and_target_tool_calls():
    case = SimulationCase.from_dict({
        "name": "booking", "target_instructions": "Book safely.",
        "simulator_instructions": "Need an afternoon slot.", "starter": "Hello",
        "expected": {
            "assistant_contains": ["3:30", "confirmation"],
            "tools_called": ["lookup", "book"],
        },
    })
    report = SimulationReport(
        "booking", "text", "target", "simulator",
        transcript=[{"role": "assistant", "text": "3:30 is available; confirmation PT-1."}],
        events=[
            {"side": "target", "type": "tool_call", "name": "lookup"},
            {"side": "target", "type": "tool_call", "name": "book"},
        ],
        termination_reason="max_turns",
    )
    report.check_results = evaluate_expectations(case, report)
    assert report.passed is True
    report.termination_reason = "expectations_met"
    assert report.passed is True
    report.check_results[0]["pass"] = False
    assert report.passed is False
