import asyncio

import numpy as np

from converse_recipes.simulation import (
    SimulationCase,
    SimulationReport,
    _TextTurnRelay,
    _VoiceTurnRelay,
    _close_voice_turn,
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


def test_voice_relay_keeps_blackholed_silence_running_through_endpoint_commit():
    class Destination:
        async def stream_audio(self, audio, **kwargs):
            self.audio = audio
            self.kwargs = kwargs

    destination = Destination()
    asyncio.run(_close_voice_turn(destination))
    assert len(destination.audio) == 32_000
    assert destination.kwargs == {"sr": 16_000, "chunk_ms": 100, "realtime": True}


def test_voice_relay_waits_for_client_tool_continuation_before_crosspipe():
    async def run():
        class Destination:
            def __init__(self):
                self.calls = []

            async def stream_audio(self, audio, **kwargs):
                self.calls.append((audio.copy(), kwargs))

        destination = Destination()
        relay = _VoiceTurnRelay(destination, settle_s=0)
        relay.audio(np.ones(800, dtype=np.float32))
        relay.done()
        relay.working(True)
        await asyncio.sleep(0)
        assert destination.calls == []
        relay.audio(np.full(400, 2.0, dtype=np.float32))
        relay.working(False)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await relay.close()
        return destination.calls

    calls = asyncio.run(run())
    assert [len(audio) for audio, _ in calls] == [1_200, 32_000]
    assert np.all(calls[0][0][:800] == 1.0)
    assert np.all(calls[0][0][800:] == 2.0)


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
