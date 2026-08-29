import asyncio

from converse_recipes.simulation import (
    SimulationCase,
    SimulationReport,
    _fixture_result,
    evaluate_expectations,
)


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
