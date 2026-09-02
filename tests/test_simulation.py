import asyncio
import json
from pathlib import Path

import pytest

from dialt_recipes.cli import collect_cases, push
from dialt_recipes.simulation import (
    SimulationCase,
    assistant_turns,
    SimulationReport,
    _fixture_result,
    evaluate_checks,
    run_simulation,
)

SAMPLE = Path(__file__).resolve().parents[1] / "examples/simulations/appointment_booking.json"


def test_case_uses_the_hosted_document_shape():
    case = SimulationCase.from_dict(json.loads(SAMPLE.read_text()))
    assert case.name.startswith("physiotherapy")
    assert case.target_tools[0]["name"] == "check_availability"
    assert case.simulator_instructions.startswith("You are Maya")
    assert case.max_turns == 10 and case.timeout_s == 240 and case.silence_s == 35
    assert [check["type"] for check in case.checks] == [
        "tool_called", "tool_called", "contains", "judge"]

    with pytest.raises(ValueError, match="target.instructions"):
        SimulationCase.from_dict({"name": "old", "starter": "hi", "target_instructions": "x"})
    with pytest.raises(ValueError, match="unsupported check type"):
        SimulationCase.from_dict({"name": "bad", "starter": "hi", "checks": [{"type": "vibes", "value": 1}]})
    with pytest.raises(ValueError, match="needs a value or criterion"):
        SimulationCase.from_dict({"name": "bad", "starter": "hi", "checks": [{"type": "contains"}]})
    with pytest.raises(ValueError, match="starter must contain"):
        SimulationCase.from_dict({"name": "no starter"})
    assert case.target_options["end_call"] is True
    assert "end_call" not in SimulationCase.from_dict({"name": "n", "starter": "hi"}).target_options
    with pytest.raises(ValueError, match="target.end_call must be true or false"):
        SimulationCase.from_dict({"name": "n", "starter": "hi", "target": {"end_call": "no"}})


class _FakeSession:
    def __init__(self, events):
        self._events = events
        self.sent = []

    async def events(self):
        for event in self._events:
            yield event
        await asyncio.sleep(30)

    async def send_text(self, text):
        self.sent.append(text)

    async def send_tool_result(self, *_args, **_kwargs):
        pass

    async def close(self):
        pass


def test_end_call_is_a_recorded_tool_call_and_the_simulator_always_has_it(monkeypatch):
    """The broker's session_end_requested means the model called end_call(farewell). It is
    recorded as a tool call, as hosted runs record it; the target's end_call flag follows the
    case and the simulated user always has the tool."""
    from types import SimpleNamespace
    target = _FakeSession([
        SimpleNamespace(type="utterance", t_ms=1, data={"text": "Goodbye."}),
        SimpleNamespace(type="done", t_ms=2, data={}),
        SimpleNamespace(type="session_end_requested", t_ms=3, data={"farewell": "Goodbye."}),
    ])
    simulator = _FakeSession([])
    sessions = iter([target, simulator])
    modes = []

    async def connect(*_args, mode, **_kwargs):
        modes.append(mode)
        return next(sessions)

    monkeypatch.setattr("dialt_recipes.simulation.DialtSession.connect", connect)
    case = SimulationCase.from_dict({
        "name": "n", "starter": "Hello", "target": {"end_call": False},
        "checks": [{"type": "tool_called", "value": "end_call"}], "limits": {"timeout_s": 10},
    })
    report = asyncio.run(run_simulation("ws://test", "key", case))

    assert [mode.end_call for mode in modes] == [False, True]
    assert report.termination_reason == "completed"
    assert [e for e in report.events if e["type"] == "tool_call"] == [{
        "side": "target", "type": "tool_call", "t_ms": 3, "name": "end_call",
        "id": "end_call", "args": {"farewell": "Goodbye."},
    }]
    assert report.check_results[0]["pass"] is True and report.passed


def test_checks_match_the_hosted_runner_and_skip_judges():
    case = SimulationCase.from_dict({
        "name": "booking", "starter": "Hello",
        "target": {"instructions": "Book safely.", "tools": [
            {"name": "lookup"},
            {"name": "intake", "parameters": {"type": "object", "properties": {"field": {}, "value": {}}}}]},
        "simulator": {"instructions": "Want a slot."},
        "fixtures": {"intake": {"fixture_type": "field_store", "field_arg": "field",
                                "value_arg": "value", "fields": [{"key": "name"}, {"key": "phone"}]}},
        "checks": [
            {"type": "contains", "value": "3:30"},
            {"type": "not_contains", "value": "refund"},
            {"type": "regex", "value": r"assistant: .*confirmation PT-\d+"},
            {"type": "tool_called", "value": "lookup", "name": "used the lookup"},
            {"type": "fixture_complete", "value": "intake"},
            {"type": "max_turns", "value": 2},
            {"type": "judge", "criterion": "Polite throughout."},
        ],
    })
    report = SimulationReport(
        "booking", "text", "target", "simulator",
        transcript=[{"role": "user", "text": "hi"},
                    {"role": "assistant", "text": "3:30 is available; confirmation PT-1."}],
        events=[{"side": "target", "type": "tool_call", "name": "lookup"}],
        fixture_state={"intake": {"name": "Maya"}},
        termination_reason="completed",
    )
    results = evaluate_checks(case, report)
    assert [(r["name"], r["pass"]) for r in results] == [
        ("contains-1", True), ("not_contains-2", True), ("regex-3", True),
        ("used the lookup", True), ("fixture_complete-5", False), ("max_turns-6", True),
        ("judge-7", None),
    ]
    assert results[4]["detail"] == "missing required fields: phone"
    assert results[6]["skipped"] is True and results[6]["detail"] == "judge checks run hosted"
    report.check_results = results
    assert report.passed is False                       # the field store is incomplete
    report.fixture_state["intake"]["phone"] = "555"
    report.check_results = evaluate_checks(case, report)
    assert report.passed is True                        # the skipped judge does not block
    report.termination_reason = "simulator_ended"       # the simulated user hung up: checks decide
    assert report.passed is True
    report.termination_reason = "silence_guard"         # the agent went quiet: a failure
    assert report.passed is False


def test_fixtures_answer_like_hosted_runs():
    async def run():
        fixed = await _fixture_result({"lookup": {"result": {"slot": "3:30"}}}, "lookup", {})
        callback = await _fixture_result({"lookup": lambda args: {"slot": args["wanted"]}},
                                         "lookup", {"wanted": "4:00"})
        missing = await _fixture_result({}, "delete", {})
        state = {}
        store = {"fixture_type": "field_store", "field_arg": "field", "value_arg": "value",
                 "fields": [{"key": "name"}, {"key": "phone", "required": False}]}
        recorded = await _fixture_result({"intake": store}, "intake",
                                         {"field": "name", "value": " Maya "}, state)
        rejected = await _fixture_result({"intake": store}, "intake",
                                         {"field": "age", "value": "40"}, state)
        return fixed, callback, missing, recorded, rejected, state

    fixed, callback, missing, recorded, rejected, state = asyncio.run(run())
    assert fixed == ({"slot": "3:30"}, "succeeded", True)
    assert callback == ({"slot": "4:00"}, "succeeded", True)
    assert missing[0]["error"] == "unhandled_tool" and missing[1:] == ("failed", False)
    assert recorded == ({"recorded": "name", "missing_required": [], "complete": True},
                        "succeeded", True)
    assert rejected[0]["error"] == "invalid_fixture_input" and rejected[1] == "failed"
    assert state == {"intake": {"name": "Maya"}}


def test_collect_cases_reads_files_and_directories(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"name": "a", "starter": "hi"}))
    (tmp_path / "b.json").write_text(json.dumps({"name": "b", "starter": "hi"}))
    assert [case.name for case in collect_cases([tmp_path, SAMPLE])] == [
        "a", "b", "physiotherapy appointment with constraints and permission"]
    (tmp_path / "c.json").write_text(json.dumps({"name": "c", "target_instructions": "old"}))
    with pytest.raises(SystemExit, match="target.instructions"):
        collect_cases([tmp_path])
    (tmp_path / "c.json").write_text(json.dumps({"name": "c"}))
    with pytest.raises(SystemExit, match="starter must contain"):
        collect_cases([tmp_path])


def test_push_upserts_then_starts_one_run(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"name": "a", "starter": "hi"}))
    (tmp_path / "b.json").write_text(json.dumps({"name": "b", "starter": "hi"}))

    class FakeClient:
        def __init__(self):
            self.calls = []

        def upsert_cases(self, documents):
            self.calls.append(("upsert", [d["name"] for d in documents]))
            return [{"id": f"id-{d['name']}", "name": d["name"]} for d in documents]

        def start_run(self, case_ids, *, modality, repetitions):
            self.calls.append(("run", case_ids, modality, repetitions))
            return {"id": "run-1234abcd", "status": "queued"}

        def dashboard_url(self, run_id):
            return f"https://example.test/evals/{run_id}"

        def wait(self, run_id):
            return {"id": run_id, "status": "passed", "attempts": [
                {"status": "passed", "case_name": "a", "termination_reason": "completed"},
                {"status": "passed", "case_name": "b", "termination_reason": "max_turns"},
            ]}

    lines, client = [], FakeClient()
    (tmp_path / "z.json").write_text(json.dumps({"name": "z", "starter": "hi", "checks": [{"type": "contains"}]}))
    with pytest.raises(ValueError, match="z: each check needs"):
        push(client, [tmp_path], modality="text", repetitions=1, wait=False, out=lines.append)
    assert client.calls == []                                   # nothing upserted on a bad file
    (tmp_path / "z.json").unlink()
    run = push(client, [tmp_path], modality="voice", repetitions=2, wait=True, out=lines.append)
    assert client.calls == [("upsert", ["a", "b"]), ("run", ["id-a", "id-b"], "voice", 2)]
    assert run["status"] == "passed"
    assert lines[0] == "2 cases pushed: a, b"
    assert lines[1] == "run run-1234 started (voice): https://example.test/evals/run-1234abcd"
    assert lines[-1] == "run run-1234 passed"


def test_bridge_and_final_are_one_turn_and_voice_starters_are_checked():
    transcript = [{"role": "user", "text": "hi"},
                  {"role": "assistant", "text": "Let me check.", "turn": "turn1"},
                  {"role": "assistant", "text": "Tuesday.", "turn": "turn1"},
                  {"role": "assistant", "text": "Else?", "turn": "turn2"},
                  {"role": "assistant", "text": "legacy entry"}]
    assert assistant_turns(transcript) == 3
    case = SimulationCase.from_dict({"name": "n", "starter": "I need help. " * 40})
    assert case.max_turns == 20
    with pytest.raises(ValueError, match="300 characters"):
        SimulationCase.from_dict({"name": "n", "starter": "I need help. " * 40}, modality="voice")


def test_dialt_sim_checks_voice_starters_before_running(tmp_path):
    (tmp_path / "long.json").write_text(json.dumps({"name": "long", "starter": "I need help. " * 40}))
    assert collect_cases([tmp_path], "text")[0].name == "long"
    with pytest.raises(SystemExit, match="300 characters"):
        collect_cases([tmp_path], "voice")
