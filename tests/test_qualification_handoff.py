import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from dialt_recipes.cli import collect_cases


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "qualification_handoff"
EVALS = EXAMPLE / "evals"
WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "qualification_handoff_workflow", EXAMPLE / "workflow.py"
)
assert WORKFLOW_SPEC is not None and WORKFLOW_SPEC.loader is not None
WORKFLOW = importlib.util.module_from_spec(WORKFLOW_SPEC)
sys.modules[WORKFLOW_SPEC.name] = WORKFLOW
WORKFLOW_SPEC.loader.exec_module(WORKFLOW)
QualificationState = WORKFLOW.QualificationState


def test_all_qualification_handoff_cases_use_the_hosted_shape() -> None:
    cases = collect_cases([EVALS], modality="voice")
    assert [case.name for case in cases] == [
        "specialist qualification and accepted handoff",
        "specialist qualification accepts a correction",
        "specialist qualification with declined handoff",
        "specialist qualification with unavailable handoff",
    ]
    for case in cases:
        assert [tool["name"] for tool in case.target_tools] == [
            "record_qualification",
            "start_handoff",
        ]
        assert case.target_tools[1]["requires_permission"] is True
        assert case.fixtures["record_qualification"]["fixture_type"] == "field_store"


def test_accepted_case_has_deterministic_completion_checks() -> None:
    case = json.loads((EVALS / "accepted_handoff.json").read_text())
    assert [check["type"] for check in case["checks"][:3]] == [
        "fixture_complete",
        "tool_called",
        "regex",
    ]
    assert case["fixtures"]["start_handoff"]["result"]["handoff_reference"] == "HX-2048"
    assert case["checks"][2]["value"] == WORKFLOW.spoken_reference_pattern("HX-2048")


def test_reference_check_accepts_a_spoken_reading() -> None:
    """An agent reads a reference aloud: letters and digits separated by pauses, hyphens or the
    word "dash", digits sometimes as words. The check accepts those and still rejects a different
    or longer reference."""
    pattern = re.compile(WORKFLOW.spoken_reference_pattern("HX-2048"))
    for spoken in ("HX-2048", "H X 2 0 4 8", "H-X-2-0-4-8", "H X dash 2 0 4 8", "H, X, two zero four eight"):
        assert pattern.search(spoken), spoken
    for other in ("HX-2049", "HX-20480", "H X two zero four eight one"):
        assert not pattern.search(other), other


def test_qualification_state_records_corrections_and_handoff() -> None:
    state = QualificationState()
    assert state.record({"field": "need", "value": "  choose a plan "})["complete"] is False
    state.record({"field": "region", "value": "south"})
    state.record({"field": "region", "value": "north"})
    result = state.record({"field": "timeframe", "value": "this week"})
    assert result == {
        "recorded": "timeframe",
        "missing_required": [],
        "complete": True,
    }
    assert state.answers == {
        "need": "choose a plan",
        "region": "north",
        "timeframe": "this week",
    }

    handoff = state.start_handoff({"summary": "North region, this week."})
    assert handoff == {
        "handoff_requested": True,
        "handoff_reference": "demo-handoff-001",
    }
    assert state.start_handoff({"summary": "duplicate"})["duplicate"] is True


def test_qualification_state_rejects_incomplete_or_invalid_calls() -> None:
    state = QualificationState()
    with pytest.raises(ValueError, match="configured qualification field"):
        state.record({"field": "unknown", "value": "x"})
    with pytest.raises(ValueError, match="non-empty"):
        state.record({"field": "need", "value": " "})
    state.record({"field": "need", "value": "choose a plan"})
    with pytest.raises(ValueError, match="missing: region, timeframe"):
        state.start_handoff({"summary": "not ready"})
