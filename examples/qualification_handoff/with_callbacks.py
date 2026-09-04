from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from dialt_recipes import SimulationCase, run_simulation

from workflow import QualificationState, spoken_reference_pattern


async def main() -> None:
    load_dotenv()
    source = Path(__file__).with_name("evals") / "accepted_handoff.json"
    case = SimulationCase.from_dict(json.loads(source.read_text()))
    state = QualificationState()

    case = replace(
        case,
        fixtures={
            "record_qualification": state.record,
            "start_handoff": state.start_handoff,
        },
        checks=tuple(
            {**check, "value": spoken_reference_pattern(state.handoff_reference)}
            if check.get("type") == "regex" and "2048" in check.get("value", "")
            else check
            for check in case.checks
            if check.get("type") != "fixture_complete"
        ),
    )
    report = await run_simulation(
        os.environ.get("DIALT_URL", "wss://dialt.com/ws"),
        os.environ["DIALT_API_KEY"],
        case,
        modality=os.environ.get("DIALT_MODALITY", "text"),
    )
    missing = [field for field in state.required_fields if field not in state.answers]
    application_check = {
        "type": "application_state",
        "name": "all qualification fields recorded",
        "pass": not missing,
        "detail": "" if not missing else f"missing: {', '.join(missing)}",
    }
    print(
        json.dumps(
            {
                "passed": report.passed and application_check["pass"],
                "qualification": state.answers,
                "application_events": state.events,
                "checks": [application_check, *report.check_results],
                "transcript": report.transcript,
                "session_ids": {
                    "target": report.target_session_id,
                    "simulator": report.simulator_session_id,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
