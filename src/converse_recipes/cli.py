from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .conversation_plan import ConversationPlan
from .guided import GuidedAssistant
from .simulation import SimulationCase, report_attempt, run_simulation


def _credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.environ.get("CONVERSE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CONVERSE_API_KEY is required in the environment or .env")
    return os.environ.get("CONVERSE_URL", "wss://converse.trelis.com/ws"), api_key


async def _guided(path: Path) -> None:
    url, api_key = _credentials()
    plan = ConversationPlan.from_dict(json.loads(path.read_text()))
    assistant = await GuidedAssistant.connect(url, api_key, plan, modality="text", greeting=False)

    async def print_events() -> None:
        async for event in assistant.events():
            if event.type == "utterance":
                print(f"assistant> {event.data.get('text', '')}")
            elif event.type == "tool_call" and event.data.get("name") == plan.tool_name:
                recorded = event.data.get("args", {}).get("field")
                print(f"  [recorded {recorded}; {len(assistant.answers)}/{len(plan.fields)} fields]")

    event_task = asyncio.create_task(print_events())
    try:
        print(f"{plan.name}. Type /quit to stop.\n")
        while True:
            text = await asyncio.to_thread(input, "you> ")
            if text.strip() == "/quit":
                break
            await assistant.send_text(text)
    finally:
        event_task.cancel()
        await asyncio.gather(event_task, return_exceptions=True)
        await assistant.close()
        print(json.dumps({"complete": assistant.complete, "answers": assistant.answers}, indent=2))


def guided_main() -> None:
    parser = argparse.ArgumentParser(description="Run a ConversationPlan with Converse")
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    asyncio.run(_guided(args.plan))


async def _simulation(args) -> None:
    url, api_key = _credentials()
    case = SimulationCase.from_dict(json.loads(args.case.read_text()))
    report = await run_simulation(url, api_key, case, modality=args.modality)
    print(json.dumps({
        "case": report.case_name, "modality": report.modality,
        "target_session_id": report.target_session_id,
        "simulator_session_id": report.simulator_session_id,
        "termination_reason": report.termination_reason, "error": report.error,
        "passed": report.passed, "checks": report.check_results,
        "transcript": report.transcript,
    }, indent=2))
    report_args = (args.report_base_url, args.run_id, args.case_id)
    if any(report_args) and not all(report_args):
        raise SystemExit("reporting requires --report-base-url, --run-id, and --case-id together")
    if all(report_args):
        await report_attempt(
            args.report_base_url, api_key, args.run_id, args.case_id, report,
            repetition=args.repetition,
        )
    if not report.passed:
        raise SystemExit(1)


def simulation_main() -> None:
    parser = argparse.ArgumentParser(description="Run Converse against a Converse simulated user")
    parser.add_argument("case", type=Path)
    parser.add_argument("--modality", choices=["text", "voice"], default="text")
    parser.add_argument("--report-base-url")
    parser.add_argument("--run-id")
    parser.add_argument("--case-id")
    parser.add_argument("--repetition", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(_simulation(args))
