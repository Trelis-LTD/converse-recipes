from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from converse_sdk.evals import EvalsClient, EvalsError, load_cases, validate_case
from dotenv import load_dotenv

from .conversation_plan import ConversationPlan
from .guided import GuidedAssistant
from .simulation import SimulationCase, report_attempt, run_simulation

DEFAULT_EVALS_URL = "https://dialt.com"


def _api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("CONVERSE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CONVERSE_API_KEY is required in the environment or .env")
    return api_key


def _credentials() -> tuple[str, str]:
    api_key = _api_key()
    return os.environ.get("CONVERSE_URL", "wss://dialt.com/ws"), api_key


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


def collect_cases(paths: list[Path], modality: str | None = None) -> list[SimulationCase]:
    """Every case file named, or every *.json case in each directory named."""
    cases = []
    for path in paths:
        for document in load_cases(path):
            try:
                cases.append(SimulationCase.from_dict(document, modality=modality))
            except ValueError as exc:
                raise SystemExit(f"{path}: {exc}") from None
    return cases


def _summary(report) -> dict:
    return {
        "case": report.case_name, "modality": report.modality,
        "target_session_id": report.target_session_id,
        "simulator_session_id": report.simulator_session_id,
        "termination_reason": report.termination_reason, "error": report.error,
        "passed": report.passed, "checks": report.check_results,
        "transcript": report.transcript,
    }


async def _simulation(args) -> None:
    url, api_key = _credentials()
    cases = collect_cases(args.paths, args.modality)
    report_args = (args.report_base_url, args.run_id, args.case_id)
    if any(report_args) and not all(report_args):
        raise SystemExit("reporting requires --report-base-url, --run-id, and --case-id together")
    if all(report_args) and len(cases) != 1:
        raise SystemExit("reporting into a hosted run takes exactly one case")
    failed = 0
    for case in cases:
        report = await run_simulation(url, api_key, case, modality=args.modality)
        print(json.dumps(_summary(report), indent=2))
        if all(report_args):
            try:
                await report_attempt(
                    args.report_base_url, api_key, args.run_id, args.case_id, report,
                    repetition=args.repetition,
                )
            except EvalsError as exc:
                raise SystemExit(f"report rejected: {exc}") from None
        failed += not report.passed
    if len(cases) > 1:
        print(f"{len(cases) - failed} of {len(cases)} cases passed")
    if failed:
        raise SystemExit(1)


def simulation_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cases locally: Converse against a Converse simulated user")
    parser.add_argument("paths", type=Path, nargs="+", metavar="CASE_OR_DIR")
    parser.add_argument("--modality", choices=["text", "voice"], default="text")
    parser.add_argument("--report-base-url")
    parser.add_argument("--run-id")
    parser.add_argument("--case-id")
    parser.add_argument("--repetition", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(_simulation(args))


def push(client: EvalsClient, paths: list[Path], *, modality: str, repetitions: int,
         wait: bool, out=print) -> dict:
    """Upsert every case, start one hosted run over them, print the dashboard link."""
    documents = []
    for path in paths:
        for document in load_cases(path):
            try:
                documents.append(validate_case(document, modality=modality))   # every file first
            except ValueError as exc:
                raise ValueError(f"{path}: {document.get('name', '?')}: {exc}") from None
    cases = client.upsert_cases(documents)
    out(f"{len(cases)} case{'s' if len(cases) != 1 else ''} pushed: "
        + ", ".join(case["name"] for case in cases))
    run = client.start_run([case["id"] for case in cases], modality=modality,
                           repetitions=repetitions)
    out(f"run {run['id'][:8]} started ({modality}): {client.dashboard_url(run['id'])}")
    if not wait:
        return run
    run = client.wait(run["id"])
    for attempt in run.get("attempts", []):
        out(f"  {attempt['status']:<9} {attempt['case_name']}"
            + (f"  ({attempt['termination_reason']})" if attempt.get("termination_reason") else ""))
    out(f"run {run['id'][:8]} {run['status']}")
    return run


def evals_main() -> None:
    parser = argparse.ArgumentParser(description="Hosted evals from your case files")
    commands = parser.add_subparsers(dest="command", required=True)
    push_cmd = commands.add_parser("push", help="upsert the cases and start a hosted run")
    push_cmd.add_argument("paths", type=Path, nargs="+", metavar="CASE_OR_DIR")
    push_cmd.add_argument("--modality", choices=["text", "voice"], default="text")
    push_cmd.add_argument("--repetitions", type=int, default=1)
    push_cmd.add_argument("--wait", action="store_true", help="poll until the run finishes")
    push_cmd.add_argument("--base-url", default=os.environ.get("CONVERSE_EVALS_URL", DEFAULT_EVALS_URL))
    args = parser.parse_args()
    client = EvalsClient(_api_key(), base_url=args.base_url)
    try:
        run = push(client, args.paths, modality=args.modality, repetitions=args.repetitions,
                   wait=args.wait)
    except (EvalsError, ValueError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from None
    if args.wait and run.get("status") != "passed":
        raise SystemExit(1)
