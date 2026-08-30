from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from converse_sdk import ConverseMode, ConverseSession
from converse_sdk.evals import EvalsError, validate_case
from converse_sdk.relay import (
    SIMULATION_SILENCE_END_S,
    SIMULATION_SILENCE_NUDGE_S,
    TextTurnRelay,
    VoiceTurnRelay,
)

Fixture = Any | Callable[[dict[str, Any]], Any | Awaitable[Any]]

# The same case document the hosted evals API accepts: run it here, push it unchanged.
# A tool turn can commit as a bridge line and then its final answer; both carry the same turn id
# root (suffixes -bridge / -final / -degraded / -recovery-final). One conversational turn.
_TURN_SUFFIX = re.compile(r"-(?:bridge-)?(?:bridge|final|degraded|recovery-final)$")


def turn_root(turn_id) -> str | None:
    if not isinstance(turn_id, str) or not turn_id:
        return None
    return _TURN_SUFFIX.sub("", turn_id)


def assistant_turns(transcript: list[dict]) -> int:
    """Conversational assistant turns: entries sharing a turn root count once."""
    roots, count = set(), 0
    for turn in transcript:
        if turn.get("role") != "assistant":
            continue
        root = turn.get("turn")
        if root is None:
            count += 1
        elif root not in roots:
            roots.add(root)
            count += 1
    return count


LEGACY_KEYS = ("target_instructions", "simulator_instructions", "target_tools", "expected")
MAX_FIXTURE_FIELD_VALUE_CHARS = 20_000


@dataclass(frozen=True)
class SimulationCase:
    name: str
    starter: str
    target_instructions: str
    simulator_instructions: str
    target_tools: tuple[dict[str, Any], ...] = ()
    target_options: dict[str, Any] = field(default_factory=dict)   # voice, web_search
    fixtures: dict[str, Fixture] = field(default_factory=dict)
    checks: tuple[dict[str, Any], ...] = ()
    max_turns: int = 20
    timeout_s: float = 600.0
    silence_s: float = 30.0

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, modality: str | None = None) -> "SimulationCase":
        """Build a case from the hosted case document.

        ``name``, ``starter``, ``target`` (``instructions``, optional ``tools``, ``voice``,
        ``web_search``), ``simulator`` (``instructions``), ``fixtures``, ``checks`` and
        ``limits`` (``max_turns``, ``timeout_s``, ``silence_s``).
        """
        stale = [key for key in LEGACY_KEYS if key in value]
        if stale:
            raise ValueError(
                f"{', '.join(stale)}: cases use the hosted shape (target.instructions, "
                "target.tools, simulator.instructions, checks, limits); see the evals guide")
        value = validate_case(value, modality=modality)   # the hosted rules and messages
        target = value.get("target") or {}
        simulator = value.get("simulator") or {}
        limits = value["limits"]
        checks = tuple(value["checks"])
        return cls(
            name=str(value["name"]),
            starter=str(value["starter"]),
            target_instructions=str(target.get("instructions") or ""),
            simulator_instructions=str(simulator.get("instructions") or ""),
            target_tools=tuple(target.get("tools") or ()),
            target_options={key: target[key] for key in ("voice", "web_search") if key in target},
            fixtures=dict(value.get("fixtures") or {}),
            checks=checks,
            max_turns=int(limits["max_turns"]),
            timeout_s=float(limits["timeout_s"]),
            silence_s=float(limits["silence_s"]),
        )


@dataclass
class SimulationReport:
    case_name: str
    modality: str
    target_session_id: str
    simulator_session_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    fixture_state: dict[str, dict[str, str]] = field(default_factory=dict)
    termination_reason: str = "completed"
    error: str = ""
    check_results: list[dict[str, Any]] = field(default_factory=list)

    def assistant_text(self) -> str:
        return "\n".join(
            turn["text"] for turn in self.transcript if turn["role"] == "assistant")

    @property
    def passed(self) -> bool:
        """No error, an end the agent is not to blame for, and every check that ran here
        passed. The simulated user ending the conversation or going quiet is named in the
        termination reason, but the checks decide. Judge checks only run hosted (they need the
        judge model); they are reported as skipped, not as failures."""
        return (
            not self.error
            and self.termination_reason in {
                "completed", "max_turns", "simulator_ended", "simulator_silent"}
            and all(result["pass"] for result in self.check_results if not result.get("skipped"))
        )


def evaluate_checks(case: SimulationCase, report: SimulationReport) -> list[dict[str, Any]]:
    """The deterministic checks, with the hosted runner's semantics. Judge checks are skipped."""
    results: list[dict[str, Any]] = []
    assistant = report.assistant_text()
    transcript = "\n".join(f"{turn['role']}: {turn['text']}" for turn in report.transcript)
    called = {
        str(event.get("name"))
        for event in report.events
        if event.get("side") == "target" and event.get("type") == "tool_call"
    }
    for index, check in enumerate(case.checks):
        kind = check.get("type")
        name = check.get("name") or f"{kind}-{index + 1}"
        if kind == "judge":
            results.append({
                "type": kind, "name": name, "criterion": check.get("criterion"),
                "pass": None, "skipped": True, "detail": "judge checks run hosted",
            })
            continue
        value = str(check.get("value") or "")
        passed, detail = False, ""
        if kind == "contains":
            passed = value.casefold() in assistant.casefold()
        elif kind == "not_contains":
            passed = value.casefold() not in assistant.casefold()
        elif kind == "regex":
            try:
                passed = re.search(value, transcript, re.IGNORECASE) is not None
            except re.error as exc:
                detail = f"invalid regex: {exc}"
        elif kind == "tool_called":
            passed = value in called
        elif kind == "fixture_complete":
            fixture = case.fixtures.get(value, {})
            stored = report.fixture_state.get(value, {})
            missing = [
                item["key"] for item in (fixture.get("fields", []) if isinstance(fixture, dict) else [])
                if item.get("required", True) and item["key"] not in stored
            ]
            passed = (isinstance(fixture, dict) and fixture.get("fixture_type") == "field_store"
                      and not missing)
            if missing:
                detail = f"missing required fields: {', '.join(missing)}"
        elif kind == "max_turns":
            passed = assistant_turns(report.transcript) <= int(check.get("value", 20))
        results.append({"type": kind, "name": name, "value": value, "pass": passed,
                        "detail": detail})
    return results


async def _fixture_result(fixtures: dict[str, Fixture], name: str, args: dict[str, Any],
                          state: dict[str, dict[str, str]] | None = None) -> tuple[Any, str, bool]:
    """Answer a target tool call: a Python callable, a field_store, or a fixed value.

    An undeclared tool fails closed, exactly as hosted runs do, so a local pass means the
    hosted run will not be answering tools the case forgot to declare.
    """
    if name not in fixtures:
        return ({"error": "unhandled_tool", "tool": name,
                 "instruction": "The case declares no fixture for this tool; fail closed."},
                "failed", False)
    value = fixtures[name]
    if callable(value):
        value = value(args)
        if inspect.isawaitable(value):
            value = await value
        return value, "succeeded", True
    if isinstance(value, dict) and value.get("fixture_type") == "field_store":
        state = state if state is not None else {}
        field_arg, value_arg = value["field_arg"], value["value_arg"]
        key, recorded = args.get(field_arg), args.get(value_arg)
        known = [item["key"] for item in value["fields"]]
        if (key not in known or not isinstance(recorded, str) or not recorded.strip()
                or len(recorded) > MAX_FIXTURE_FIELD_VALUE_CHARS):
            return ({"error": "invalid_fixture_input", "tool": name,
                     "instruction": (f"{field_arg} must name a configured field and {value_arg} "
                                     f"must be 1 to {MAX_FIXTURE_FIELD_VALUE_CHARS} characters of text.")},
                    "failed", False)
        stored = state.setdefault(name, {})
        stored[str(key)] = recorded.strip()
        missing = [item["key"] for item in value["fields"]
                   if item.get("required", True) and item["key"] not in stored]
        return {"recorded": key, "missing_required": missing, "complete": not missing}, "succeeded", True
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    return value, "succeeded", True


async def run_simulation(url: str, api_key: str, case: SimulationCase, *,
                         modality: str = "text") -> SimulationReport:
    """Run target and simulated user as two ordinary Converse sessions.

    In voice mode, received Float32 audio is paced by the service and immediately sent to the
    other session. It is never opened on a sound device or written to an output file.
    """
    if modality not in {"text", "voice"}:
        raise ValueError("modality must be text or voice")
    suffix = uuid.uuid4().hex[:10]
    target_id, simulator_id = f"recipe-target-{suffix}", f"recipe-user-{suffix}"
    target = await ConverseSession.connect(
        url, api_key=api_key, session_id=target_id,
        mode=ConverseMode(
            modality=modality, instructions=case.target_instructions,
            tools=list(case.target_tools) or None, greeting=False,
            voice=case.target_options.get("voice"),
            web_search=bool(case.target_options.get("web_search", False)),
            silence_nudge_s=SIMULATION_SILENCE_NUDGE_S if modality == "voice" else None,
            silence_end_s=SIMULATION_SILENCE_END_S if modality == "voice" else None,
        ),
    )
    try:
        simulator = await ConverseSession.connect(
            url, api_key=api_key, session_id=simulator_id,
            mode=ConverseMode(
                modality=modality, instructions=case.simulator_instructions,
                tools=None, greeting=case.starter if modality == "voice" else False,
                silence_nudge_s=SIMULATION_SILENCE_NUDGE_S if modality == "voice" else None,
                silence_end_s=SIMULATION_SILENCE_END_S if modality == "voice" else None,
            ),
        )
    except Exception:
        await target.close()
        raise

    report = SimulationReport(case.name, modality, target_id, simulator_id)
    stop = asyncio.Event()
    last_activity = {"at": time.monotonic()}
    target_turns = {"count": 0}
    target_turn_roots: set[str] = set()
    last_speaker = {"side": None}    # who spoke last: a silence is the other side's to explain
    repetition = {"text": "", "count": 0}

    async def forward_text(destination: ConverseSession, text: str) -> None:
        normalized = " ".join(text.casefold().split())
        if normalized and normalized == repetition["text"]:
            repetition["count"] += 1
        else:
            repetition.update(text=normalized, count=1)
        if repetition["count"] >= 3:
            report.termination_reason = "repetition_guard"
            stop.set()
            return
        await destination.send_text(text)

    def relay_failed(error: BaseException) -> None:
        if stop.is_set():
            return
        report.termination_reason = "connection_error"
        report.error = repr(error)[:4000]
        stop.set()

    relays = {
        "target": TextTurnRelay(
            lambda text: forward_text(simulator, text), on_error=relay_failed),
        "simulator": TextTurnRelay(
            lambda text: forward_text(target, text), on_error=relay_failed),
    }
    voice_relays = {
        "target": VoiceTurnRelay(simulator, on_error=relay_failed),
        "simulator": VoiceTurnRelay(target, on_error=relay_failed),
    }

    async def consume(side: str, source: ConverseSession,
                      destination: ConverseSession) -> None:
        try:
            async for event in source.events():
                last_activity["at"] = time.monotonic()
                if len(report.events) < 2000:
                    report.events.append({
                        "side": side, "type": event.type, "t_ms": event.t_ms, **event.data,
                    })
                if event.type == "asr":
                    if modality == "voice":
                        incoming_side = "simulator" if side == "target" else "target"
                        voice_relays[incoming_side].input_committed()
                    if side == "target":
                        report.transcript.append({
                            "role": "user", "text": event.data.get("text", ""),
                        })
                elif side == "target" and event.type == "utterance":
                    text = str(event.data.get("text") or "")
                    root = turn_root(event.data.get("turn_id"))
                    entry = {"role": "assistant", "text": text}
                    if root is not None:
                        entry["turn"] = root
                    report.transcript.append(entry)
                    last_speaker["side"] = "target"
                    if modality == "text":
                        relays[side].utterance(text)
                    if root is None or root not in target_turn_roots:   # bridge + final: one turn
                        if root is not None:
                            target_turn_roots.add(root)
                        target_turns["count"] += 1
                    if target_turns["count"] >= case.max_turns:
                        report.termination_reason = "max_turns"
                        stop.set()
                elif side == "simulator" and event.type == "utterance":
                    text = str(event.data.get("text") or "")
                    last_speaker["side"] = "simulator"
                    if modality == "text" and text:
                        relays[side].utterance(text)
                elif event.type == "working":
                    active = bool(event.data.get("active"))
                    if modality == "text":
                        relays[side].working(active)
                    else:
                        voice_relays[side].working(active)
                elif event.type == "audio" and modality == "voice" and event.audio is not None:
                    await voice_relays[side].audio(event.audio)
                elif event.type == "done":
                    if modality == "text":
                        relays[side].done()
                    else:
                        voice_relays[side].done()
                elif side == "target" and event.type == "tool_call":
                    call_id = str(event.data.get("id") or "")
                    tool_name = str(event.data.get("name") or "")
                    value, outcome, verified = await _fixture_result(
                        case.fixtures, tool_name, event.data.get("args") or {},
                        report.fixture_state,
                    )
                    await source.send_tool_result(call_id, value, outcome=outcome, verified=verified)
                    fixture = case.fixtures.get(tool_name)
                    if len(report.events) < 2000:
                        report.events.append({
                            "side": "target", "type": "tool_result", "t_ms": event.t_ms,
                            "id": call_id, "name": tool_name, "outcome": outcome,
                            "verified": verified,
                            "fixture": ("unhandled" if fixture is None else "callable"
                                        if callable(fixture) else fixture.get("fixture_type", "fixed")
                                        if isinstance(fixture, dict) else "fixed"),
                        })
                elif event.type == "session_end_requested":
                    if not stop.is_set():
                        # The simulated user hanging up is a harness event, not the agent's.
                        report.termination_reason = (
                            "completed" if side == "target" else "simulator_ended")
                        stop.set()
                elif event.type == "error":
                    report.termination_reason = "connection_error"
                    report.error = str(event.data.get("detail") or event.data.get("code") or "error")
                    stop.set()
            if not stop.is_set():
                report.termination_reason = "connection_closed"
                stop.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            report.termination_reason = "connection_error"
            report.error = repr(exc)[:4000]
            stop.set()

    async def watchdog() -> None:
        while not stop.is_set():
            await asyncio.sleep(1)
            if time.monotonic() - last_activity["at"] >= case.silence_s:
                # The agent spoke last and the simulated user never answered: simulator_silent.
                report.termination_reason = (
                    "simulator_silent" if last_speaker["side"] == "target" else "silence_guard")
                stop.set()

    tasks = [
        asyncio.create_task(consume("target", target, simulator)),
        asyncio.create_task(consume("simulator", simulator, target)),
        asyncio.create_task(watchdog()),
    ]
    try:
        if modality == "text":
            await target.send_text(case.starter)
        try:
            await asyncio.wait_for(stop.wait(), timeout=case.timeout_s)
        except TimeoutError:
            report.termination_reason = "timeout"
            stop.set()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(relay.close() for relay in relays.values()))
        await asyncio.gather(*(relay.close() for relay in voice_relays.values()))
        await asyncio.gather(target.close(), simulator.close(), return_exceptions=True)
    report.check_results = evaluate_checks(case, report)
    return report


async def report_attempt(base_url: str, api_key: str, run_id: str, case_id: str,
                         report: SimulationReport, *, repetition: int = 1,
                         idempotency_key: str | None = None) -> dict[str, Any]:
    """Post a local result into a hosted run created with execution="local"."""
    status = "passed" if report.passed else "failed"
    payload = {
        "idempotency_key": idempotency_key or f"recipe-{report.target_session_id}",
        "case_id": case_id, "repetition": repetition, "status": status,
        "target_session_id": report.target_session_id,
        "simulator_session_id": report.simulator_session_id,
        "transcript": report.transcript, "events": report.events,
        "check_results": [r for r in report.check_results if not r.get("skipped")],
        # A judge the local run could not evaluate stays visible on the run page as Skipped.
        "judge_results": [r for r in report.check_results if r.get("skipped")],
        "termination_reason": report.termination_reason, "error": report.error,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/app/evals/runs/{run_id}/report",
            headers={"Authorization": f"Bearer {api_key}"}, json=payload,
        )
        if response.is_error:
            try:
                detail = response.json().get("error") or response.json().get("detail")
            except ValueError:
                detail = response.text[:500]
            raise EvalsError(response.status_code, str(detail or "report rejected"))
        return response.json()
