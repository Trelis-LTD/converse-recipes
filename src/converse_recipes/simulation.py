from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from converse_sdk import ConverseMode, ConverseSession
from converse_sdk.relay import (
    SIMULATION_SILENCE_END_S,
    SIMULATION_SILENCE_NUDGE_S,
    TextTurnRelay,
    VoiceTurnRelay,
)

Fixture = Any | Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class SimulationCase:
    name: str
    target_instructions: str
    simulator_instructions: str
    starter: str
    target_tools: tuple[dict[str, Any], ...] = ()
    fixtures: dict[str, Fixture] = field(default_factory=dict)
    expected: dict[str, tuple[str, ...]] = field(default_factory=dict)
    max_turns: int = 8
    timeout_s: float = 180.0
    silence_s: float = 30.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SimulationCase":
        return cls(
            name=str(value["name"]),
            target_instructions=str(value["target_instructions"]),
            simulator_instructions=str(value["simulator_instructions"]),
            starter=str(value["starter"]),
            target_tools=tuple(value.get("target_tools", [])),
            fixtures=dict(value.get("fixtures", {})),
            expected={
                str(key): tuple(str(item) for item in items)
                for key, items in value.get("expected", {}).items()
            },
            max_turns=int(value.get("max_turns", 8)),
            timeout_s=float(value.get("timeout_s", 180)),
            silence_s=float(value.get("silence_s", 30)),
        )


@dataclass
class SimulationReport:
    case_name: str
    modality: str
    target_session_id: str
    simulator_session_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = "completed"
    error: str = ""
    check_results: list[dict[str, Any]] = field(default_factory=list)

    def assistant_text(self) -> str:
        return "\n".join(
            turn["text"] for turn in self.transcript if turn["role"] == "assistant")

    @property
    def passed(self) -> bool:
        return (
            not self.error
            and self.termination_reason in {"expectations_met", "max_turns"}
            and all(result["pass"] for result in self.check_results)
        )


def evaluate_expectations(case: SimulationCase,
                          report: SimulationReport) -> list[dict[str, Any]]:
    """Evaluate the deliberately small, deterministic recipe check vocabulary."""
    results: list[dict[str, Any]] = []
    assistant_text = report.assistant_text().casefold()
    called_tools = {
        str(event.get("name"))
        for event in report.events
        if event.get("side") == "target" and event.get("type") == "tool_call"
    }
    for value in case.expected.get("assistant_contains", ()):
        results.append({
            "type": "assistant_contains", "value": value,
            "pass": value.casefold() in assistant_text,
        })
    for value in case.expected.get("tools_called", ()):
        results.append({
            "type": "tool_called", "value": value, "pass": value in called_tools,
        })
    return results


async def _fixture_result(fixtures: dict[str, Fixture], name: str,
                          args: dict[str, Any]) -> tuple[Any, str, bool]:
    if name not in fixtures:
        return {"error": "unhandled_tool", "tool": name}, "failed", False
    value = fixtures[name]
    if callable(value):
        value = value(args)
        if inspect.isawaitable(value):
            value = await value
    if isinstance(value, dict) and set(value) == {"result"}:
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
                    report.transcript.append({"role": "assistant", "text": text})
                    if modality == "text":
                        relays[side].utterance(text)
                    target_turns["count"] += 1
                    if target_turns["count"] >= case.max_turns:
                        report.termination_reason = "max_turns"
                        stop.set()
                elif side == "simulator" and event.type == "utterance":
                    text = str(event.data.get("text") or "")
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
                    if side == "target" and case.expected:
                        interim = evaluate_expectations(case, report)
                        if interim and all(check["pass"] for check in interim):
                            report.check_results = interim
                            report.termination_reason = "expectations_met"
                            stop.set()
                            continue
                    if modality == "text":
                        relays[side].done()
                    else:
                        voice_relays[side].done()
                elif side == "target" and event.type == "tool_call":
                    value, outcome, verified = await _fixture_result(
                        case.fixtures, str(event.data.get("name") or ""),
                        event.data.get("args") or {},
                    )
                    await source.send_tool_result(
                        str(event.data.get("id") or ""), value,
                        outcome=outcome, verified=verified,
                    )
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
                report.termination_reason = "silence_guard"
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
    report.check_results = evaluate_expectations(case, report)
    return report


async def report_attempt(base_url: str, api_key: str, run_id: str, case_id: str,
                         report: SimulationReport, *, repetition: int = 1,
                         idempotency_key: str | None = None) -> dict[str, Any]:
    status = "passed" if report.passed else "failed"
    payload = {
        "idempotency_key": idempotency_key or f"recipe-{report.target_session_id}",
        "case_id": case_id, "repetition": repetition, "status": status,
        "target_session_id": report.target_session_id,
        "simulator_session_id": report.simulator_session_id,
        "transcript": report.transcript, "events": report.events,
        "check_results": report.check_results,
        "termination_reason": report.termination_reason, "error": report.error,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/app/evals/runs/{run_id}/report",
            headers={"Authorization": f"Bearer {api_key}"}, json=payload,
        )
        response.raise_for_status()
        return response.json()
