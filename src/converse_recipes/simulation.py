from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
import numpy as np
from converse_sdk import ConverseMode, ConverseSession

Fixture = Any | Callable[[dict[str, Any]], Any | Awaitable[Any]]


def _voice_tail_s() -> float:
    # Read lazily (not at import): the CLI only calls load_dotenv() at runtime, and a
    # malformed value must not take down text-only entry points.
    raw_timeout_ms = os.environ.get("CARTESIA_TURN_END_TIMEOUT_MS", "").strip()
    try:
        timeout_ms = float(raw_timeout_ms) if raw_timeout_ms else 5_600
    except ValueError:
        timeout_ms = 5_600
    return max(0.0, timeout_ms / 1_000) + 0.5


SIMULATION_SILENCE_NUDGE_S = 3_600.0
SIMULATION_SILENCE_END_S = 7_200.0
TURN_RELAY_SETTLE_S = 0.15


class _TextTurnRelay:
    """Forward committed text only after any tool work behind that turn has settled."""

    def __init__(self, forward: Callable[[str], Awaitable[None]], *,
                 settle_s: float = TURN_RELAY_SETTLE_S, on_error=None):
        self._forward = forward
        self._settle_s = settle_s
        self._on_error = on_error
        self._idle = asyncio.Event()
        self._idle.set()
        self._pending: str | None = None
        self._tasks: set[asyncio.Task] = set()

    def utterance(self, text: str) -> None:
        self._pending = text

    def working(self, active: bool) -> None:
        self._idle.clear() if active else self._idle.set()

    def done(self) -> None:
        if self._pending is None:
            return
        task = asyncio.create_task(self._flush())
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None and self._on_error:
            self._on_error(error)

    async def _flush(self) -> None:
        await asyncio.sleep(self._settle_s)
        await self._idle.wait()
        text, self._pending = self._pending, None
        if text:
            await self._forward(text)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def _close_voice_turn(
        destination: ConverseSession, committed: asyncio.Event | None = None,
        stopped: asyncio.Event | None = None) -> None:
    """Advance the virtual mic until Ink commits, bounded by its hard fallback."""
    committed = committed or asyncio.Event()
    stopped = stopped or asyncio.Event()
    remaining = round(16_000 * _voice_tail_s())
    while remaining > 0 and not committed.is_set() and not stopped.is_set():
        samples = min(1_600, remaining)
        await destination.stream_audio(
            np.zeros(samples, dtype=np.float32),
            sr=16_000,
            chunk_ms=100,
            realtime=True,
        )
        remaining -= samples


class _VoiceTurnRelay:
    """Cross-pipe SDK audio live, then close the receiving turn after tool work settles."""

    def __init__(self, destination: ConverseSession, *,
                 settle_s: float = TURN_RELAY_SETTLE_S, on_error=None):
        self._destination = destination
        self._settle_s = settle_s
        self._on_error = on_error
        self._idle = asyncio.Event()
        self._idle.set()
        self._committed = asyncio.Event()
        self._tail_stop = asyncio.Event()
        self._receiving = False
        self._finish_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    async def _stop_finish(self) -> None:
        task = self._finish_task
        if task is not None and not task.done():
            self._tail_stop.set()
            await asyncio.gather(task, return_exceptions=True)
        self._finish_task = None

    async def audio(self, chunk: np.ndarray) -> None:
        await self._stop_finish()
        if not self._receiving:
            self._receiving = True
            self._committed.clear()
        await self._destination.send_audio(np.asarray(chunk, dtype=np.float32))

    def working(self, active: bool) -> None:
        self._idle.clear() if active else self._idle.set()

    def done(self) -> None:
        if not self._receiving:
            return
        self._receiving = False
        self._tail_stop = asyncio.Event()
        task = asyncio.create_task(self._flush())
        self._finish_task = task
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None and self._on_error:
            self._on_error(error)

    def input_committed(self) -> None:
        self._committed.set()

    async def _flush(self) -> None:
        await asyncio.sleep(self._settle_s)
        idle = asyncio.create_task(self._idle.wait())
        stopped = asyncio.create_task(self._tail_stop.wait())
        try:
            await asyncio.wait({idle, stopped}, return_when=asyncio.FIRST_COMPLETED)
            if self._tail_stop.is_set():
                return
        finally:
            for waiter in (idle, stopped):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(idle, stopped, return_exceptions=True)
        await _close_voice_turn(self._destination, self._committed, self._tail_stop)

    async def close(self) -> None:
        await self._stop_finish()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


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
        "target": _TextTurnRelay(
            lambda text: forward_text(simulator, text), on_error=relay_failed),
        "simulator": _TextTurnRelay(
            lambda text: forward_text(target, text), on_error=relay_failed),
    }
    voice_relays = {
        "target": _VoiceTurnRelay(simulator, on_error=relay_failed),
        "simulator": _VoiceTurnRelay(target, on_error=relay_failed),
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
