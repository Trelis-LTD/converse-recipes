"""Twilio Media Streams to Dialt call bridge: the transport core, reusable by any host.

The host owns everything that is its own: the FastAPI routes and Twilio credentials, the
`DialtMode` (instructions, greeting, tools), what a tool call does, and any bookkeeping about the
call. This module owns the transport: signature helpers, TwiML for the stream, the stream
handshake, μ-law and sample-rate conversion, paced outbound audio with playback accounting,
barge-in clears, tool dispatch, and the drain and hang-up at the end.

Outbound audio is sent at the line's own rate: one 20 ms frame per 20 ms, at most
OUTBOUND_LEAD_MS ahead of the clock, with a playback mark every MARK_EVERY_MS. Sending each reply
as a burst with a mark per frame was measured (Twilio's dual-channel recording against the
session's own track, 2026-09-04) at one fifth of frames never reaching the caller; pacing halved
that and removed every hole longer than 180 ms.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from dialt import DialtError, DialtMode, DialtSession, float32_to_pcm16
from twilio.request_validator import RequestValidator

from .telephony_audio import TelephonyAudioBridge

logger = logging.getLogger("dialt_recipes.twilio")

TWILIO_SR = 8_000
FRAME_MS = 20
MULAW_CHUNK_BYTES = TWILIO_SR * FRAME_MS // 1_000  # 160 bytes: one 20 ms frame
OUTBOUND_LEAD_MS = 200      # the sender may run this far ahead of the line's clock
MARK_EVERY_MS = 100         # playback accounting granularity; bounds hard-clear error
PLAYBACK_DRAIN_TIMEOUT_S = 10.0
MAX_WS_MESSAGE_CHARS = 65_536
MAX_MULAW_FRAME_BYTES = 4_096

EventHook = Callable[[Any], Awaitable[None]]
ToolHook = Callable[[str, dict[str, Any]], Awaitable[Any]]
SessionHook = Callable[[DialtSession], Awaitable[None]]


@dataclass(frozen=True)
class TwilioBridgeSettings:
    """What the transport needs. Hosts usually extend this with their own settings."""

    dialt_api_key: str
    twilio_auth_token: str
    public_base_url: str
    dialt_url: str = "wss://dialt.com/ws"

    def http_url(self, path: str) -> str:
        return f"{self.public_base_url.rstrip('/')}{path}"

    def websocket_url(self, path: str) -> str:
        base = self.public_base_url.rstrip("/")
        if base.startswith("https://"):
            base = f"wss://{base.removeprefix('https://')}"
        elif base.startswith("http://"):
            base = f"ws://{base.removeprefix('http://')}"
        return f"{base}{path}"


def twilio_signature_is_valid(
    auth_token: str, url: str, params: dict[str, Any], signature: str | None
) -> bool:
    """Twilio signs the exact external URL it called, form params included for POSTs."""
    if not signature:
        return False
    return RequestValidator(auth_token).validate(url, params, signature)


def connect_stream_twiml(settings: TwilioBridgeSettings, *, prelude: str = "",
                         media_path: str = "/media", status_path: str = "/stream-status") -> str:
    """TwiML that connects the call to the bridge's media websocket. `prelude` is raw TwiML
    played before the stream connects (a ringback tone, a compliance notice)."""
    stream_url = html.escape(settings.websocket_url(media_path), quote=True)
    status_url = html.escape(settings.http_url(status_path), quote=True)
    return (
        f"<Response>{prelude}<Connect>"
        f'<Stream url="{stream_url}" statusCallback="{status_url}" '
        'statusCallbackMethod="POST" />'
        "</Connect><Hangup /></Response>"
    )


async def receive_stream_start(websocket: Any) -> dict[str, Any]:
    """Wait for Twilio's `start` message and return it. Raises ValueError on a bad message and
    ConnectionError if the stream stops first."""
    while True:
        raw = await websocket.receive_text()
        if len(raw) > MAX_WS_MESSAGE_CHARS:
            raise ValueError("Twilio message too large")
        message = json.loads(raw)
        event = message.get("event")
        if event == "start":
            start = message.get("start")
            if not isinstance(start, dict) or not start.get("streamSid"):
                raise ValueError("Twilio start message missing streamSid")
            return start
        if event == "stop":
            raise ConnectionError("Twilio stream stopped before it started")


class PlaybackLedger:
    """Audio sent to Twilio but not yet acknowledged as played, by mark."""

    def __init__(self) -> None:
        self._pending: OrderedDict[str, float] = OrderedDict()
        self._sequence = 0
        self.changed = asyncio.Event()

    def add(self, duration_ms: float) -> str:
        self._sequence += 1
        name = f"dialt-{self._sequence}"
        self._pending[name] = duration_ms
        self.changed.set()
        return name

    def acknowledge(self, name: str) -> None:
        if self._pending.pop(name, None) is not None:
            self.changed.set()

    def pending_ms(self) -> float:
        return sum(self._pending.values())

    def empty(self) -> bool:
        return not self._pending

    def clear(self) -> float:
        discarded_ms = self.pending_ms()
        self._pending.clear()
        self.changed.set()
        return discarded_ms

    def interruption(self, *, hard_clear: bool) -> tuple[float, float]:
        """(remaining_ms, discarded_ms) for Dialt's playback_stopped report."""
        if hard_clear:
            return 0, self.clear()
        return self.pending_ms(), 0


@dataclass
class BridgeHooks:
    """Where the host plugs in.

    execute_tool(name, args): run one application tool and return its result; raise to report
        a failed tool. Undeclared tools never reach it.
    on_event(event): every Dialt session event, before the bridge acts on it, for bookkeeping
        (transcripts, tool timing, permission outcomes). Must not block for long.
    on_connected(session): once the Dialt session is open, before audio flows (start a
        recording, inject context).
    end_call: an event the host sets to end the call itself; the bridge closes the Dialt
        session, drains playback and returns. Dialt's own end_call needs nothing from the host.
    """

    execute_tool: ToolHook
    on_event: EventHook | None = None
    on_connected: SessionHook | None = None
    end_call: asyncio.Event | None = None


async def run_call_bridge(websocket: Any, stream_sid: str, call_sid: str, *,
                          settings: TwilioBridgeSettings, mode: DialtMode,
                          hooks: BridgeHooks) -> None:
    """Bridge one Twilio media stream to one Dialt session until either side ends."""
    ledger = PlaybackLedger()
    audio = TelephonyAudioBridge()
    upstream_ended = asyncio.Event()

    async with await DialtSession.connect(
        settings.dialt_url, session_id=call_sid[:64], api_key=settings.dialt_api_key, mode=mode,
    ) as session:
        tool_tasks: dict[str, asyncio.Task[None]] = {}
        if hooks.on_connected is not None:
            await hooks.on_connected(session)

        async def receive_twilio() -> None:
            async for raw in websocket.iter_text():
                if len(raw) > MAX_WS_MESSAGE_CHARS:
                    raise ValueError("Twilio message too large")
                message = json.loads(raw)
                event = message.get("event")
                if event == "media":
                    payload = message.get("media", {}).get("payload")
                    if isinstance(payload, str) and not upstream_ended.is_set():
                        mulaw = base64.b64decode(payload, validate=True)
                        if len(mulaw) > MAX_MULAW_FRAME_BYTES:
                            raise ValueError("Twilio media frame too large")
                        pcm = audio.twilio_to_dialt(mulaw)
                        if pcm:
                            await session.send_audio(pcm)
                elif event == "mark":
                    name = message.get("mark", {}).get("name")
                    if isinstance(name, str):
                        ledger.acknowledge(name)
                elif event == "stop":
                    return

        async def run_tool(tool_id: str, name: str, args: dict[str, Any]) -> None:
            try:
                try:
                    result = await hooks.execute_tool(name, args)
                    outcome, verified = "succeeded", True
                except Exception as exc:  # noqa: BLE001 - a failed tool is a result, not a crash
                    logger.exception("Dialt tool %s failed call_sid=%s", name, call_sid)
                    # The model gets the failure as stated by the host, so it can decide what
                    # to say; a bare "tool_failed" told it nothing.
                    result = {"error": "tool_failed", "detail": str(exc) or type(exc).__name__}
                    outcome, verified = "failed", False
                await session.send_tool_result(tool_id, result, outcome=outcome, verified=verified)
            except asyncio.CancelledError:
                return
            finally:
                tool_tasks.pop(tool_id, None)

        async def send_media(chunk: bytes) -> None:
            await websocket.send_json({
                "event": "media", "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode("ascii")},
            })

        async def send_mark(duration_ms: float) -> None:
            await websocket.send_json({
                "event": "mark", "streamSid": stream_sid,
                "mark": {"name": ledger.add(duration_ms)},
            })

        outbound = bytearray()
        outbound_ready = asyncio.Event()

        def queue_outbound(mulaw: bytes) -> None:
            outbound.extend(mulaw)
            outbound_ready.set()

        async def pace_outbound() -> None:
            loop = asyncio.get_running_loop()
            next_at = loop.time()
            unmarked_ms = 0.0
            while True:
                if len(outbound) < MULAW_CHUNK_BYTES and not (upstream_ended.is_set() and outbound):
                    if unmarked_ms:
                        await send_mark(unmarked_ms)
                        unmarked_ms = 0.0
                    outbound_ready.clear()
                    await outbound_ready.wait()
                    next_at = loop.time()
                    continue
                chunk = bytes(outbound[:MULAW_CHUNK_BYTES])
                del outbound[:MULAW_CHUNK_BYTES]
                await send_media(chunk)
                unmarked_ms += len(chunk) * 1_000 / TWILIO_SR
                if unmarked_ms >= MARK_EVERY_MS:
                    await send_mark(unmarked_ms)
                    unmarked_ms = 0.0
                next_at += FRAME_MS / 1_000
                delay = next_at - loop.time() - OUTBOUND_LEAD_MS / 1_000
                if delay > 0:
                    await asyncio.sleep(delay)

        async def send_twilio() -> None:
            async for event in session.events():
                if hooks.on_event is not None:
                    await hooks.on_event(event)
                if event.type == "audio" and event.audio is not None:
                    queue_outbound(audio.dialt_to_twilio(float32_to_pcm16(event.audio)))
                elif event.type == "interrupted":
                    hard_clear = bool(event.data.get("clear"))
                    if hard_clear:
                        outbound.clear()
                    remaining_ms, discarded_ms = ledger.interruption(hard_clear=hard_clear)
                    if hard_clear:
                        await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                    await session.send_client_event(
                        "playback_stopped", remaining_ms=round(remaining_ms),
                        discarded_ms=round(discarded_ms), barge_seq=event.data.get("barge_seq"))
                elif event.type == "canceled":
                    outbound.clear()
                    ledger.clear()
                    await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                elif event.type == "tool_call":
                    tool_id = str(event.data["id"])
                    tool_tasks[tool_id] = asyncio.create_task(run_tool(
                        tool_id, str(event.data["name"]), dict(event.data.get("args") or {})))
                elif event.type == "tool_cancel":
                    task = tool_tasks.get(str(event.data.get("id")))
                    if task is not None:
                        task.cancel()

            upstream_ended.set()
            queue_outbound(audio.dialt_to_twilio(b"", final=True))
            try:
                async with asyncio.timeout(PLAYBACK_DRAIN_TIMEOUT_S):
                    while outbound or not ledger.empty():
                        ledger.changed.clear()
                        if not outbound and ledger.empty():
                            break
                        await ledger.changed.wait()
            except TimeoutError:
                logger.warning("Twilio playback drain timed out call_sid=%s pending_ms=%.0f",
                               call_sid, ledger.pending_ms())

        async def host_end_call() -> None:
            assert hooks.end_call is not None
            await hooks.end_call.wait()
            await session.close()

        receive_task = asyncio.create_task(receive_twilio())
        send_task = asyncio.create_task(send_twilio())
        upstream_end_task = asyncio.create_task(upstream_ended.wait())
        # Only these three decide when the bridge ends. Helpers (the pacer, the host's end_call
        # watcher) are cancelled at the end and never part of the wait set: a helper finishing
        # is not the call ending.
        bridge_tasks = (receive_task, send_task, upstream_end_task)
        helper_tasks: list[asyncio.Task[None]] = [asyncio.create_task(pace_outbound())]
        if hooks.end_call is not None:
            helper_tasks.append(asyncio.create_task(host_end_call()))
        try:
            done, _ = await asyncio.wait(bridge_tasks, return_when=asyncio.FIRST_COMPLETED)
            try:
                if send_task in done:
                    send_task.result()
                elif upstream_end_task in done:
                    await send_task
                else:
                    receive_task.result()
            except DialtError as exc:
                # The caller hung up while a send was in flight: the session is closed. Any
                # other SDK error is a real failure.
                if exc.code != "connection_closed":
                    raise
        finally:
            for task in (*bridge_tasks, *helper_tasks):
                task.cancel()
            await asyncio.gather(*bridge_tasks, *helper_tasks, return_exceptions=True)
            active_tool_tasks = list(tool_tasks.values())
            for task in active_tool_tasks:
                task.cancel()
            await asyncio.gather(*active_tool_tasks, return_exceptions=True)
