"""Inbound Twilio Media Streams bridge for Dialt.

Twilio sends 8 kHz mu-law audio; Dialt receives PCM16 and returns Float32 audio at 16 kHz.
The bridge also maps Twilio mark/clear playback state onto Dialt's playback contract so
barge-in truncates conversation context to what the caller could actually hear.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from dialt import DialtMode, DialtSession, float32_to_pcm16
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from telephony_audio import TelephonyAudioBridge

logger = logging.getLogger("dialt_twilio")
MULAW_CHUNK_BYTES = 160  # 20 ms; bounds hard-clear playback accounting error.
TWILIO_SR = 8_000
PLAYBACK_DRAIN_TIMEOUT_S = 10


@dataclass(frozen=True)
class Settings:
    dialt_api_key: str
    twilio_auth_token: str
    public_base_url: str
    twilio_account_sid: str | None = None
    human_handoff_url: str | None = None
    dialt_url: str = "wss://dialt.com/ws"
    voice: str | None = None
    instructions: str | None = None
    greeting: str | bool | None = None

    @classmethod
    def from_env(cls) -> Settings:
        missing = [
            name
            for name in ("DIALT_API_KEY", "TWILIO_AUTH_TOKEN", "PUBLIC_BASE_URL")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        human_handoff_url = os.environ.get("TWILIO_HUMAN_HANDOFF_URL", "").strip() or None
        twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip() or None
        if human_handoff_url:
            if not twilio_account_sid:
                raise RuntimeError(
                    "TWILIO_ACCOUNT_SID is required when TWILIO_HUMAN_HANDOFF_URL is set"
                )
            parsed_handoff_url = urlsplit(human_handoff_url)
            if parsed_handoff_url.scheme != "https" or not parsed_handoff_url.netloc:
                raise RuntimeError(
                    "TWILIO_HUMAN_HANDOFF_URL must be an absolute https URL"
                )

        raw_greeting = os.environ.get("DIALT_GREETING")
        greeting: str | bool | None
        if raw_greeting is None:
            greeting = None
        elif raw_greeting.strip().lower() in {"false", "off", "none"}:
            greeting = False
        else:
            greeting = raw_greeting

        return cls(
            dialt_api_key=os.environ["DIALT_API_KEY"],
            twilio_auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            public_base_url=os.environ["PUBLIC_BASE_URL"].rstrip("/"),
            twilio_account_sid=twilio_account_sid,
            human_handoff_url=human_handoff_url,
            dialt_url=os.environ.get("DIALT_URL", "wss://dialt.com/ws"),
            voice=os.environ.get("DIALT_VOICE") or None,
            instructions=os.environ.get("DIALT_INSTRUCTIONS") or None,
            greeting=greeting,
        )

    def http_url(self, path: str) -> str:
        return f"{self.public_base_url}{path}"

    def websocket_url(self, path: str) -> str:
        base = self.public_base_url
        if base.startswith("https://"):
            base = f"wss://{base.removeprefix('https://')}"
        elif base.startswith("http://"):
            base = f"ws://{base.removeprefix('http://')}"
        return f"{base}{path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


class PlaybackLedger:
    """Tracks audio sent to Twilio but not yet acknowledged as played."""

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
        if hard_clear:
            return 0, self.clear()
        return self.pending_ms(), 0



def tool_manifest() -> list[dict[str, Any]]:
    """Declare application tools here; keep credentials and execution in your own service."""
    if not get_settings().human_handoff_url:
        return []
    return [{
        "name": "request_human_handoff",
        "description": (
            "Transfer the active call to the customer's configured human-support workflow. "
            "Use when the caller clearly asks for a human or the application instructions "
            "require escalation. The host, never the model, owns the destination."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this call needs human support.",
                },
                "summary": {
                    "type": "string",
                    "description": "A concise summary for the human-support workflow.",
                },
            },
            "required": ["reason", "summary"],
            "additionalProperties": False,
        },
        "requires_permission": True,
        "expected_duration": "seconds",
        "status_label": "human handoff",
    }]


async def execute_tool(call_sid: str, name: str, args: dict[str, Any]) -> Any:
    """Replace with calls into your application. Undeclared tools never reach this function."""
    if name == "request_human_handoff":
        settings = get_settings()
        if not settings.human_handoff_url or not settings.twilio_account_sid:
            raise RuntimeError("Human handoff is not configured")
        for field in ("reason", "summary"):
            value = args.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"request_human_handoff requires a non-empty {field}")

        def redirect_call() -> None:
            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            client.calls(call_sid).update(
                url=settings.human_handoff_url,
                method="POST",
            )

        await asyncio.to_thread(redirect_call)
        return {"handoff_requested": True}
    raise RuntimeError(f"No handler configured for tool {name!r}")


def _signature_is_valid(
    url: str, params: dict[str, Any], signature: str | None
) -> bool:
    if not signature:
        return False
    return RequestValidator(get_settings().twilio_auth_token).validate(
        url, params, signature
    )


app = FastAPI(title="Dialt Twilio bridge")


@app.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Return TwiML that starts one inbound bidirectional Media Stream."""
    settings = get_settings()
    form = dict(await request.form())
    signature = request.headers.get("x-twilio-signature")
    if not _signature_is_valid(settings.http_url("/voice"), form, signature):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")

    stream_url = html.escape(settings.websocket_url("/media"), quote=True)
    status_url = html.escape(settings.http_url("/stream-status"), quote=True)
    twiml = (
        "<Response><Connect>"
        f'<Stream url="{stream_url}" statusCallback="{status_url}" '
        'statusCallbackMethod="POST" />'
        "</Connect><Hangup /></Response>"
    )
    return Response(twiml, media_type="application/xml")


@app.post("/stream-status", status_code=204)
async def stream_status(request: Request) -> Response:
    """Log Twilio's authenticated stream lifecycle events."""
    settings = get_settings()
    form = dict(await request.form())
    signature = request.headers.get("x-twilio-signature")
    if not _signature_is_valid(settings.http_url("/stream-status"), form, signature):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")
    logger.info(
        "Twilio stream status call_sid=%s stream_sid=%s event=%s error=%s",
        form.get("CallSid"),
        form.get("StreamSid"),
        form.get("StreamEvent"),
        form.get("StreamError") or "",
    )
    return Response(status_code=204)


@app.websocket("/media")
async def media_stream(websocket: WebSocket) -> None:
    """Bridge one Twilio call to one Dialt session."""
    settings = get_settings()
    signature = websocket.headers.get("x-twilio-signature")
    if not _signature_is_valid(settings.websocket_url("/media"), {}, signature):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        start = await _receive_start(websocket)
        stream_sid = str(start["streamSid"])
        call_sid = str(start.get("callSid") or stream_sid)
        logger.info(
            "Twilio bridge started call_sid=%s stream_sid=%s", call_sid, stream_sid
        )
        await _run_bridge(websocket, stream_sid, call_sid, settings)
    except (KeyError, ValueError, json.JSONDecodeError):
        logger.exception("Twilio sent an invalid media message")
        await _close_websocket(websocket, 1003)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Twilio bridge failed")
        await _close_websocket(websocket, 1011)
    else:
        logger.info(
            "Twilio bridge completed call_sid=%s stream_sid=%s", call_sid, stream_sid
        )
        await _close_websocket(websocket, 1000)

async def _close_websocket(websocket: WebSocket, code: int) -> None:
    try:
        await websocket.close(code=code)
    except RuntimeError:
        pass




async def _receive_start(websocket: WebSocket) -> dict[str, Any]:
    while True:
        message = json.loads(await websocket.receive_text())
        event = message.get("event")
        if event == "start":
            start = message.get("start")
            if not isinstance(start, dict) or not start.get("streamSid"):
                raise ValueError("Twilio start message missing streamSid")
            return start
        if event == "stop":
            raise WebSocketDisconnect()


async def _run_bridge(
    websocket: WebSocket,
    stream_sid: str,
    call_sid: str,
    settings: Settings,
) -> None:
    ledger = PlaybackLedger()
    audio = TelephonyAudioBridge()
    upstream_ended = asyncio.Event()
    mode = DialtMode(
        voice=settings.voice,
        instructions=settings.instructions,
        greeting=settings.greeting,
        tools=tool_manifest() or None,
    )

    async with await DialtSession.connect(
        settings.dialt_url,
        session_id=call_sid[:64],
        api_key=settings.dialt_api_key,
        mode=mode,
    ) as session:
        tool_tasks: dict[str, asyncio.Task[None]] = {}

        async def receive_twilio() -> None:
            async for raw in websocket.iter_text():
                message = json.loads(raw)
                event = message.get("event")
                if event == "media":
                    payload = message.get("media", {}).get("payload")
                    if isinstance(payload, str):
                        mulaw = base64.b64decode(payload, validate=True)
                        pcm = audio.twilio_to_dialt(mulaw)
                        if pcm and not upstream_ended.is_set():
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
                    result = await execute_tool(call_sid, name, args)
                    outcome = "succeeded"
                    verified = True
                except Exception:
                    logger.exception("Dialt tool %s failed", name)
                    result = {"error": "tool_failed"}
                    outcome = "failed"
                    verified = False
                await session.send_tool_result(
                    tool_id, result, outcome=outcome, verified=verified)
            except asyncio.CancelledError:
                return
            finally:
                tool_tasks.pop(tool_id, None)

        async def send_mulaw(mulaw: bytes) -> None:
            for offset in range(0, len(mulaw), MULAW_CHUNK_BYTES):
                chunk = mulaw[offset : offset + MULAW_CHUNK_BYTES]
                await websocket.send_json(
                    {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": base64.b64encode(chunk).decode("ascii")
                        },
                    }
                )
                mark_name = ledger.add(len(chunk) * 1000 / TWILIO_SR)
                await websocket.send_json(
                    {
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": mark_name},
                    }
                )


        async def send_twilio() -> None:

            async for event in session.events():
                if event.type == "audio" and event.audio is not None:
                    pcm = float32_to_pcm16(event.audio)
                    await send_mulaw(audio.dialt_to_twilio(pcm))
                elif event.type == "interrupted":
                    hard_clear = bool(event.data.get("clear"))
                    remaining_ms, discarded_ms = ledger.interruption(
                        hard_clear=hard_clear
                    )
                    if hard_clear:
                        await websocket.send_json(
                            {"event": "clear", "streamSid": stream_sid}
                        )
                    await session.send_client_event(
                        "playback_stopped",
                        remaining_ms=round(remaining_ms),
                        discarded_ms=round(discarded_ms),
                        barge_seq=event.data.get("barge_seq"),
                    )
                elif event.type == "canceled":
                    ledger.clear()
                    await websocket.send_json(
                        {"event": "clear", "streamSid": stream_sid}
                    )
                elif event.type == "tool_call":
                    tool_id = str(event.data["id"])
                    task = asyncio.create_task(
                        run_tool(
                            tool_id,
                            str(event.data["name"]),
                            dict(event.data.get("args") or {}),
                        )
                    )
                    tool_tasks[tool_id] = task
                elif event.type == "tool_cancel":
                    task = tool_tasks.get(str(event.data.get("id")))
                    if task is not None:
                        task.cancel()

            upstream_ended.set()
            await send_mulaw(audio.dialt_to_twilio(b"", final=True))
            try:
                async with asyncio.timeout(PLAYBACK_DRAIN_TIMEOUT_S):
                    while not ledger.empty():
                        ledger.changed.clear()
                        if ledger.empty():
                            break
                        await ledger.changed.wait()
            except TimeoutError:
                logger.warning(
                    "Twilio playback drain timed out call_sid=%s pending_ms=%.0f",
                    call_sid,
                    ledger.pending_ms(),
                )
        receive_task = asyncio.create_task(receive_twilio())
        send_task = asyncio.create_task(send_twilio())
        upstream_end_task = asyncio.create_task(upstream_ended.wait())
        bridge_tasks = (receive_task, send_task, upstream_end_task)
        try:
            done, _ = await asyncio.wait(
                bridge_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            if send_task in done:
                send_task.result()
            elif upstream_end_task in done:
                await send_task
            else:
                receive_task.result()
        finally:
            for task in bridge_tasks:
                task.cancel()
            await asyncio.gather(*bridge_tasks, return_exceptions=True)

            active_tool_tasks = list(tool_tasks.values())
            for task in active_tool_tasks:
                task.cancel()
            await asyncio.gather(*active_tool_tasks, return_exceptions=True)
