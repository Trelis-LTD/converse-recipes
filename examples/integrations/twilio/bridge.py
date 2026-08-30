"""Inbound Twilio Media Streams bridge for the Trelis Converse API.

Twilio sends 8 kHz mu-law audio; Converse receives PCM16 and returns Float32 audio at 16 kHz.
The bridge also maps Twilio mark/clear playback state onto Converse's playback contract so
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

import numpy as np
from converse_sdk import ConverseMode, ConverseSession, float32_to_pcm16
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

logger = logging.getLogger("converse_twilio")
MULAW_CHUNK_BYTES = 160  # 20 ms; bounds hard-clear playback accounting error.
TWILIO_SR = 8_000


@dataclass(frozen=True)
class Settings:
    converse_api_key: str
    twilio_auth_token: str
    public_base_url: str
    twilio_account_sid: str | None = None
    human_handoff_url: str | None = None
    converse_url: str = "wss://converse.trelis.com/ws"
    voice: str | None = None
    instructions: str | None = None
    greeting: str | bool | None = None

    @classmethod
    def from_env(cls) -> Settings:
        missing = [
            name
            for name in ("CONVERSE_API_KEY", "TWILIO_AUTH_TOKEN", "PUBLIC_BASE_URL")
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

        raw_greeting = os.environ.get("CONVERSE_GREETING")
        greeting: str | bool | None
        if raw_greeting is None:
            greeting = None
        elif raw_greeting.strip().lower() in {"false", "off", "none"}:
            greeting = False
        else:
            greeting = raw_greeting

        return cls(
            converse_api_key=os.environ["CONVERSE_API_KEY"],
            twilio_auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            public_base_url=os.environ["PUBLIC_BASE_URL"].rstrip("/"),
            twilio_account_sid=twilio_account_sid,
            human_handoff_url=human_handoff_url,
            converse_url=os.environ.get("CONVERSE_URL", "wss://converse.trelis.com/ws"),
            voice=os.environ.get("CONVERSE_VOICE") or None,
            instructions=os.environ.get("CONVERSE_INSTRUCTIONS") or None,
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

    def add(self, duration_ms: float) -> str:
        self._sequence += 1
        name = f"converse-{self._sequence}"
        self._pending[name] = duration_ms
        return name

    def acknowledge(self, name: str) -> None:
        self._pending.pop(name, None)

    def pending_ms(self) -> float:
        return sum(self._pending.values())

    def clear(self) -> float:
        discarded_ms = self.pending_ms()
        self._pending.clear()
        return discarded_ms

    def interruption(self, *, hard_clear: bool) -> tuple[float, float]:
        if hard_clear:
            return 0, self.clear()
        return self.pending_ms(), 0


def mulaw_8k_to_pcm16_16k(data: bytes) -> bytes:
    """Decode G.711 mu-law and duplicate samples to Converse's 16 kHz input rate."""
    if not data:
        return b""
    encoded = np.frombuffer(data, dtype=np.uint8)
    value = np.bitwise_not(encoded).astype(np.int32)
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    magnitude = ((mantissa << 3) + 0x84) << exponent
    decoded = magnitude - 0x84
    decoded = np.where(sign != 0, -decoded, decoded).astype("<i2")
    return np.repeat(decoded, 2).astype("<i2", copy=False).tobytes()


def pcm16_16k_to_mulaw_8k(data: bytes) -> bytes:
    """Downsample PCM16 to 8 kHz and encode G.711 mu-law for Twilio."""
    if not data:
        return b""
    samples = np.frombuffer(data, dtype="<i2").astype(np.int32)
    if len(samples) % 2:
        samples = np.pad(samples, (0, 1), mode="edge")
    samples = (samples[0::2] + samples[1::2]) // 2

    sign = np.where(samples < 0, 0x80, 0).astype(np.int32)
    magnitude = np.minimum(np.abs(samples), 32635) + 0x84
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = np.bitwise_not(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8).tobytes()


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


app = FastAPI(title="Converse Twilio bridge")


@app.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Return TwiML that starts one inbound bidirectional Media Stream."""
    settings = get_settings()
    form = dict(await request.form())
    signature = request.headers.get("x-twilio-signature")
    if not _signature_is_valid(settings.http_url("/voice"), form, signature):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")

    stream_url = html.escape(settings.websocket_url("/media"), quote=True)
    twiml = f'<Response><Connect><Stream url="{stream_url}" /></Connect></Response>'
    return Response(twiml, media_type="application/xml")


@app.websocket("/media")
async def media_stream(websocket: WebSocket) -> None:
    """Bridge one Twilio call to one Converse session."""
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
        await _run_bridge(websocket, stream_sid, call_sid, settings)
    except (KeyError, ValueError, json.JSONDecodeError):
        await websocket.close(code=1003)
    except WebSocketDisconnect:
        return


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
    mode = ConverseMode(
        voice=settings.voice,
        instructions=settings.instructions,
        greeting=settings.greeting,
        tools=tool_manifest() or None,
    )

    async with await ConverseSession.connect(
        settings.converse_url,
        session_id=call_sid[:64],
        api_key=settings.converse_api_key,
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
                        await session.send_audio(mulaw_8k_to_pcm16_16k(mulaw))
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
                    logger.exception("Converse tool %s failed", name)
                    result = {"error": "tool_failed"}
                    outcome = "failed"
                    verified = False
                await session.send_tool_result(
                    tool_id, result, outcome=outcome, verified=verified)
            except asyncio.CancelledError:
                return
            finally:
                tool_tasks.pop(tool_id, None)

        async def send_twilio() -> None:
            async for event in session.events():
                if event.type == "audio" and event.audio is not None:
                    mulaw = pcm16_16k_to_mulaw_8k(float32_to_pcm16(event.audio))
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

        tasks = {
            asyncio.create_task(receive_twilio()),
            asyncio.create_task(send_twilio()),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        active_tool_tasks = list(tool_tasks.values())
        for task in active_tool_tasks:
            task.cancel()
        await asyncio.gather(*active_tool_tasks, return_exceptions=True)
        for task in done:
            task.result()
