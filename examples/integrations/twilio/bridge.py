"""Inbound Twilio Media Streams bridge for Dialt, built on `dialt_recipes.twilio`.

Twilio owns the phone number and call; `dialt_recipes.twilio` owns the transport (paced audio,
playback accounting, barge-in, tool dispatch); this file owns what is yours: configuration,
the HTTP routes Twilio calls, the agent's mode, and what each tool does.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from dialt import DialtMode
from dialt_recipes.twilio import (
    BridgeHooks,
    TwilioBridgeSettings,
    connect_stream_twiml,
    receive_stream_start,
    run_call_bridge,
    twilio_signature_is_valid,
)
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from twilio.rest import Client

logger = logging.getLogger("dialt_twilio")


@dataclass(frozen=True)
class Settings(TwilioBridgeSettings):
    twilio_account_sid: str | None = None
    human_handoff_url: str | None = None
    voice: str | None = None
    instructions: str | None = None
    greeting: str | bool | None = None

    @classmethod
    def from_env(cls) -> Settings:
        missing = [name for name in ("DIALT_API_KEY", "TWILIO_AUTH_TOKEN", "PUBLIC_BASE_URL")
                   if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

        human_handoff_url = os.environ.get("TWILIO_HUMAN_HANDOFF_URL", "").strip() or None
        twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip() or None
        if human_handoff_url:
            if not twilio_account_sid:
                raise RuntimeError("TWILIO_ACCOUNT_SID is required when TWILIO_HUMAN_HANDOFF_URL is set")
            parsed = urlsplit(human_handoff_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise RuntimeError("TWILIO_HUMAN_HANDOFF_URL must be an absolute https URL")

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
            dialt_url=os.environ.get("DIALT_URL", "wss://dialt.com/ws"),
            twilio_account_sid=twilio_account_sid,
            human_handoff_url=human_handoff_url,
            voice=os.environ.get("DIALT_VOICE") or None,
            instructions=os.environ.get("DIALT_INSTRUCTIONS") or None,
            greeting=greeting,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


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
                "reason": {"type": "string", "description": "Why this call needs human support."},
                "summary": {"type": "string",
                            "description": "A concise summary for the human-support workflow."},
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
            client.calls(call_sid).update(url=settings.human_handoff_url, method="POST")

        await asyncio.to_thread(redirect_call)
        return {"handoff_requested": True}
    raise RuntimeError(f"No handler configured for tool {name!r}")


def _signature_is_valid(url: str, params: dict[str, Any], signature: str | None) -> bool:
    return twilio_signature_is_valid(get_settings().twilio_auth_token, url, params, signature)


app = FastAPI(title="Dialt Twilio bridge")


@app.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Return TwiML that starts one inbound bidirectional Media Stream."""
    settings = get_settings()
    form = dict(await request.form())
    if not _signature_is_valid(settings.http_url("/voice"), form, request.headers.get("x-twilio-signature")):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")
    return Response(connect_stream_twiml(settings), media_type="application/xml")


@app.post("/stream-status", status_code=204)
async def stream_status(request: Request) -> Response:
    """Log Twilio's authenticated stream lifecycle events."""
    settings = get_settings()
    form = dict(await request.form())
    if not _signature_is_valid(settings.http_url("/stream-status"), form,
                               request.headers.get("x-twilio-signature")):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")
    error = form.get("StreamError") or ""
    (logger.warning if error else logger.info)(
        "Twilio stream status call_sid=%s stream_sid=%s event=%s error=%s",
        form.get("CallSid"), form.get("StreamSid"), form.get("StreamEvent"), error)
    return Response(status_code=204)


@app.websocket("/media")
async def media_stream(websocket: WebSocket) -> None:
    """Bridge one Twilio call to one Dialt session."""
    settings = get_settings()
    if not _signature_is_valid(settings.websocket_url("/media"), {},
                               websocket.headers.get("x-twilio-signature")):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        start = await receive_stream_start(websocket)
        stream_sid = str(start["streamSid"])
        call_sid = str(start.get("callSid") or stream_sid)
        logger.info("Twilio bridge started call_sid=%s stream_sid=%s", call_sid, stream_sid)
        await _run_bridge(websocket, stream_sid, call_sid, settings)
    except (KeyError, ValueError):
        logger.exception("Twilio sent an invalid media message")
        await _close_websocket(websocket, 1003)
    except (WebSocketDisconnect, ConnectionError):
        return
    except Exception:
        logger.exception("Twilio bridge failed")
        await _close_websocket(websocket, 1011)
    else:
        logger.info("Twilio bridge completed call_sid=%s stream_sid=%s", call_sid, stream_sid)
        await _close_websocket(websocket, 1000)


async def _close_websocket(websocket: WebSocket, code: int) -> None:
    try:
        await websocket.close(code=code)
    except RuntimeError:
        pass


async def _run_bridge(websocket: WebSocket, stream_sid: str, call_sid: str, settings: Settings) -> None:
    mode = DialtMode(voice=settings.voice, instructions=settings.instructions,
                     greeting=settings.greeting, tools=tool_manifest() or None)

    async def run_tool(name: str, args: dict[str, Any]) -> Any:
        return await execute_tool(call_sid, name, args)

    await run_call_bridge(websocket, stream_sid, call_sid, settings=settings, mode=mode,
                          hooks=BridgeHooks(execute_tool=run_tool))
