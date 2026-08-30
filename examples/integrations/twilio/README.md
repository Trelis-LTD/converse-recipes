# Converse + Twilio inbound bridge

This runnable integration connects an inbound Twilio phone call to one Converse session. Twilio
owns the phone number and call; this bridge owns deployment, audio transport and application
tools; Converse owns the realtime voice conversation.

It uses Twilio bidirectional Media Streams. It does not require Twilio ConversationRelay.

## Run it

From this directory:

```sh
cp env.example .env
uv sync --frozen
uv run uvicorn bridge:app --env-file .env --host 0.0.0.0 --port 8000
```

Put the app behind public HTTPS and set `PUBLIC_BASE_URL` to that exact external origin, for
example `https://voice.example.com`. Configure the Twilio phone number's incoming Voice webhook
as `POST https://voice.example.com/voice`.

The webhook returns `<Connect><Stream>`. The bridge:

- verifies Twilio signatures for the HTTP and WebSocket requests;
- converts Twilio's 8 kHz G.711 mu-law audio to Converse's 16 kHz wire format;
- converts Converse output back to Twilio audio; and
- maps Twilio `mark` and `clear` playback state to Converse interruption events.

## Add application tools

Edit `tool_manifest()` and `execute_tool()` in `bridge.py`. Keep service credentials and effects
inside the bridge. Only tool schemas, bounded arguments and bounded results should cross the
Converse session.

## Optional human handoff

Set both variables below to expose the permission-gated `request_human_handoff` tool:

```sh
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_HUMAN_HANDOFF_URL=https://voice.example.com/handoff
```

After the caller approves a handoff, the bridge redirects the active Twilio call to the configured
HTTPS URL. That customer-owned endpoint returns the TwiML for the real destination, such as a
`<Dial>`, queue, conference, Flex flow or TaskRouter workflow. The destination is configuration
and is never supplied by the model.

This is a cold-transfer reference, not a generic contact-center implementation. The tool provides
a reason and concise summary to `execute_tool()`. Persist them by `CallSid` before redirecting if
the receiving agent needs context. Keep the handoff endpoint authenticated according to Twilio's
webhook-security guidance.

## Test it

```sh
uv run pytest -q
```

The tests are offline and require no Converse or Twilio credentials.

This is an inbound reference integration, not a dialer. Converse's acceptable-use policy prohibits
outbound robocalling.
