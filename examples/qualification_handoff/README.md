# Qualification and specialist handoff

This recipe is a domain-neutral starting point for a voice agent that:

1. Collects a small set of qualification fields naturally.
2. Accepts corrections without restarting the flow.
3. Reflects the collected details and asks for explicit consent.
4. Requests a customer-owned specialist handoff and reports the real result.

It does not originate calls or choose a handoff destination. Your application owns dialing, routing, availability, and the final transfer.

## Exercise the conversation first

Run all four cases in text, then voice. They cover an accepted handoff, a corrected answer, a declined handoff, and an unavailable specialist.

```sh
uv sync --frozen
cp .env.example .env
uv run converse-sim examples/qualification_handoff/evals --modality text
uv run converse-sim examples/qualification_handoff/evals --modality voice
```

Local runs apply the deterministic checks. The behavioral `judge` checks run when the same JSON cases are pushed to hosted evals:

```sh
uv run converse-evals push examples/qualification_handoff/evals --modality text --wait
```

## Connect application behavior

[`workflow.py`](workflow.py) shows the two application-owned operations. `QualificationState.record` stores or corrects an answer. `QualificationState.start_handoff` rejects an incomplete qualification and makes repeated requests idempotent.

Run the accepted case with those Python callbacks:

```sh
uv run python -u examples/qualification_handoff/with_callbacks.py
```

Replace `QualificationState` with your database and routing service. Keep the tool names and overall contract stable while adapting the field enum, descriptions, and application handlers for your domain.

## Add Twilio after the evals pass

Reuse the maintained [`examples/integrations/twilio`](../integrations/twilio) Media Streams bridge instead of creating another transport:

1. Add these two tools to its `tool_manifest`.
2. Keep one qualification state object per Twilio `CallSid`.
3. Route tool calls to your record and handoff handlers.
4. Keep `CallSid` as the Converse session ID and on every application event.
5. Log Twilio call and transfer events under the same ID.

For an outbound workflow, the customer-owned dialer originates the call. Once answered, return the same bidirectional Media Streams TwiML used by the integration. The bridge connects the live call to Converse; your application still controls pacing, destinations, and transfer policy.

Start with simulated evals, including hosted judge checks, then use a controlled Twilio sandbox pool, then compare a small live cohort against the existing flow. Promote repeated integration friction into SDK or API changes only after the recipe exposes it.
