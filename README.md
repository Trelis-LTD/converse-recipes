# Dialt recipes

Runnable patterns built entirely from the public
[`dialt-sdk`](https://pypi.org/project/dialt-sdk/) and
[`@dialt/sdk`](https://www.npmjs.com/package/@dialt/sdk) surfaces.

The boundary is deliberate: Dialt owns conversation; recipes own application policy and
orchestration. There is no parallel simulator client and no eval-only conversation engine. A
simulated user is another Dialt session. Text and voice select different I/O on the same
session primitive.

## Integrations

[`examples/integrations/twilio`](examples/integrations/twilio) is the maintained inbound Twilio
Media Streams bridge. It is a standalone uv project with configuration instructions and offline
tests, including the optional customer-controlled human handoff.

```sh
cd examples/integrations/twilio
cp env.example .env
uv sync --frozen
uv run pytest -q
```

## Install

Version 0.2 uses the `dialt_recipes` import namespace, `DIALT_*` environment names, and
`dialt-guided`, `dialt-sim`, and `dialt-evals` commands. The pre-0.2 `converse_recipes`
namespace and legacy names were removed rather than kept as parallel compatibility paths.


```sh
uv sync
cp .env.example .env
```

## Qualification and specialist handoff

[`examples/qualification_handoff`](examples/qualification_handoff) collects configurable qualification fields, handles corrections, obtains explicit consent, and calls a customer-owned specialist handoff. It includes text and voice eval cases plus Python callback seams, and reuses the maintained Twilio bridge for live calls.

## Guided customer-research assistant

[`examples/guided_customer_research`](examples/guided_customer_research) is a non-trivial guided
interview. A `ConversationPlan` declares the evidence to collect; a normal client tool records
answers. The model still handles wording, clarification, corrections, order and transitions
naturally.

Run the terminal version in text mode:

```sh
uv run dialt-guided examples/guided_customer_research/plan.json
```

The browser example supports text and voice with the same plan and shows collected evidence live.
Serve the repository directory, open the example, and paste a short-lived scoped session key (not
a persistent `ck_` key).

## Evals: run cases locally, then push them

A case is one JSON file, the same document the hosted evals API accepts: `name`, `starter`,
`target` (`instructions`, `tools`, `end_call`), `simulator` (`instructions`), `fixtures`, `checks`
and `limits`. [`examples/simulations/appointment_booking.json`](examples/simulations/appointment_booking.json)
is a complete one. Field reference: the [evals guide](https://dialt.com/docs/api/evals/).

The agent ends a call by calling the managed `end_call` tool, which `target.end_call` (default
true) gives it; set it to false for an agent that must never hang up. The simulated user always
has it. Either call is recorded as a tool call named `end_call`, so `completed` means the agent
ended the call and `simulator_ended` means the simulated user did.

Run a case, or every case in a directory, locally. Two Dialt sessions talk to each other: the
target agent and a simulated caller. Text forwards committed utterances; voice gives each session
a virtual microphone that streams for the whole call (the other side's audio at real time, line
noise in between, exactly as a phone line would) and never touches a speaker or microphone.

```sh
uv run dialt-sim examples/simulations/ --modality text
uv run dialt-sim examples/simulations/appointment_booking.json --modality voice
```

The deterministic checks (`contains`, `not_contains`, `regex`, `tool_called`,
`fixture_complete`, `max_turns`) run here with the same rules as hosted runs; `judge` checks
need the judge model and are reported as skipped locally. An undeclared tool fails closed, as it
does hosted, so a local pass means the case declares every tool the agent uses.

Push the same files and run them hosted. Cases are matched by name, so a re-push updates the
hosted case instead of duplicating it; the run appears on the
[Evals dashboard](https://dialt.com/evals).

```sh
uv run dialt-evals push examples/simulations/ --modality text --wait
```

With `--wait` the command exits non-zero unless the run passes, so it can gate a CI job in your
agent's repository: put `DIALT_API_KEY` in a secret and run it on every change to the agent or
its cases. Without `--wait` it returns as soon as the run is queued.

To answer tool calls with your own Python instead of fixed fixtures, build the case in code; see
[`examples/simulations/with_callbacks.py`](examples/simulations/with_callbacks.py). Hosted runs
accept fixed and `field_store` fixtures only.

## Tests

```sh
uv run pytest
```

The default suite is offline and secret-free. The manual GitHub workflow runs a bounded live text
or blackholed-voice smoke when `DIALT_API_KEY` is configured.

Licensed under the [Apache License 2.0](LICENSE).
