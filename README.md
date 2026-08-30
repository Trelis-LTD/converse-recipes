# Converse recipes

Runnable patterns built entirely from the public
[`converse-sdk`](https://pypi.org/project/converse-sdk/) and
[`@trelis/converse`](https://www.npmjs.com/package/@trelis/converse) surfaces.

The boundary is deliberate: Converse owns conversation; recipes own application policy and
orchestration. There is no parallel simulator client and no eval-only conversation engine. A
simulated user is another Converse session. Text and voice select different I/O on the same
session primitive.

## Install

```sh
uv sync
cp .env.example .env
```

Until the text SDK releases land on PyPI/npm, this repository pins the reviewed public SDK commit
used by the reference implementation. The Python and browser examples therefore remain runnable
without publishing a prerelease package.

## Guided customer-research assistant

[`examples/guided_customer_research`](examples/guided_customer_research) is a non-trivial guided
interview. A `ConversationPlan` declares the evidence to collect; a normal client tool records
answers. The model still handles wording, clarification, corrections, order and transitions
naturally.

Run the terminal version in text mode:

```sh
uv run converse-guided examples/guided_customer_research/plan.json
```

The browser example supports text and voice with the same plan and shows collected evidence live.
Serve the repository directory, open the example, and paste a short-lived scoped session key (not
a persistent `ck_` key).

## Evals: run cases locally, then push them

A case is one JSON file, the same document the hosted evals API accepts: `name`, `starter`,
`target` (`instructions`, `tools`), `simulator` (`instructions`), `fixtures`, `checks` and
`limits`. [`examples/simulations/appointment_booking.json`](examples/simulations/appointment_booking.json)
is a complete one. Field reference: the [evals guide](https://converse.trelis.com/docs/api/evals/).

Run a case, or every case in a directory, locally. Two Converse sessions talk to each other: the
target agent and a simulated caller. Text forwards committed utterances; voice cross-pipes the
audio between the sessions and never touches a speaker or microphone.

```sh
uv run converse-sim examples/simulations/ --modality text
uv run converse-sim examples/simulations/appointment_booking.json --modality voice
```

The deterministic checks (`contains`, `not_contains`, `regex`, `tool_called`,
`fixture_complete`, `max_turns`) run here with the same rules as hosted runs; `judge` checks
need the judge model and are reported as skipped locally. An undeclared tool fails closed, as it
does hosted, so a local pass means the case declares every tool the agent uses.

Push the same files and run them hosted. Cases are matched by name, so a re-push updates the
hosted case instead of duplicating it; the run appears on the
[Evals dashboard](https://converse.trelis.com/evals).

```sh
uv run converse-evals push examples/simulations/ --modality text --wait
```

To answer tool calls with your own Python instead of fixed fixtures, build the case in code; see
[`examples/simulations/with_callbacks.py`](examples/simulations/with_callbacks.py). Hosted runs
accept fixed and `field_store` fixtures only.

## Tests

```sh
uv run pytest
```

The default suite is offline and secret-free. The manual GitHub workflow runs a bounded live text
or blackholed-voice smoke when `CONVERSE_API_KEY` is configured.

Licensed under the [Apache License 2.0](LICENSE).
