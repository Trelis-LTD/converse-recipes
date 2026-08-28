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

## Converse-vs-Converse simulation

[`examples/simulations/appointment_booking.json`](examples/simulations/appointment_booking.json)
runs a support agent against a Converse simulated caller. The target may use fixed fixtures; the
simulator receives no tools. Text forwards committed utterances. Voice buffers complete turns,
cross-pipes their 16 kHz PCM in memory, and sends it to no speaker, sound device or file output—a
blackholed acoustic test.
The example deterministically checks the committed target transcript and target tool calls, so the
same case produces an explicit report instead of relying on a model to grade itself.
It stops cleanly when every declared expectation is satisfied at a committed target turn; guard
terminations and missing expectations still fail.

```sh
uv run converse-sim examples/simulations/appointment_booking.json --modality text
uv run converse-sim examples/simulations/appointment_booking.json --modality voice
```

Use Python callbacks instead of JSON fixture values for local integration tests; see
[`examples/simulations/with_callbacks.py`](examples/simulations/with_callbacks.py). Hosted evals
intentionally accept fixed fixtures only and fail closed on any undeclared tool.

To attach local results to the Converse eval dashboard, create the run with
`execution: "local"` and pass `--report-base-url`, `--run-id`, and `--case-id`. Reports use an
idempotency key and the parent run becomes terminal after every attempt reports.

## Tests

```sh
uv run pytest
```

The default suite is offline and secret-free. The manual GitHub workflow runs a bounded live text
or blackholed-voice smoke when `CONVERSE_API_KEY` is configured.

Licensed under the [Apache License 2.0](LICENSE).
