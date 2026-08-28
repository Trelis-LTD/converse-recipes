# Guided customer-research assistant

This assistant investigates how small support teams handle urgent escalations. The plan asks for
evidence, not a questionnaire script: Converse chooses the order, follows useful threads and
clarifies ambiguity. `record_plan_field` is an ordinary client tool, so the application—not the
prompt—owns completion state.

The terminal example is the quickest way to try it:

```sh
uv run converse-guided examples/guided_customer_research/plan.json
```

For the browser example, serve the repository, open this directory, and enter a short-lived scoped
session credential. Voice opens the microphone; text uses `sendText()` and no media pipeline.
