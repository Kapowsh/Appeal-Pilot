# Feedback: Nebius Token Factory and NVIDIA Nemotron

Notes on friction hit while building Appeal-Pilot, for the Devpost feedback section.

## Status

No live Token Factory calls have been made yet. The app, tests and eval harness were built offline, before an API key was available. The entries below are the design decisions we made in case of known integration issues. Each one is marked **to verify** until it has been checked against the live API.

## Integration notes (to verify)

- **`response_format: json_schema` support per model.** We didn't know which Nemotron models on Token Factory accept `json_schema`. `app/llm.py` falls back to `json_object` plus Pydantic validation on the first 400, and `scripts/discover_models.py` probes both. To record: which model IDs pass each probe.
- **Reasoning output in `content`.** Some Nemotron reasoning modes emit `<think>…</think>` before the answer. The client strips it before JSON parsing. To record: whether this shows up on Token Factory, and whether a system-prompt or parameter switch turns it off.
- **Tool calling.** The discovery script probes `tools`/`tool_choice`, although the pipeline uses structured outputs instead. To record: the results.
- **Embeddings model.** To record: which embeddings model `/v1/models` lists and whether it is NVIDIA-hosted. If none is usable, the app falls back to BM25.
- **Rate limits and latency.** The client retries 429 and 5xx responses with backoff. To record: limits observed during the eval run, and per-step latency.

## Friction log

| Date | Area | What happened | Workaround |
|---|---|---|---|
| (add entries during live testing) | | | |
