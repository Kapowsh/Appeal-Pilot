# Appeal-Pilot

> **Synthetic data only · Drafting aid, not medical advice · Clinician sign-off required before use**

Entry for the Nebius x NVIDIA Global AI Hackathon, Best Apps and Agents track.

## What it is

Appeal-Pilot reads a prior-authorization denial letter and the patient's chart notes. It matches chart evidence to the payer's coverage criteria, drafts an appeal letter in which every factual sentence cites a chart line, fact-checks the draft, and waits for a clinician to approve it.

The problem: prior-authorization appeals eat clinician and staff time, and a letter that misstates the chart is worse than no letter. Appeal-Pilot makes each claim traceable to a numbered chart line and flags any claim it can't support.

Nothing is ever sent. Approving only marks the letter final and enables the download.

## Architecture

```mermaid
flowchart LR
    U[Denial + chart<br/>PDF or text] --> P[Preprocess<br/>no LLM: text and L1..Ln lines]
    P --> I[Intake agent<br/>MODEL_FAST]
    I --> PO[Policy agent<br/>retrieval: embeddings or BM25<br/>+ MODEL_FAST parsing]
    PO --> M[Matcher agent<br/>MODEL_REASON]
    M --> W[Writer agent<br/>MODEL_REASON]
    W --> R[Reviewer agent<br/>MODEL_REASON, fresh context]
    R --> D[Deterministic checks<br/>no LLM: IDs, citations, numbers]
    D --> H{Human review<br/>Keep / Remove each flag}
    H -->|all flags resolved| A[Approve and export .txt or .md]
    T[(Tavily, optional)] -.->|display only| PO
```

- **Preprocess:** `app/tools/pdf.py` extracts text and splits the chart into numbered lines. The letter can cite only these IDs.
- **Intake:** `app/agents/intake.py` turns the denial into a structured `Denial`.
- **Policy:** `app/agents/policy.py` queries the policy index (`app/retrieval.py`, top k = 4) with payer + service + denial reason, then extracts the numbered criteria, each tied to its policy ID and section.
- **Matcher:** `app/agents/matcher.py` marks each criterion met, not met or unclear, citing chart line IDs. IDs that aren't in the chart are dropped, and a "met" with no valid evidence is downgraded to "unclear".
- **Writer:** `app/agents/writer.py` argues only from met criteria, and asks for peer-to-peer review for the rest.
- **Reviewer:** `app/agents/reviewer.py` sees only the letter, the chart lines and the criteria, never the writer's reasoning, and flags unsupported sentences.
- **Deterministic checks:** `app/checks.py` flags cited IDs that don't exist, factual sentences with no citation, and numbers or dates missing from the cited lines.
- **Streaming:** `app/pipeline.py` streams each step to the browser over Server-Sent Events.

## How Nebius and NVIDIA are used

- **Endpoint.** Every LLM call goes through `app/llm.py`, which uses the `openai` SDK pointed at **Nebius Token Factory**: `https://api.tokenfactory.nebius.com/v1/`. No other model provider is used.
- **Models.** All chat steps use **NVIDIA Nemotron** models. Their IDs come only from `.env`, as found by `scripts/discover_models.py`:

  | Step | Env var | Model ID |
  |---|---|---|
  | Intake, policy parsing | `MODEL_FAST` | TODO (run `scripts/discover_models.py`), a Nemotron Nano / Lightning model |
  | Matcher, writer, reviewer | `MODEL_REASON` | TODO (run `scripts/discover_models.py`), Nemotron 3 Super |
  | Policy embeddings | `MODEL_EMBED` | TODO (run `scripts/discover_models.py`). Empty means the BM25 fallback |

- **Structured outputs.** Every call requests `response_format: json_schema`, generated from the Pydantic model. If the model rejects that, the client falls back to `json_object` with the schema in the prompt. Replies are validated with Pydantic and retried once, with the validation error fed back to the model. `<think>` blocks and code fences are stripped first. Transport errors (429/5xx) are retried with exponential backoff and jitter, up to 4 attempts. Calls use a 60 s timeout, temperature 0.1 and explicit `max_tokens`.
- **Why the models are tiered.** Extraction and criteria parsing copy fields out of short documents, so a small fast Nemotron handles them cheaply. Matching evidence to criteria, writing a careful argument, and catching unsupported claims need the stronger reasoning model. The reviewer runs on the same model in a fresh context, so it judges the letter without the writer's framing.
- **Usage shown in the UI.** Per-call model, latency and token usage are recorded on the case and shown in the page footer.

## Eval results

`eval/run_eval.py` runs the full pipeline on the 10 bundled cases: 7 where the chart fully supports the appeal, 2 with a genuinely missing criterion, and 1 where the chart contradicts a criterion. It measures:

- **Extraction:** denial fields matched, using normalized exact match. `service` and `denial_reason` are scored by token overlap.
- **Status acc:** criterion status accuracy against `expected.json`.
- **Evidence precision:** the share of cited chart lines that appear in the expected evidence.
- **Citation validity:** the share of factual sentences that pass the deterministic checks.
- **Catch rate:** two fabricated sentences (an impossible date and a symptom absent from the chart) are injected into each draft. The reviewer and checks are re-run, and the rate is the share they flag.
- Latency and tokens per case.

TODO: run `uv run python eval/run_eval.py` with Token Factory credentials, then paste `eval/results.md` here.

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`. This uses Python 3.11.
2. `cp .env.example .env` and set `NEBIUS_API_KEY`.
3. Discover models: `uv run python scripts/discover_models.py`. Copy the recommended `MODEL_FAST`, `MODEL_REASON` and `MODEL_EMBED` lines into `.env`.
4. Run locally: `uv run uvicorn app.main:app --port 8080`, then open http://localhost:8080. Choose a sample case or upload a denial and chart (PDF or `.txt`; demo PDFs are in `data/cases/case_01..03`).
5. Run the eval: `uv run python eval/run_eval.py`. Results are written to `eval/results.md`.
6. Run the tests: `uv run pytest -q`. The tests never call the network; the LLM is mocked.
7. Docker: `docker build -t appeal-pilot .`, then `docker run -p 8080:8080 --env-file .env appeal-pilot`.
8. Regenerate the demo PDFs: `uv run python scripts/make_pdfs.py`.

Runs are rate-limited per IP (`MAX_RUNS_PER_IP_PER_HOUR`, default 10) to protect credits on the public demo. Behind a reverse proxy, set `TRUST_PROXY=true` so the limit uses the proxy-appended client IP; otherwise `X-Forwarded-For` is ignored. The demo needs no login. Tavily search is optional and off by default (`ENABLE_TAVILY=false`). Its results are shown as external references only and are never used for matching.

Hosted demo: TODO (deployment URL).

## Disclaimer

All patients, members, clinicians, payers ("Acme Health", "Northwind Care") and policies in this repository are **fictional**, and all data is **synthetic**. Appeal-Pilot is a drafting aid. It is **not medical advice** and does not make coverage determinations. A licensed clinician must review, edit and sign off on every letter before use. The app never sends anything.

## License

MIT. See [LICENSE](LICENSE).
