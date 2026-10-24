# Appeal-Pilot — Implementation Plan

Build a web app that drafts insurance prior-authorization appeal letters. It reads a denial letter and chart notes, matches chart evidence to payer criteria, drafts a cited letter, fact-checks the draft, and waits for a human to approve. The app is an entry in the Nebius x NVIDIA Global AI Hackathon, Best Apps and Agents track.

## Hard requirements (do not violate)

- Every LLM call goes through **Nebius Token Factory** at runtime, using **NVIDIA Nemotron** models.
- Use **synthetic data only**. No real patient data. Payers are fictional (e.g. "Acme Health", "Northwind Care"). Show a visible UI banner saying synthetic data, drafting aid, clinician sign-off required.
- The repo is public with an **MIT license**.
- Nothing is "sent". Approval only marks the letter final and enables export.
- The hosted demo needs no login.

## Stack

- Python 3.11, FastAPI, Uvicorn, Pydantic v2
- `openai` SDK pointed at Token Factory (OpenAI-compatible)
- `pypdf` for PDF text, `numpy` for vector search (in-memory), `rank_bm25` as fallback
- Frontend: a single static `index.html` with vanilla JS. No build step. The server streams progress with Server-Sent Events.
- `tavily-python` (optional, behind a flag)
- Docker for deployment

## Config (`.env`, add a `.env.example` too)

```
NEBIUS_API_KEY=
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
MODEL_FAST=        # Nemotron Lightning/Nano: extraction
MODEL_REASON=      # Nemotron 3 Super: matching, drafting, review
MODEL_EMBED=       # embeddings model from /v1/models; empty -> BM25 fallback
TAVILY_API_KEY=    # optional
ENABLE_TAVILY=false
MAX_RUNS_PER_IP_PER_HOUR=10
```

## Step 0 — Model discovery (do this first)

Create `scripts/discover_models.py`:
1. `GET /v1/models` and print every ID. Highlight IDs that contain `nemotron`, plus any embedding models.
2. For each Nemotron candidate, test (a) a `response_format` json_schema call and (b) a tool call. Print pass or fail for each.
3. Recommend values for `MODEL_FAST`, `MODEL_REASON` and `MODEL_EMBED`.

Do not hardcode model IDs anywhere else. Read them from config. If json_schema is not supported, fall back to `json_object` plus Pydantic validation (see LLM client).

## Repo layout

```
appeal-pilot/
  app/
    main.py            # FastAPI app + routes
    config.py
    llm.py             # Token Factory client wrapper
    schemas.py         # Pydantic models
    pipeline.py        # orchestrates the agents, yields progress events
    agents/
      intake.py
      policy.py
      matcher.py
      writer.py
      reviewer.py
    retrieval.py       # policy index (embeddings or BM25)
    tools/
      pdf.py           # PDF/text -> plain text
      tavily_search.py
    store.py           # in-memory case store (dict), JSON dump to disk
  static/index.html
  data/
    policies/*.md      # fictional payer criteria
    cases/case_XX/     # denial.txt, chart.txt, expected.json (+ PDFs for 3 demo cases)
  eval/run_eval.py
  scripts/discover_models.py
  tests/
  Dockerfile
  README.md
  LICENSE
  .env.example
```

## LLM client (`app/llm.py`)

- `chat_json(model, system, user, schema: type[BaseModel]) -> BaseModel`
  - Prefer `response_format={"type":"json_schema", ...}` generated from the Pydantic model. Fall back to `json_object`.
  - Validate with Pydantic. If validation fails, retry once and include the validation error in the prompt.
- Retry on 429 and 5xx with exponential backoff and jitter (max 4 attempts). Set a 60s timeout.
- Set `temperature` low (0.1 to 0.2) and explicit `max_tokens` on every call.
- Log per-call model, latency and token usage, and attach them to the case so the UI can show cost and latency.

## Schemas (`app/schemas.py`)

```python
class ChartLine(BaseModel): id: str; text: str               # "L1", "L2", ...
class Denial(BaseModel):
    payer: str; member_id: str | None; patient_name: str
    service: str; cpt_or_hcpcs: str | None; icd10: list[str]
    denial_reason: str; denial_date: str | None; appeal_deadline: str | None
class Criterion(BaseModel): id: str; text: str; source_doc: str; source_section: str
class CriterionMatch(BaseModel):
    criterion_id: str
    status: Literal["met", "not_met", "unclear"]
    evidence_line_ids: list[str]; rationale: str
class LetterSentence(BaseModel): id: str; text: str; cites: list[str]   # line ids / criterion ids
class AppealDraft(BaseModel): sentences: list[LetterSentence]
class ReviewFlag(BaseModel): sentence_id: str; issue: str; action: Literal["remove", "revise"]
class Review(BaseModel): flags: list[ReviewFlag]; summary: str
class Case(BaseModel):
    id: str; denial_text: str; chart_lines: list[ChartLine]
    denial: Denial | None; criteria: list[Criterion]; matches: list[CriterionMatch]
    draft: AppealDraft | None; review: Review | None
    final_letter: str | None; status: Literal["new", "running", "needs_review", "approved", "error"]
    usage: list[dict]
```

## Pipeline (`app/pipeline.py`)

The pipeline is an async generator that yields `{"step", "status", "data"}` events for SSE. Run the steps in order:

1. **Preprocess (no LLM).** Extract text from the files. Split the chart into numbered lines `L1..Ln` (by sentence or line). The letter can only cite these IDs.
2. **Intake agent (`MODEL_FAST`).** Turns the denial text into `Denial`.
3. **Policy agent (no LLM for retrieval, `MODEL_FAST` for parsing).**
   - Query the policy index with `payer + service + denial_reason`. Take the top k=4 chunks.
   - Extract the specific coverage criteria as `list[Criterion]`, each tied to its source doc and section.
   - If `ENABLE_TAVILY`, also run a Tavily search for public coverage criteria for the service. Show the results as "external references" in the UI only. Do not merge them into the criteria used for matching.
4. **Matcher agent (`MODEL_REASON`).**
   - Input: the criteria and the numbered chart lines. Output: `list[CriterionMatch]`.
   - Instruct the model to cite only line IDs that exist. If the evidence is absent, it must mark the criterion `not_met` or `unclear` and never infer.
5. **Writer agent (`MODEL_REASON`).** Drafts a formal appeal letter as `AppealDraft`.
   - Every factual sentence must cite chart line IDs or criterion IDs. Boilerplate sentences (salutation, closing) have empty `cites`.
   - Only argue from criteria marked `met`. For criteria marked `not_met` or `unclear`, the letter asks for peer-to-peer review rather than claiming they are met.
6. **Reviewer agent (`MODEL_REASON`, fresh context).**
   - Input: the draft, the chart lines and the criteria. The reviewer gets no writer reasoning.
   - It flags any sentence whose claim is not supported by the lines it cites, or that states facts absent from the chart.
7. **Deterministic checks (no LLM, run after the reviewer).**
   - Each cited ID must exist.
   - A sentence that has factual content but no citations is flagged.
   - Dates and numbers that appear in a sentence must appear in the cited lines.
   - Merge these flags with the reviewer's flags.
8. Set the status to `needs_review` and stop. Wait for the human.

On error, emit an `error` event with a readable message and set the status to `error`.

## API (`app/main.py`)

- `POST /api/cases`: multipart upload (denial and chart, PDF or txt), or JSON `{sample_id}` to load a bundled demo case. Returns `{case_id}`.
- `GET /api/samples`: lists the bundled demo cases.
- `GET /api/cases/{id}/run`: SSE stream that runs the pipeline.
- `GET /api/cases/{id}`: full case JSON.
- `PATCH /api/cases/{id}/letter`: human edits (sentence text, accept or reject a flag).
- `POST /api/cases/{id}/approve`: assembles the final letter (removed sentences dropped) and sets status to `approved`.
- `GET /api/cases/{id}/export`: final letter as `.txt` or `.md`. Requires `approved`.
- `GET /healthz`
- Rate-limit the run endpoint per IP (`MAX_RUNS_PER_IP_PER_HOUR`) to protect credits while the demo stays up through judging.

## UI (`static/index.html`)

One page with this layout, top to bottom:
1. Banner: synthetic data, drafting aid, clinician sign-off required.
2. Input: dropdown of sample cases, or upload two files. A **Run** button.
3. A step tracker (Intake → Policy → Match → Draft → Review), updated live from SSE.
4. Denial summary card: the extracted fields, with the appeal deadline highlighted.
5. Criteria table: criterion, source, status badge (met, not met, unclear) and evidence. Clicking a row highlights the cited lines in the chart panel.
6. Chart panel: the numbered lines, with highlights.
7. Letter editor:
   - Sentences are inline-editable.
   - Hovering a sentence shows its citations.
   - Flagged sentences are shown struck through in red with the reason, plus **Keep** and **Remove** buttons.
8. **Approve** button (disabled while unresolved flags remain), then **Download letter**.
9. Footer: models used, total tokens and latency per step.

Keep it clean and readable. Use system fonts and a single accent color. It must work at laptop width.

## Data (`data/`)

- **Policies.** About 15 markdown files for 2 fictional payers. Cover common prior-auth services: MRI lumbar spine, knee arthroscopy, GLP-1 for type 2 diabetes, biologics for psoriasis or RA, sleep study, CGM, PT extension, CT abdomen, bariatric surgery, and others. Model the criteria structure on how real public coverage policies are written, with numbered criteria and sections. Add front matter: `payer`, `service`, `codes`, `policy_id`.
- **Cases.** 10 synthetic cases. Each has:
  - `denial.txt`, a realistic denial letter citing a fictional policy ID
  - `chart.txt`, about 20 to 40 lines of clinic notes
  - `expected.json`, with the expected `Denial` fields, the expected status per criterion and evidence line IDs, and the overall verdict (appealable or not)
- **Mix.** 7 cases where the chart fully supports the appeal. 2 where one criterion is genuinely missing (the correct behavior is to mark it `not_met` or `unclear`). 1 where the chart contradicts a criterion.
- **Demo PDFs.** Render 3 cases to PDF for the demo. Use `reportlab`, or write the PDFs once via a script.

## Eval (`eval/run_eval.py`)

Run the pipeline on all 10 cases and print a markdown table with these metrics:
- **Extraction accuracy:** exact or normalized match per `Denial` field.
- **Criterion status accuracy** against `expected.json`.
- **Evidence precision:** share of cited line IDs that appear in the expected evidence.
- **Citation validity:** share of factual sentences whose citations pass the deterministic checks.
- **Reviewer catch rate:** for each case, inject 2 fabricated sentences into the draft (a false date and a symptom not in the chart). Re-run the reviewer and deterministic checks and measure what share they flag.
- **Mean latency and tokens per case.**

Write the results to `eval/results.md` and paste them into the README.

## Tests (`tests/`)

Use pytest. The tests must not call the LLM; mock `llm.py`. Cover:
- chart line splitting
- the deterministic checks
- schema validation and the retry path
- the retrieval fallback (BM25 when `MODEL_EMBED` is empty)
- approve and export rules

## Deployment

- `Dockerfile`: python:3.11-slim, install requirements, `uvicorn app.main:app --host 0.0.0.0 --port 8080`.
- Prefer deploying to Nebius (Serverless or AI Cloud). If that is blocked, use any container host. Document the URL in the README.
- Build the policy index at startup and cache the embeddings to disk.

## README (required by the submission)

- What it is, in two sentences, and the problem it solves
- An architecture diagram (Mermaid): the agent pipeline and which model each step uses
- **How Nebius and NVIDIA are used**: the Token Factory endpoint, the exact Nemotron model IDs per step, the embeddings model, structured outputs, and why the steps are tiered between the fast and reasoning models
- Eval results table
- Setup: `.env`, `discover_models.py`, run locally, run eval, run tests
- Synthetic data disclaimer and the not-medical-advice statement
- License: MIT

Also keep a `FEEDBACK.md` with notes on any friction with Token Factory or Nemotron hit during development (errors, missing features, docs gaps). The Devpost submission requires a feedback section.

## Definition of done

- [ ] `discover_models.py` runs, and the config uses verified Nemotron IDs
- [ ] All 3 demo cases run end to end from the UI in under 60s each
- [ ] The criteria table links each criterion to its highlighted chart lines
- [ ] The reviewer plus the deterministic checks visibly flag unsupported sentences
- [ ] Approve is blocked until all flags are resolved, and export works only after approval
- [ ] The eval runs, and the results are in the README
- [ ] Tests pass without network access
- [ ] The Docker image builds, and the app runs from it
- [ ] The README covers the Nebius and NVIDIA usage, and an MIT LICENSE file is present

## Out of scope

Real PHI, authentication, EHR or fax integration, multi-user accounts, a database (in-memory plus a JSON dump is enough), fine-tuning.
