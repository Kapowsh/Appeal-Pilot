# Code guide

A walk through every file in Appeal-Pilot: what it is for, and what each block of code does. Line numbers refer to the file as committed; they shift if you edit the file.

If you only read one section, read [How a run works](#how-a-run-works).

---

## How a run works

1. **You pick a sample or upload two files** (the denial letter and the chart notes). The browser sends them to `POST /api/cases` (`app/main.py`).
2. **The server turns the files into text** (`app/tools/pdf.py`) and splits the chart into numbered lines `L1, L2, …`. Those numbers are what every later citation points to.
3. **You press Run.** The browser opens `GET /api/cases/{id}/run`, and the server runs the pipeline (`app/pipeline.py`), sending a progress event after each step:

   | Step | File | Model | What it produces |
   |---|---|---|---|
   | intake | `app/agents/intake.py` | fast (Nano) | Structured denial: payer, service, codes, reason, dates |
   | policy | `app/agents/policy.py` + `app/retrieval.py` | fast | The coverage criteria `C1, C2, …` from the matching policy |
   | match | `app/agents/matcher.py` | reasoning (Super) | For each criterion: met / not met / unclear, plus the chart lines that prove it |
   | draft | `app/agents/writer.py` | reasoning | The appeal letter, as sentences `S1, S2, …`, each citing its sources |
   | review | `app/agents/reviewer.py` | reasoning | An independent fact-check that flags unsupported sentences |
   | checks | `app/checks.py` | none (plain code) | Rule-based flags: unknown citations, uncited facts, numbers not in the sources |

4. **You resolve each flag** (Keep or Remove) and can edit sentences (`PATCH /letter`).
5. **Approve** joins the kept sentences into the final letter. **Download** exports it. Nothing is ever sent anywhere.

Every model call goes through one class, `LLMClient` in `app/llm.py`, which talks to Nebius Token Factory.

---

## Repo layout

| Path | What it is |
|---|---|
| `app/` | The application (server, pipeline, agents) |
| `static/index.html` | The whole web page: HTML, CSS and JavaScript in one file |
| `data/policies/` | 15 fictional payer policies, each adapted from a real public policy |
| `data/cases/` | 10 synthetic test cases (denial, chart, answer key) |
| `data/imr_patterns.md` | Real-world denial statistics from California review decisions |
| `scripts/` | Helper tools you run by hand |
| `eval/` | The scoring harness |
| `tests/` | 152 automated tests (no network needed) |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Dependencies |
| `Dockerfile`, `.dockerignore` | How to build the deployable image |
| `.env.example` | Template for your settings; copy to `.env` |

---

## `app/config.py` — settings

- **Lines 7–20, `Settings`:** every setting the app reads, with its default. `pydantic-settings` fills each field from the matching environment variable or from `.env` (`nebius_api_key` ← `NEBIUS_API_KEY`, and so on). `extra="ignore"` means unknown variables in `.env` don't cause errors.
  - `model_fast`, `model_reason`, `model_embed` are empty by default on purpose: no model ID is hard-coded, so you choose them in `.env`.
  - `max_runs_per_ip_per_hour` is the per-visitor rate limit for the public demo.
  - `trust_proxy` only matters when deployed behind a reverse proxy (see `_client_ip` in `main.py`).
- **Lines 23–25, `get_settings()`:** reads the settings once and reuses them (`lru_cache`).

## `app/schemas.py` — the data shapes

Each class is a Pydantic model: a typed record that is validated whenever it is created. The models' JSON schemas are also sent to the LLM, so it knows the exact shape to return.

| Class | Meaning |
|---|---|
| `ChartLine` | One numbered chart line: `id` ("L3") and `text` |
| `Denial` | Fields pulled from the denial letter. `None` means "not in the letter" |
| `Criterion` | One coverage requirement: `id` ("C1"), text, which policy and section it came from |
| `CriterionMatch` | The verdict for one criterion: `status` (only `met` / `not_met` / `unclear` are allowed), the evidence line IDs and a short rationale |
| `LetterSentence` | One sentence of the letter, with `cites` (the L/C IDs or `DENIAL` backing it) |
| `AppealDraft` | The letter: an ordered list of sentences |
| `ReviewFlag` | A problem with one sentence: which sentence, what is wrong, and whether to `remove` or `revise` it |
| `Review` | All flags plus a one-line summary |
| `Case` | Everything about one run: inputs, each step's output, `status` (`new` → `running` → `needs_review` → `approved`, or `error`), token usage, your Keep/Remove decisions (`flag_resolutions`) and optional web references |
| `CriteriaList`, `MatchList` (lines 77–82) | Wrappers, because structured output needs a JSON object at the top level, not a bare list |

## `app/llm.py` — the only place models are called

- **Lines 18–21:** constants. 4 attempts per request, 60-second timeout, and two regexes: one for `<think>…</think>` reasoning blocks, one for ```` ```json ```` code fences.
- **Lines 24–25, `LLMError`:** the single error type the rest of the app catches.
- **Lines 28–36, `_clean()`:** gets the JSON out of a model reply. It removes reasoning blocks, unwraps a code fence if there is one, then keeps the text from the first `{` to the last `}`.
- **Lines 39–48, `LLMClient.__init__`:** builds the OpenAI SDK client pointed at Nebius (`base_url`). The SDK's own retries are off (`max_retries=0`) because this class does its own. Without an API key no client is built, and calls fail with a clear message instead.
- **Lines 50–63, `_with_retries()`:** retries rate limits (429), connection errors and server errors (5xx), waiting 0.5 s, 1 s, 2 s plus a little randomness between attempts. Client errors (other 4xx) fail at once, because retrying won't fix them.
- **Lines 65–83, `_complete()`:** sends the request. It first asks for `json_schema` (the model must follow the exact schema). If the model rejects that, it remembers (`json_schema_ok = False`) and uses the looser `json_object` mode from then on.
- **Lines 85–91, `_record()`:** logs each call's step, model, latency and token counts. The usage footer in the UI shows these.
- **Lines 93–111, `chat_json()`:** the method every agent uses. It appends the JSON schema to the system prompt, calls the model and validates the reply against the schema. If validation fails, it retries once, telling the model what was wrong.
- **Lines 113–118, `_check_ready()`:** fails fast with a readable message when a model ID or the API key is missing.
- **Lines 120–125, `embed()`:** gets embedding vectors for policy search, using the same retry logic.

## `app/tools/pdf.py` — reading files

- **Lines 12–21, `extract_text()`:** a PDF (detected by file name or by the `%PDF` header) is read page by page with `pypdf`; anything else is read as UTF-8 text. A scanned PDF with no text layer is rejected with a clear error.
- **Lines 24–35, `split_chart_lines()`:** numbers the chart as `L1…Ln`. Blank lines are skipped. Lines over 200 characters are split at sentence ends, so a citation points at one fact rather than a whole paragraph. The numbering must be deterministic, because the answer keys refer to these IDs.

## `app/tools/tavily_search.py` — optional web search

- **`search()`:** asks Tavily for public pages about the service. It returns `[]` on any failure, so web search can never break a run. The results are shown as "external references" only and are never used as evidence. It's off unless `ENABLE_TAVILY=true`.

## `app/retrieval.py` — finding the right policy

- **Lines 19–54, `PolicyDoc` / `parse_policy()`:** reads a policy file. The block between the two `---` lines is front matter (`policy_id`, `payer`, `service`, `codes`, `source`, `source_url`); the rest is the policy text. Missing required keys raise an error naming the file.
- **Lines 57–58, `load_policies()`:** loads every `.md` file in `data/policies/`.
- **Lines 63–86, `Chunk` / `chunk_policy()`:** cuts each policy into one chunk per `## ` section, with the policy's title, payer and codes repeated at the top so each chunk makes sense on its own.
- **Lines 93–106, `PolicyIndex.__init__`:** always builds a BM25 keyword index. If an embedding model is set, it also builds a vector index; if that fails for any reason, it logs a warning and stays on BM25 so the demo keeps working.
- **Lines 108–117, `_load_vectors()`:** embeddings are cached in `cache/embeddings-<hash>.npy`. The hash covers the model name and all policy text, so editing a policy automatically rebuilds the cache.
- **Lines 119–136, `search()`:** returns the top 4 chunks, after three safety filters:
  1. Only the denial's payer. If no policy matches the payer, it returns nothing, rather than another payer's rules.
  2. If the denial cites a policy ID (e.g. `AH-RAD-012`), only that policy.
  3. In BM25 mode, chunks with zero shared words are dropped.
- **Lines 138–144, `build()`:** builds the index at server start.
- **Lines 147–150, `_payer_matches()`:** tolerant payer-name match, so "Acme Health Plan" still matches "Acme Health".
- **Lines 153–155, `_normalize()`:** scales vectors to length 1, so a dot product equals cosine similarity.

## `app/agents/` — the five model steps

Each agent file has a `SYSTEM` prompt (the instructions) and a `run()` function. The prompt tells the model what to do; the code after the model call enforces the rules anyway, because a model can ignore instructions.

- **`__init__.py`:** `format_lines()` renders chart lines as `L3: text`, and `format_criteria()` renders criteria as `C1 [AH-RAD-012 §2.1]: text`, both for prompts.
- **`intake.py`:** sends the denial letter to the fast model and gets back a `Denial`. The prompt says to copy values as written and use `null` for anything missing.
- **`policy.py`:**
  - Line 19 builds the search query from payer + service + reason. Line 20 finds a policy ID cited in the denial text.
  - Line 21 runs `index.search`, and line 22 runs the optional web search.
  - Lines 23–24: if nothing was retrieved, return no criteria without calling the model.
  - Lines 25–28 ask the fast model to list the numbered criteria from the retrieved excerpts.
  - Lines 31–35 enforce that criteria come only from retrieved policies, and only from one policy (the most common), never a mix.
  - Lines 36–41 renumber the criteria `C1…Cn` and keep the policy's own section number (e.g. "2.3").
- **`matcher.py`:**
  - The reasoning model judges each criterion against the chart lines.
  - Lines 17–27 clean up the reply: IDs are uppercased, duplicates and chart line IDs that don't exist are dropped, and a "met" with no valid evidence is downgraded to "unclear".
  - Lines 28–29 make sure every criterion gets a verdict, defaulting to "unclear".
- **`writer.py`:**
  - The reasoning model writes the letter as cited sentences. Key prompt rules: argue only from "met" criteria, request a peer-to-peer review for the rest, cite `DENIAL` for facts from the denial letter, and never put IDs inside the sentence text.
  - Lines 28–31 drop empty sentences, renumber them `S1…Sn` and remove duplicate citations.
- **`reviewer.py`:**
  - A separate reasoning call acts as a fact-checker.
  - It sees only the finished sentences and the sources, never the writer's prompt or reasoning, so it can't be talked into agreeing.
  - Lines 22–24 drop flags that point at sentences that don't exist.

## `app/checks.py` — rule-based checks (no model)

These run after the reviewer and catch mistakes even when the reviewer misses them.

- **Lines 6–7, `_BOILERPLATE`:** greetings and closings ("Dear", "Sincerely", "Re:", …), which never need citations.
- **Line 9, `DENIAL_ID`:** the special citation meaning "from the denial letter".
- **Lines 19–22, `is_boilerplate()`:** true for those openers, or for very short lines with no digits (e.g. a signature).
- **Lines 29–34, `is_factual()`:** treats a sentence as a factual claim if it contains a number, or shares at least two meaningful words with the chart.
- **Lines 37–46, `_numbers()`:** pulls every number out of a text, ignoring leading zeros. Spelled-out dates also yield the month number, so "September 18, 2026" matches "2026-09-18".
- **Lines 49–81, `deterministic_checks()`:** for each sentence:
  1. Every cited ID must exist (a chart line, a criterion or `DENIAL`).
  2. A factual sentence with no citations is flagged.
  3. Every number in the sentence must appear in its cited sources. Inline IDs like "(L4)" and section numbers like "2.3" are removed first, since they aren't clinical numbers. A missing number is flagged `remove`, the strongest action.
- **Lines 84–96, `merge_flags()`:** combines the reviewer's flags and these flags, one per sentence. Issue texts are joined, and `remove` wins over `revise`.

## `app/pipeline.py` — running the steps in order

- **Line 15, `STEPS`:** the step names the UI's progress tracker uses.
- **Lines 30–35, `stage()`:** runs one agent in a background thread (`asyncio.to_thread`) so the server stays responsive, and copies token usage onto the case after each step.
- **Lines 38–70:** each step sends a "running" event, does its work, then sends a "done" event with its result. If the policy step finds no criteria, the run stops with a clear error (lines 48–49), because a letter with no criteria to argue from would be invented.
- **Lines 72–73:** on success the case becomes `needs_review`, waiting for you.
- **Lines 74–76:** if the browser disconnects mid-run, the case is set to `error` so it can be run again instead of being stuck at "running".
- **Lines 77–80:** any other failure is logged and sent to the UI as an error event naming the step.

## `app/store.py` — storage and letter assembly

- **Lines 16–34, `CaseStore`:** keeps cases in memory, keyed by a random 12-character ID, and writes each one to `cache/cases/<id>.json` on every change (handy for debugging). A server restart empties the in-memory store.
- **Lines 37–43, `list_samples()`:** lists `data/cases/case_XX` folders with their titles, for the dropdown.
- **Lines 46–53, `load_sample()`:** reads a sample's denial and chart. The ID must match `case_NN` exactly, which blocks path tricks like `../../`.
- **Lines 56–72, `assemble_letter()`:** builds the final letter. It skips sentences you removed, puts greetings and closings on their own lines, and joins body sentences into paragraphs.
- **Lines 75–89, `RateLimiter`:** remembers each visitor's run times from the last hour and refuses a run once there are `max_per_hour` of them.

## `app/main.py` — the web server

- **Lines 24–26:** where the web page lives, the 5 MB upload limit, and the 15-second heartbeat.
- **Lines 29–32, `LetterEdit`:** the body of an edit request: which sentence, new text and/or a Keep/Remove decision.
- **Lines 35–40, `_client_ip()`:** the visitor's IP, for rate limiting. The `X-Forwarded-For` header can be faked by anyone, so it is used only when `TRUST_PROXY` is on, and only its last entry (the one your own proxy added).
- **Lines 43–71, `stream_events()`:** turns pipeline events into Server-Sent Events (`data: {...}`), the format the browser reads as a live stream. If no event arrives for 15 s it sends a `: ping` comment, so proxies don't close the connection while a slow model thinks. When the stream ends for any reason, the case is saved and never left at "running".
- **Lines 74–86, `_read_upload()`:** reads an uploaded file, enforces the 5 MB limit and extracts its text, turning errors into clear 400/413 responses.
- **Lines 89–103, `create_app()`:** sets up the app. It builds the policy index once at startup (`lifespan`) and serves `static/`. `llm_factory` exists so the tests can substitute a fake model.
- **Routes:**

  | Route | Lines | What it does |
  |---|---|---|
  | `GET /` | 111–113 | Serves the web page |
  | `GET /healthz` | 115–117 | Returns `{"ok": true}` for uptime checks |
  | `GET /api/samples` | 119–121 | Lists the sample cases |
  | `POST /api/cases` | 123–138 | Creates a case from a sample ID (JSON) or two uploaded files (form). Returns the case ID |
  | `GET /api/cases/{id}` | 140–142 | Returns the whole case |
  | `GET /api/cases/{id}/run` | 144–154 | Starts the pipeline and streams progress. 409 if it is already running or done; 429 if over the hourly limit |
  | `PATCH /api/cases/{id}/letter` | 156–169 | Edits a sentence or records Keep/Remove. Only allowed while the case awaits review |
  | `POST /api/cases/{id}/approve` | 171–182 | Refuses until every flag has a decision, then builds the final letter and marks it approved |
  | `GET /api/cases/{id}/export` | 184–190 | Downloads the approved letter as `.txt` or `.md` |

- **Line 195:** `app = create_app()` is the object `uvicorn app.main:app` runs.

---

## `static/index.html` — the web page

One file with no build step and no framework.

- **Lines 1–99:** CSS (layout, colors, the step-tracker states, the flagged-sentence styling).
- **Lines 100–166:** the page structure: the synthetic-data banner, then six numbered cards (choose a case, progress, denial summary, coverage criteria, chart notes, appeal letter) and the usage footer.
- **Lines 176–186, `el()`:** creates page elements safely. Text is always inserted as text, never as HTML (the file never uses `innerHTML`), so content from a model or an upload can't inject scripts.
- **`api()`:** wraps `fetch` and turns error responses into readable messages.
- **`loadSamples()`, `createCase()`, `run()`:** fill the dropdown, create the case and start the run.
- **`streamRun()` (line ~264):** reads the live event stream with `fetch` rather than `EventSource`, so a 409 or 429 can show its message.
- **`onEvent()` (line ~282):** updates the step tracker and renders each step's result as it arrives.
- **Render functions:**
  - `renderDenial()` shows the extracted denial fields.
  - `renderCriteria()` shows the criteria table; clicking a row highlights its evidence lines in the chart panel.
  - `renderChart()` shows the numbered chart lines.
  - `renderLetter()` shows the sentences. Each sentence is editable, hovering highlights its citations, and flagged sentences get Keep / Remove buttons.
- **`updateButtons()` (line ~421):** Approve stays disabled until every flag is resolved; Download stays disabled until the letter is approved.
- **`renderUsage()`:** the footer showing each model call's model, latency and tokens.
- External-reference links are shown only if they start with `http://` or `https://`.

---

## `scripts/` — tools you run by hand

- **`discover_models.py`:** lists the models your Nebius key can use, sorts them into Nemotron and embedding models, and probes each Nemotron model: does it support `json_schema`, `json_object` and tool calls? It then prints recommended `MODEL_FAST` / `MODEL_REASON` / `MODEL_EMBED` values.
- **`make_pdfs.py`:** renders the denial and chart of demo cases 01–03 as PDFs with `reportlab`, one text line per PDF line, so the PDF upload path can be tested.
- **`imr_patterns.py`:** summarizes the public California Independent Medical Review CSV into `data/imr_patterns.md`.
  - `match_service()` maps a decision to one of this app's services by keyword.
  - `summarize()` counts decisions, overturn rates and review themes.
  - `render()` writes the Markdown table.

## `eval/run_eval.py` — scoring the pipeline

Runs all 10 cases through the real pipeline and compares the results with each case's `expected.json`.

| Metric | Function | Meaning |
|---|---|---|
| Extraction | `extraction_accuracy()` | Share of denial fields extracted correctly. Free-text fields are scored by word overlap, dates in any format |
| Status acc | `criterion_scores()` | Share of criteria given the right met / not met / unclear verdict |
| Evidence precision | `criterion_scores()` | Share of cited evidence lines that are in the answer key |
| Citation validity | `citation_validity()` | Share of factual sentences that pass the rule-based checks |
| Catch rate | `inject_fabrications()` + `catch_rate()` | Two false sentences (an impossible date, 02/29/2019, and a symptom absent from every chart) are added to each letter; this is the share the reviewer and checks catch |
| Latency, tokens | `run_case()` | Time and tokens per case |

Results are written to `eval/results.md`.

---

## `data/`

- **`policies/*.md`:** 15 policies for the fictional payers Acme Health and Northwind Care. Front matter says which real public policy each is adapted from (`source`, `source_url`). Section 2 holds the numbered coverage criteria the policy agent extracts.
- **`cases/case_XX/`:** for each synthetic case:
  - `denial.txt`: the denial letter.
  - `chart.txt`: the chart notes.
  - `expected.json`: the answer key (correct denial fields; correct status and evidence lines for each criterion).
  - `denial.pdf` and `chart.pdf`: PDF copies, cases 01–03 only.

  The mix is 7 appealable cases, 2 missing a criterion and 1 contradicted by the chart.
- **`imr_patterns.md`:** real-world denial statistics (see the README's "Denial realism" section).

## `tests/`

One test file per module, for example `test_checks.py` for `checks.py`.

- **`conftest.py`:** blocks all network access except to the local machine, so the tests never spend API credits.
- **`fakes.py`:** stand-ins used instead of the real services.
  - `FakeOpenAI` replays scripted replies.
  - `ScriptedLLM` returns a fixed answer per agent.
  - `api_error` builds fake HTTP errors.

Run them all with `uv run pytest -q`.

## Deployment files

- **`Dockerfile`:**
  1. Starts from Python 3.11 slim and installs `requirements.txt`.
  2. Copies the app, page, data and scripts. Tests, `.env` and caches are excluded by `.dockerignore`.
  3. Runs `uvicorn` on port 8080.
- **`.env.example`:** every setting with a placeholder. Copy it to `.env` and fill in your key and model IDs; `.env` is git-ignored.
