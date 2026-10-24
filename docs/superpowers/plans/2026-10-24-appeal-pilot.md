# Appeal-Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI web app that turns a synthetic prior-auth denial letter plus chart notes into a cited, fact-checked appeal letter, which waits for a human to approve it.

**Architecture:** A five-agent pipeline (intake → policy → matcher → writer → reviewer) plus deterministic checks, run as an async generator that streams Server-Sent Events. Every LLM call goes through one wrapper (`app/llm.py`) that calls Nebius Token Factory's OpenAI-compatible API with Nemotron model IDs read from env. Policies are retrieved by in-memory embeddings, with BM25 as the fallback. A single static HTML page drives the flow.

**Tech Stack:** Python 3.11 (uv), FastAPI, Uvicorn, Pydantic v2 + pydantic-settings, openai SDK, pypdf, numpy, rank_bm25, tavily-python, pytest + pytest-asyncio, reportlab (dev, for PDFs).

**Spec:** `PLAN.md` (repo root). It is the binding authority. This plan argues from it.

## Global Constraints

- Every LLM call goes through **Nebius Token Factory** at runtime, using **NVIDIA Nemotron** models. No other provider and no fake or offline mode in app code. Fakes live only in `tests/`.
- Don't hardcode model IDs anywhere except `scripts/discover_models.py` output. Read `MODEL_FAST`, `MODEL_REASON` and `MODEL_EMBED` from config.
- Synthetic data only. Payers are fictional: **Acme Health** and **Northwind Care**.
- UI banner copy (exact): `Synthetic data only · Drafting aid, not medical advice · Clinician sign-off required before use`
- Nothing is "sent". Approval only marks the letter final and enables export. No login.
- MIT license. The repo is public.
- Env names and defaults (exact): `NEBIUS_API_KEY=`, `NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/`, `MODEL_FAST=`, `MODEL_REASON=`, `MODEL_EMBED=`, `TAVILY_API_KEY=`, `ENABLE_TAVILY=false`, `MAX_RUNS_PER_IP_PER_HOUR=10`.
- LLM calls: temperature 0.1, explicit `max_tokens`, 60s timeout, at most 4 attempts on 429/5xx with exponential backoff plus jitter, and one validation retry.
- Retrieval top-k = 4.
- Tests never touch the network. `tests/conftest.py` blocks socket connects.
- Code lives at the repo root, using the spec's layout minus the `appeal-pilot/` prefix.
- Test command for every task: `uv run pytest -q`.

## Review Focus

1. **An uploaded PDF with no extractable text** (scanned image) or a file over 5 MB: the user expects a readable 400/413 error, not an empty pipeline run. Covered by a test in Task 10.
2. **A denial whose payer or service matches no policy:** the user expects a readable `error` event ("No matching coverage criteria found …"), not a letter argued from nothing. Covered by a test in Task 9.
3. **Uploaded text containing HTML or `<script>`:** the user expects it shown as literal text in the chart and letter panels. Covered by a test in Task 12 (no `innerHTML` in the page).
4. **Re-running `/run` on a case that is running or already approved:** the user expects a 409, with the approved letter untouched. Covered by a test in Task 10.
5. **The LLM returns IDs with odd casing or whitespace** (`" l4"`, `"c2 "`): the user expects them normalized to `L4` and `C2`, not dropped as unknown and not flagged as bogus. Covered by tests in Tasks 7 and 8.

---

### Task 1: Scaffold and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `app/__init__.py`, `app/config.py`, `tests/__init__.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (pydantic-settings, `env_file=".env"`, `extra="ignore"`) with fields `nebius_api_key: str = ""`, `nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"`, `model_fast: str = ""`, `model_reason: str = ""`, `model_embed: str = ""`, `tavily_api_key: str = ""`, `enable_tavily: bool = False`, `max_runs_per_ip_per_hour: int = 10`, `data_dir: Path = Path("data")`, `cache_dir: Path = Path("cache")`. Also `get_settings() -> Settings` (`lru_cache`).
- Produces: `tests/conftest.py`, with an autouse fixture that patches `socket.socket.connect` to raise `RuntimeError("network disabled in tests")` and clears `get_settings.cache_clear()`.

- [ ] **Step 1: Write `pyproject.toml`.** Use `[project] requires-python = ">=3.11,<3.12"` with deps `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `pydantic-settings`, `openai`, `pypdf`, `numpy`, `rank-bm25`, `tavily-python` and `python-multipart`. Add `[dependency-groups] dev = ["pytest", "pytest-asyncio", "httpx", "reportlab"]`, `[tool.uv] package = false` and `[tool.pytest.ini_options] pythonpath = ["."]`, `asyncio_mode = "auto"`, `testpaths = ["tests"]`. Write `.gitignore`: `.superpowers/ .venv/ .env cache/ __pycache__/ *.pyc`. Write `.env.example` with the exact Global Constraints env lines. Write `LICENSE`: the MIT License text, `Copyright (c) 2026 Appeal-Pilot contributors`. Run `uv sync`. Expected: it resolves and creates `.venv`.
- [ ] **Step 2: Write the failing tests**

```python
def test_defaults(monkeypatch):
    for k in ["MODEL_FAST","MODEL_REASON","MODEL_EMBED","ENABLE_TAVILY","MAX_RUNS_PER_IP_PER_HOUR"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.nebius_base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert s.model_embed == "" and s.enable_tavily is False and s.max_runs_per_ip_per_hour == 10

def test_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_REASON", "x/y"); monkeypatch.setenv("ENABLE_TAVILY", "true")
    s = Settings(_env_file=None)
    assert s.model_reason == "x/y" and s.enable_tavily is True

def test_no_hardcoded_model_ids():
    for p in Path("app").rglob("*.py"):
        assert "nemotron" not in p.read_text(encoding="utf-8").lower(), p

def test_network_blocked():
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("example.com", 80))
```

- [ ] **Step 3: Run** `uv run pytest tests/test_config.py -q`. Expected: FAIL (`ModuleNotFoundError: app.config`).
- [ ] **Step 4: Implement** `app/config.py` and `tests/conftest.py` per Interfaces.
- [ ] **Step 5: Run** `uv run pytest -q`. Expected: 4 passed.
- [ ] **Step 6: Commit** with message `feat: project scaffold, config, MIT license`. Include `uv.lock`.

### Task 2: Schemas

**Files:** Create `app/schemas.py`. Test: `tests/test_schemas.py`.

**Interfaces:**
- Produces: the spec's models verbatim: `ChartLine`, `Denial`, `Criterion`, `CriterionMatch`, `LetterSentence`, `AppealDraft`, `ReviewFlag`, `Review`, `Case`. `Case` gets defaults: `denial=None`, `criteria=[]`, `matches=[]`, `draft=None`, `review=None`, `final_letter=None`, `status="new"`, `usage=[]`. It also gets two extra fields: `flag_resolutions: dict[str, Literal["keep","remove"]] = {}` (keyed by sentence_id) and `external_refs: list[dict] = []` (Tavily results, display only).
- Produces: `CriteriaList(criteria: list[Criterion])`, `MatchList(matches: list[CriterionMatch])`. These are LLM wrapper schemas.

- [ ] **Step 1: Write the failing tests**

```python
def test_match_rejects_bad_status():
    with pytest.raises(ValidationError):
        CriterionMatch(criterion_id="C1", status="maybe", evidence_line_ids=[], rationale="")

def test_flag_action_literal():
    with pytest.raises(ValidationError):
        ReviewFlag(sentence_id="S1", issue="x", action="delete")

def test_case_defaults_and_roundtrip():
    c = Case(id="a", denial_text="d", chart_lines=[ChartLine(id="L1", text="t")])
    assert c.status == "new" and c.flag_resolutions == {} and c.usage == []
    assert Case.model_validate_json(c.model_dump_json()) == c

def test_denial_optional_fields():
    d = Denial(payer="Acme Health", member_id=None, patient_name="Jo Doe", service="MRI",
               cpt_or_hcpcs=None, icd10=["M54.16"], denial_reason="r", denial_date=None, appeal_deadline=None)
    assert d.icd10 == ["M54.16"]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_schemas.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `app/schemas.py`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: pydantic schemas`.

### Task 3: Text extraction and chart line splitting

**Files:** Create `app/tools/__init__.py` and `app/tools/pdf.py`. Test: `tests/test_pdf.py`.

**Interfaces:**
- Produces: `extract_text(data: bytes, filename: str) -> str`. This treats the input as a PDF when the filename ends `.pdf` or the data starts with `b"%PDF"`; it uses pypdf and joins pages with `"\n"`. Otherwise it decodes UTF-8 with `errors="replace"`. It raises `ValueError("No extractable text in <filename>")` when the result is blank.
- Produces: `split_chart_lines(text: str) -> list[ChartLine]`. Split on newlines, strip, and drop blank lines. A line longer than 200 chars is further split on `(?<=[.!?])\s+(?=[A-Z])`. IDs are `L1..Ln` in order. The output is deterministic, because `expected.json` and the eval depend on it.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_drops_blanks_and_numbers_sequentially():
    lines = split_chart_lines("  A one.\n\n\nB two.\n C three. ")
    assert [(l.id, l.text) for l in lines] == [("L1","A one."),("L2","B two."),("L3","C three.")]

def test_short_line_with_abbreviation_not_split():
    assert len(split_chart_lines("Seen by Dr. Lee today. BP 120/80.")) == 1

def test_long_line_sentence_split():
    long = "Patient reports low back pain radiating to the left leg. " * 3 + "Exam shows positive straight leg raise on the left side today."
    assert len(split_chart_lines(long)) == 4

def test_extract_txt():
    assert extract_text("héllo\n".encode(), "chart.txt").strip() == "héllo"

def test_extract_pdf(tmp_path):
    from reportlab.pdfgen import canvas
    p = tmp_path / "x.pdf"; c = canvas.Canvas(str(p)); c.drawString(72, 720, "Synthetic chart line"); c.save()
    assert "Synthetic chart line" in extract_text(p.read_bytes(), "x.pdf")

def test_blank_pdf_raises(tmp_path):
    from reportlab.pdfgen import canvas
    p = tmp_path / "b.pdf"; c = canvas.Canvas(str(p)); c.showPage(); c.save()
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text(p.read_bytes(), "b.pdf")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_pdf.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: text extraction and chart line splitting`.

### Task 4: LLM client

**Files:** Create `app/llm.py`. Tests: `tests/fakes.py` (shared test doubles) and `tests/test_llm.py`.

**Interfaces:**
- Consumes: `Settings`.
- Produces: `class LLMError(Exception)`.
- Produces: `class LLMClient`, with:
  - `__init__(self, settings: Settings, client=None, sleep=time.sleep)`. The default client is `OpenAI(base_url=settings.nebius_base_url, api_key=settings.nebius_api_key, timeout=60, max_retries=0)`.
  - `usage: list[dict]`. Each entry has keys `step, model, latency_ms, prompt_tokens, completion_tokens, total_tokens`.
  - `chat_json(self, model: str, system: str, user: str, schema: type[T], *, step: str = "", max_tokens: int = 2048, temperature: float = 0.1) -> T`.
  - `embed(self, model: str, texts: list[str]) -> list[list[float]]`. This calls `client.embeddings.create` through the same retry loop.
- Behavior:
  - The first attempt uses `response_format={"type":"json_schema","json_schema":{"name":schema.__name__,"schema":schema.model_json_schema(),"strict":False}}`.
  - On `openai.BadRequestError`, set `self.json_schema_ok = False` and use `{"type":"json_object"}` from then on. The system prompt always ends with `"\n\nReturn only JSON matching this schema:\n" + json.dumps(schema)`.
  - Before validation, strip `<think>…</think>` blocks and code fences from the content.
  - On `ValidationError` or `JSONDecodeError`, retry once with the user message plus `"\n\nYour previous reply failed validation: <error>. Return only valid JSON matching the schema."`. A second failure raises `LLMError`.
  - Transport retry: `RateLimitError`, `APIConnectionError`, `APITimeoutError` and `APIStatusError` with status ≥ 500 lead to `sleep(min(8, 0.5 * 2**attempt) + random.uniform(0, 0.25))`. Stop at 4 attempts total, then raise `LLMError("Token Factory request failed after 4 attempts: …")`.
- Produces in `tests/fakes.py`:
  - `FakeOpenAI(responses: list)`. Each item is a content string, a dict `{"content":…, "usage":(p,c)}`, or an exception instance. `.chat.completions.create(**kw)` records `kw` in `.calls`. `.embeddings.create` is also available.
  - `api_error(cls, status)`, which builds openai exceptions with an `httpx.Response`.
  - `ScriptedLLM(by_schema: dict[type, list[BaseModel] | BaseModel])`, which mimics `LLMClient.chat_json`/`embed` and records `calls: list[dict(model, system, user, schema, step)]` and `usage`.

- [ ] **Step 1: Write the failing tests.** These are `tests/test_llm.py`, using a tiny `class Out(BaseModel): x: int`.
  - `test_uses_json_schema_and_low_temperature`: `calls[0]["response_format"]["type"] == "json_schema"`, `calls[0]["temperature"] <= 0.2`, and `calls[0]["max_tokens"] == 2048`.
  - `test_falls_back_to_json_object_on_400`: the responses are `[api_error(BadRequestError, 400), '{"x":1}']`. The result is `.x == 1`, `calls[1]["response_format"] == {"type":"json_object"}` and `client.json_schema_ok is False`.
  - `test_validation_retry_includes_error`: the responses are `['{"x":"nope"}', '{"x":2}']`. The result is 2, and `"failed validation"` appears in `calls[1]["messages"][-1]["content"]`.
  - `test_validation_fails_twice_raises`: two bad replies lead to `pytest.raises(LLMError)`.
  - `test_retries_429_then_succeeds`: the responses are `[429 error, 503 error, '{"x":3}']`. The fake `sleep` was called twice, and both delays are ≤ 8.25.
  - `test_gives_up_after_4_attempts`: there are 5 × 500 errors. `LLMError` is raised and `len(calls) == 4`.
  - `test_records_usage`: `usage=(10,5)` and `step="intake"` lead to `usage[0]` having `{"step":"intake","model":"m","prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`, and `latency_ms >= 0`.
  - `test_strips_think_and_fences`: content `'<think>hmm</think>\n```json\n{"x":4}\n```'` results in 4.
- [ ] **Step 2: Run** `uv run pytest tests/test_llm.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `app/llm.py` and `tests/fakes.py` per Interfaces.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: Token Factory LLM client with structured output, retries, usage`.

### Task 5: Policy corpus and parser

**Files:** Create `data/policies/*.md` (15 files) and `app/retrieval.py` (parser part only). Test: `tests/test_policies.py`.

**Interfaces:**
- Produces: `@dataclass PolicyDoc(policy_id, payer, service, codes: list[str], title: str, body: str, filename: str)` and `parse_policy(text: str, filename: str) -> PolicyDoc`. The front matter is `---`-fenced `key: value` lines, and `codes` is comma-separated. Add no YAML dependency. Missing keys raise `ValueError`. Also `load_policies(dir: Path) -> list[PolicyDoc]`, sorted by filename.
- Produces the corpus file format, where the criteria are numbered lines `N.M <text>` under `## N. Coverage Criteria`:

```
---
policy_id: AH-RAD-012
payer: Acme Health
service: MRI lumbar spine
codes: 72148, 72149, 72158
---
# Acme Health Medical Policy AH-RAD-012: MRI of the Lumbar Spine
## 1. Scope
## 2. Coverage Criteria
2.1 …
## 3. Exclusions
## 4. Documentation Requirements
```

- Corpus (policy_id, filename):
  - **Acme Health:** AH-RAD-012 `acme-mri-lumbar.md`, AH-ORTH-031 `acme-knee-arthroscopy.md`, AH-RX-044 `acme-glp1-t2d.md`, AH-RX-051 `acme-biologics-psoriasis.md`, AH-SLP-007 `acme-sleep-study.md`, AH-DME-019 `acme-cgm.md`, AH-REH-003 `acme-pt-extension.md`, AH-SURG-022 `acme-bariatric.md`.
  - **Northwind Care:** NWC-IMG-104 `northwind-mri-lumbar.md`, NWC-IMG-118 `northwind-ct-abdomen.md`, NWC-PH-210 `northwind-biologics-ra.md`, NWC-DME-305 `northwind-cgm.md`, NWC-SLP-401 `northwind-sleep-study.md`, NWC-RX-220 `northwind-glp1-t2d.md`, NWC-ORTH-512 `northwind-hip-replacement.md`.
  - Each policy has 3–6 criteria with concrete, checkable thresholds (durations, scores, lab values, BMI), modeled on public policy structure. The criteria are distinct between the payers, even for the same service.

- [ ] **Step 1: Write the failing tests**
  - `test_parse_front_matter`, run on an inline sample: `codes == ["72148","72149"]` and `title` starts with `"Acme Health Medical Policy"`.
  - `test_parse_missing_key_raises` checks for `ValueError`.
  - `test_corpus_shape` checks that `len(docs) == 15`, `{d.payer for d in docs} == {"Acme Health","Northwind Care"}` and that the policy_ids are unique. Each body contains `"Coverage Criteria"` and has ≥ 3 lines matching `^\d+\.\d+ ` inside that section.
- [ ] **Step 2: Run** `uv run pytest tests/test_policies.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Write** the parser and the 15 policy files.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: fictional payer policy corpus and parser`.

### Task 6: Policy index (embeddings + BM25 fallback)

**Files:** Modify `app/retrieval.py`. Test: `tests/test_retrieval.py`.

**Interfaces:**
- Consumes: `load_policies`, `Settings`, and `LLMClient.embed`.
- Produces: `@dataclass Chunk(policy_id, payer, service, section: str, text: str, source_doc: str)`. There is one chunk per `## ` section. `section` is the heading text (`"2. Coverage Criteria"`), and `text` is the doc title + heading + body. `source_doc` is the policy_id.
- Produces: `class PolicyIndex`, with:
  - `__init__(chunks, embedder: Callable[[list[str]], list[list[float]]] | None = None, cache_dir: Path | None = None, cache_key: str = "")`.
  - `mode: Literal["embeddings","bm25"]`.
  - `search(query: str, k: int = 4, payer: str | None = None) -> list[Chunk]`. When `payer` case-insensitively equals the payer of any chunk, restrict to that payer's chunks.
  - `@classmethod build(cls, settings: Settings, llm: LLMClient | None) -> PolicyIndex`, which reads from `settings.data_dir / "policies"`. The embedder is `lambda t: llm.embed(settings.model_embed, t)` only when `settings.model_embed` and `llm` are both set.
- Behavior:
  - Embeddings are cached at `cache_dir / f"embeddings-{sha256(cache_key + all chunk texts)[:16]}.npy"`. A cache hit skips chunk embedding.
  - An embedder exception during build logs a warning and falls back to `mode="bm25"`.
  - Similarity is cosine. BM25 uses `rank_bm25.BM25Okapi` on lowercase `\w+` tokens.

- [ ] **Step 1: Write the failing tests**
  - `test_bm25_when_no_embed_model`: `PolicyIndex.build(Settings(_env_file=None, model_embed=""), llm=None)` gives `mode == "bm25"`. A search for `"Acme Health MRI lumbar spine radiculopathy conservative therapy"` with `payer="Acme Health"` returns a first result with `source_doc == "AH-RAD-012"`, and `len(results) == 4`.
  - `test_payer_filter`: the same query with `payer="Northwind Care"` returns results that all have payer Northwind Care.
  - `test_embedding_path_uses_cache(tmp_path)`: the fake embedder returns deterministic vectors (a hashed bag-of-words into 64 dims) and counts the texts it receives. The first build has `mode == "embeddings"` and embeds all chunks. A second build with the same `cache_dir` embeds 0 chunks, and the search embeds only the 1 query text.
  - `test_embed_failure_falls_back`: an embedder that raises gives `mode == "bm25"`.
- [ ] **Step 2: Run** `uv run pytest tests/test_retrieval.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: policy index with embeddings cache and BM25 fallback`.

### Task 7: Deterministic checks

**Files:** Create `app/checks.py`. Test: `tests/test_checks.py`.

**Interfaces:**
- Consumes: `AppealDraft`, `ChartLine`, `Criterion` and `ReviewFlag`.
- Produces:
  - `normalize_id(s: str) -> str` (strip, then upper).
  - `is_boilerplate(text: str) -> bool`. True when the text matches `^\s*(dear|sincerely|respectfully submitted|thank you|regards|re:|to whom|cc:|enclosures?)\b` (case-insensitive), or when it has fewer than 4 words and no digit.
  - `is_factual(text: str, chart_text: str) -> bool`. False for boilerplate. Otherwise true when the text has a digit, or shares ≥ 2 content words (`\w{5,}`, lowercase, excluding a small stopword set) with `chart_text`.
  - `deterministic_checks(draft: AppealDraft, chart_lines: list[ChartLine], criteria: list[Criterion], denial_text: str = "") -> list[ReviewFlag]`.
  - `merge_flags(*lists: list[ReviewFlag]) -> list[ReviewFlag]`. This keeps one flag per sentence_id in first-seen order and joins distinct issues with `"; "`. `remove` wins over `revise`.
- Rules (one flag per sentence, issues joined):
  1. A normalized cite that is not in the line IDs or criterion IDs adds `"cites unknown ID <id>"`, with action `revise`.
  2. A factual sentence with no cites adds `"factual claim without citation"`, with action `revise`.
  3. Numbers: digit runs (`\d+`, with leading zeros stripped, so `03/14/2025` gives `{3,14,2025}`) must all appear among the digit runs of the sources. For a cited sentence, the sources are the cited lines' text plus the cited criteria's `text`, `source_section` and `source_doc`. For an uncited sentence, they are `denial_text` plus all chart text. Otherwise add `"number/date not found in cited sources: <n>"`, with action `remove`.

- [ ] **Step 1: Write the failing tests.** These use fixtures with lines L1 "Low back pain radiating to left leg for 8 weeks." and L2 "Completed 6 weeks of physical therapy ending 09/02/2026.", plus criterion C1 (section "2.1", text "At least 6 weeks of conservative therapy").
  - `test_unknown_id_flagged`: cites `["L9"]` gives a flag with `"unknown ID L9"`.
  - `test_ids_normalized`: cites `[" l2", "c1 "]` on "The patient completed 6 weeks of physical therapy." gives no flags.
  - `test_uncited_factual_flagged`: "The patient reported radiating pain for 8 weeks." with `[]` gives `"without citation"`.
  - `test_boilerplate_not_flagged`: "Dear Appeals Committee," and "Sincerely," with `[]` give no flags.
  - `test_number_not_in_cited_line_flagged`: "Therapy ended 03/14/2025." with `["L2"]` gives action `remove`.
  - `test_date_leading_zero_ok`: "Therapy ended 9/2/2026." with `["L2"]` gives no flags.
  - `test_merge_dedupes_remove_wins`: `merge_flags([S1 revise "a"], [S1 remove "b", S2 revise "c"])` gives `[S1 remove "a; b", S2 revise "c"]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_checks.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: deterministic citation and number checks`.

### Task 8: Agents and Tavily tool

**Files:** Create `app/agents/__init__.py`, `intake.py`, `policy.py`, `matcher.py`, `writer.py` and `reviewer.py`, plus `app/tools/tavily_search.py`. Test: `tests/test_agents.py`.

**Interfaces:**
- Consumes: `LLMClient.chat_json` (`ScriptedLLM` in tests), `PolicyIndex.search`, the schemas and `normalize_id`.
- Produces (each module exposes `SYSTEM: str` and `run(...)`). Chart lines are rendered into prompts as `"L1: text"` lines.
  - `intake.run(llm, settings, denial_text: str) -> Denial`. Uses `model_fast` and step `"intake"`.
  - `policy.run(llm, settings, index, denial: Denial, tavily=None) -> tuple[list[Criterion], list[dict]]`.
    - It searches `f"{denial.payer} {denial.service} {denial.denial_reason}"` with `k=4` and `payer=denial.payer`, then calls `model_fast` (step `"policy"`) with schema `CriteriaList` over the chunk texts.
    - It **renumbers ids to `C1..Cn`** in order. It keeps the policy's own `N.M` in `source_section` and sets `source_doc` to the policy_id when the model leaves it blank.
    - `tavily` is `Callable[[str], list[dict]] | None`. Its results come back as the second element only, never as criteria.
  - `matcher.run(llm, settings, criteria, chart_lines) -> list[CriterionMatch]`. Uses `model_reason`, step `"match"` and schema `MatchList`.
    - Post-processing normalizes IDs and drops evidence IDs that aren't in the chart.
    - A `met` match left with no evidence becomes `unclear`, with `" (evidence citation invalid)"` appended to the rationale.
    - Every criterion appears exactly once, in criteria order. Missing ones become `unclear` with rationale `"No determination returned."`.
  - `writer.run(llm, settings, denial, criteria, matches, chart_lines) -> AppealDraft`. Uses `model_reason` and step `"draft"`. The system prompt states the spec's writer rules: cite every factual sentence; argue only from `met`; request peer-to-peer review for `not_met`/`unclear`; leave boilerplate cites empty. Sentence ids are renumbered `S1..Sn`.
  - `reviewer.run(llm, settings, draft, chart_lines, criteria) -> Review`. Uses `model_reason` and step `"review"`, in a fresh context. The user prompt contains only the draft sentences (id, text, cites), the chart lines and the criteria. It contains no match rationales and no writer prompt. Flags for unknown sentence ids are dropped.
  - `tavily_search.search(query: str, api_key: str, max_results: int = 5) -> list[dict]`. Each dict has keys `title, url, snippet`. It returns `[]` on any exception.

- [ ] **Step 1: Write the failing tests** (`ScriptedLLM` and BM25 index from Task 6):
  - `test_intake_uses_fast_model`: `calls[0]["model"] == settings.model_fast` and the denial text is in `calls[0]["user"]`.
  - `test_policy_query_and_renumber`: the scripted `CriteriaList` has ids `"2.1","2.2"`, which become `["C1","C2"]`. The `source_section` values are kept, and the user prompt contains `"AH-RAD-012"`.
  - `test_policy_tavily_kept_separate`: `tavily=lambda q: [{"title":"T","url":"u","snippet":"s"}]` gives `refs == [...]`, and no criterion text contains `"T"`.
  - `test_matcher_drops_bad_ids_and_downgrades`: a scripted match `met` with `[" l2","L99"]` against 3 chart lines gives `["L2"]`, status `met`. A match `met` with `["L99"]` gives `unclear`, `[]`.
  - `test_matcher_fills_missing`: 2 criteria with 1 scripted match give 2 results, the second `unclear`.
  - `test_writer_rules_and_renumber`: `"peer-to-peer"` appears in `SYSTEM`, the model is `model_reason`, and the ids are `S1..Sn`.
  - `test_reviewer_gets_no_writer_reasoning`: the match rationale `"SECRET-RATIONALE"` doesn't appear in `calls[-1]["system"] + calls[-1]["user"]`.
  - `test_tavily_search_swallows_errors`: patching `TavilyClient` to raise gives `[]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_agents.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: intake, policy, matcher, writer, reviewer agents`.

### Task 9: Pipeline

**Files:** Create `app/pipeline.py`. Test: `tests/test_pipeline.py`.

**Interfaces:**
- Consumes: the Task 8 agents, `deterministic_checks`, `merge_flags` and `tavily_search.search`.
- Produces: `async def run_pipeline(case: Case, llm, settings: Settings, index: PolicyIndex) -> AsyncIterator[dict]`.
  - It mutates `case` and yields `{"step", "status", "data"}` events.
  - The steps run in order: `preprocess`, `intake`, `policy`, `match`, `draft`, `review` and `checks`. Each yields `running`, then `done` with that step's model_dump data.
  - It ends with `{"step":"complete","status":"needs_review","data":{"case_id":…}}`.
  - Agents run through `asyncio.to_thread`. `case.status` is `"running"` during the run, and `case.usage = llm.usage` after each step.
  - `case.review = Review(flags=merge_flags(reviewer.flags, det_flags), summary=reviewer.summary)`.
  - Tavily runs only when `settings.enable_tavily and settings.tavily_api_key`.
  - Zero criteria raise `"No matching coverage criteria found for <payer> / <service>."`.
  - Any exception yields `{"step": <current>, "status":"error", "data":{"message": str(e)}}`, sets `case.status = "error"` and stops.

- [ ] **Step 1: Write the failing tests** (`ScriptedLLM` scripts one full happy path; collect events with `[e async for e in run_pipeline(...)]`):
  - `test_event_order`: the `(step, status)` pairs equal `[(s,"running"),(s,"done")]` for each step, then `("complete","needs_review")`.
  - `test_case_populated`: `case.status == "needs_review"` and `case.denial`, `case.criteria`, `case.matches`, `case.draft` and `case.review` are set. `case.usage` has ≥ 5 entries.
  - `test_deterministic_flags_merged`: a writer sentence citing `"L99"` adds a `case.review.flags` entry for its id, even though the reviewer returned none.
  - `test_error_event`: an intake script raising `LLMError("boom")` makes the last event `{"step":"intake","status":"error","data":{"message":"boom"}}`, and `case.status == "error"`.
  - `test_no_criteria_error`: an empty `CriteriaList` makes the last event status `error`, with `"No matching coverage criteria"` in the message.
- [ ] **Step 2: Run** `uv run pytest tests/test_pipeline.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: streaming agent pipeline`.

### Task 10: Store and API

**Files:** Create `app/store.py` and `app/main.py`. Test: `tests/test_api.py`.

**Interfaces:**
- Produces in `store.py`:
  - `class CaseStore(dump_dir: Path | None)`, with:
    - `create(denial_text: str, chart_text: str) -> Case`. The id is `uuid4().hex[:12]`, and the chart is split via `split_chart_lines`.
    - `get(case_id) -> Case`, which raises `KeyError`.
    - `save(case)`, which writes `<dump_dir>/cases/<id>.json` when `dump_dir` is set.
  - `list_samples(data_dir) -> list[dict]`. Each dict is `{"id": "case_01", "title": expected["title"]}`, sorted.
  - `load_sample(data_dir, sample_id) -> tuple[str, str]`. Reads `denial.txt` and `chart.txt`, and raises `KeyError` when they're missing. The `sample_id` must match `^case_\d{2}$`, which prevents path traversal.
  - `assemble_letter(case: Case) -> str`. Drops sentences whose `flag_resolutions[id] == "remove"`. Boilerplate sentences (`is_boilerplate`) sit on their own lines. Consecutive non-boilerplate sentences are joined with a space into one paragraph, and paragraphs are separated by a blank line.
  - `class RateLimiter(max_per_hour: int, clock=time.time)`, with `.hit(key) -> bool` (a sliding 3600 s window).
- Produces in `main.py`: `create_app(settings: Settings | None = None, llm_factory: Callable[[], Any] | None = None) -> FastAPI`, and the module-level `app = create_app()`.
  - The default `llm_factory` is `lambda: LLMClient(settings)`, giving a fresh client per run so usage is per case.
  - `PolicyIndex.build` runs in the lifespan (with an `LLMClient` only when `model_embed` is set).
  - Mount `static/`, and serve `GET /` → `static/index.html`.
- Routes (the error body is always `{"detail": "<readable message>"}`):
  - `GET /healthz` returns `{"ok": true}`.
  - `GET /api/samples`.
  - `POST /api/cases` takes JSON `{"sample_id"}` (404 for an unknown one) or multipart `denial` + `chart` files (each ≤ 5 MB, else 413; `ValueError` from `extract_text` returns 400). It returns `{"case_id"}`.
  - `GET /api/cases/{id}` returns the case JSON, or 404.
  - `GET /api/cases/{id}/run` returns 404 for an unknown case, 409 unless the status is `new` or `error`, and 429 when `RateLimiter.hit(ip)` is false. The IP is the first `X-Forwarded-For` entry, else `request.client.host`. Otherwise it returns a `StreamingResponse(media_type="text/event-stream")` with `data: <json>\n\n` per event, and saves the case at the end.
  - `PATCH /api/cases/{id}/letter` takes the body `{"sentence_id": str, "text": str | None, "flag_action": "keep"|"remove"|None}`. It returns 409 unless the status is `needs_review`, and 404 for an unknown sentence. It returns the updated case.
  - `POST /api/cases/{id}/approve` returns 409 unless the status is `needs_review`. It also returns 409 with `"Resolve all flags before approving"` when any flagged sentence_id has no resolution. Otherwise it sets `final_letter = assemble_letter(case)` and `status = "approved"`, and returns the case.
  - `GET /api/cases/{id}/export?format=txt|md` returns 409 unless approved. Otherwise it returns `PlainTextResponse` with `Content-Disposition: attachment; filename="appeal-<id>.<fmt>"`.

- [ ] **Step 1: Write the failing tests.** Use `TestClient(create_app(Settings(_env_file=None, data_dir=tmp_data, cache_dir=tmp_path, max_runs_per_ip_per_hour=1), llm_factory=lambda: scripted()))` as a context manager. `tmp_data` holds a copy of `data/policies` plus one sample `case_01` with a small denial/chart/expected.json written inline. The scripted writer produces sentences S1 (boilerplate), S2 (valid) and S3 (cites L99, so it gets flagged).
  - `test_healthz`.
  - `test_samples_and_create_from_sample` checks for the `case_01` title, `case_id` returned, and `GET` status `new`.
  - `test_unknown_sample_404` covers `{"sample_id":"case_99"}` and `{"sample_id":"../x"}`, both 404.
  - `test_upload_txt` checks 200 and that the chart lines were split.
  - `test_upload_blank_pdf_400` checks that `"No extractable text"` is in the detail.
  - `test_upload_too_large_413`.
  - `test_run_streams_to_needs_review` parses the `data:` lines. The last one is `needs_review`, and `GET` shows the review flags include S3.
  - `test_rerun_conflict_409` checks that a run after `needs_review` returns 409.
  - `test_rate_limit_429` creates two cases. The second case's run returns 429.
  - `test_approve_blocked_until_resolved` gets 409 first. After a PATCH with S3 `remove`, approve returns 200, the S3 text is absent from `final_letter`, and the S2 text is present.
  - `test_patch_edits_text` checks that the text is updated in the draft.
  - `test_export_requires_approved` gets 409 before approval. After approval it gets 200, an `attachment` header and the letter body.
  - `test_approved_case_untouched_by_run` checks that a run on an approved case returns 409 and leaves `final_letter` unchanged.
- [ ] **Step 2: Run** `uv run pytest tests/test_api.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Create a placeholder `static/index.html` containing the banner copy so `/` works. Task 12 replaces it.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: case store and HTTP API with SSE, rate limit, approve/export`.

### Task 11: Synthetic cases and demo PDFs

**Files:** Create `data/cases/case_01..case_10/{denial.txt,chart.txt,expected.json}`, `scripts/make_pdfs.py` and `data/cases/case_0{1,2,3}/{denial,chart}.pdf`. Test: `tests/test_cases.py`.

**Interfaces:**
- Consumes: `split_chart_lines`, `load_policies`, `extract_text` and `list_samples`.
- Produces the `expected.json` format, which the Task 14 eval consumes:

```json
{"title": "Acme Health · Lumbar MRI", "policy_id": "AH-RAD-012", "category": "supported", "appealable": true,
 "denial": {"payer": "Acme Health", "member_id": "AH-48213907", "patient_name": "…", "service": "MRI lumbar spine",
            "cpt_or_hcpcs": "72148", "icd10": ["M54.16"], "denial_reason": "…", "denial_date": "2026-09-18", "appeal_deadline": "2026-11-17"},
 "criteria": [{"section": "2.1", "status": "met", "evidence": ["L4", "L9"]}]}
```

- Case mix. **supported** cases have `appealable: true`. **missing** cases have one criterion `unclear` or `not_met` and `appealable: false`. **contradicted** cases have one criterion `not_met` and `appealable: false`.
  - **Supported:** 01 AH-RAD-012, 02 AH-RX-044, 03 NWC-DME-305, 04 AH-ORTH-031, 05 NWC-PH-210, 06 AH-SLP-007, 07 NWC-IMG-118.
  - **Missing:** 08 AH-REH-003 (no objective functional measures), 09 NWC-ORTH-512 (no documented duration of conservative therapy).
  - **Contradicted:** 10 AH-SURG-022 (chart BMI 32.1, below the threshold).
- Each `denial.txt` names its payer, policy_id, a fictional member ID, CPT/HCPCS, ICD-10, reason, denial date and appeal deadline. All names are obviously synthetic. `chart.txt` has 20–40 lines as split.
- `scripts/make_pdfs.py`: `render_case(case_dir: Path) -> None` writes `denial.pdf` and `chart.pdf` with reportlab (one text line per source line, wrapped). `main()` renders case_01..03.

- [ ] **Step 1: Write the failing tests**
  - `test_ten_cases_and_mix`: `len(list_samples(Path("data"))) == 10`, and `Counter(category) == {"supported":7,"missing":2,"contradicted":1}`.
  - `test_chart_length_and_evidence_ids_exist` (parametrized): `20 <= len(lines) <= 40`, and every expected evidence ID is in the split ids.
  - `test_policy_and_sections_exist`: the policy_id is in the corpus, and every expected `section` appears as a `^N.M ` criterion line in that policy. The policy_id is in `denial.txt`.
  - `test_expected_denial_validates`: `Denial.model_validate(expected["denial"])`.
  - `test_demo_pdfs_extract` (cases 01–03): `extract_text(chart.pdf)` contains the patient's last name, and `split_chart_lines` on the PDF text gives the same number of lines as the txt.
- [ ] **Step 2: Run** `uv run pytest tests/test_cases.py -q`. Expected: FAIL.
- [ ] **Step 3: Write** the 10 cases and `make_pdfs.py`, then run `uv run python scripts/make_pdfs.py`. Expected: 6 PDFs written.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: 10 synthetic cases and demo PDFs`.

### Task 12: UI

**Files:** Replace `static/index.html`. Test: `tests/test_ui.py`.

**Interfaces:**
- Consumes: the Task 10 routes and the Task 9 event shapes.
- Element ids, which the tests pin: `banner`, `sample-select`, `denial-file`, `chart-file`, `run-btn`, `step-tracker` (children `data-step` = intake, policy, match, draft, review), `denial-card`, `criteria-table`, `external-refs` (list of `case.external_refs` as links, hidden when empty, labeled "External references (not used for matching)"), `chart-panel`, `letter`, `approve-btn`, `download-btn`, `usage-footer`.
- Behavior (from the spec's UI section):
  - **Run:** POST, then `EventSource` on `/run`, updating the tracker per event. On `complete`, fetch the case and render it. On `error`, show the message.
  - **Denial card:** highlights `appeal_deadline`.
  - **Criteria table:** clicking a row toggles the `hl` class on the cited `#chart-panel [data-line=Ln]` rows and scrolls to the first one.
  - **Letter:** each sentence is a `contenteditable` span. Blur PATCHes `text`. `title` shows the cites on hover.
  - **Flagged sentences:** shown with class `flagged` (red strike-through), the reason and **Keep**/**Remove** buttons, which PATCH `flag_action`.
  - **Approve:** disabled while any flag is unresolved. Download is enabled only after approval and links to `/export?format=txt`.
  - **Footer:** sums usage per step (model, tokens, latency).
  - **Style:** system font stack, one accent color (`--accent: #2f6fde`), max-width 1200px, a two-column criteria/chart grid that stacks below 900px.
  - **Text rendering:** all dynamic text is set with `textContent` or `createElement`. Never use `innerHTML`.

- [ ] **Step 1: Write the failing tests**
  - `test_index_served_with_banner`: `GET /` returns 200, and the exact banner copy is in the body.
  - `test_required_ids_present`: every id above is in the html.
  - `test_no_innerhtml`: `"innerHTML" not in html`.
- [ ] **Step 2: Run** `uv run pytest tests/test_ui.py -q`. Expected: FAIL. `test_required_ids_present` fails on the placeholder.
- [ ] **Step 3: Implement** `static/index.html`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Manual smoke.** Run `uv run uvicorn app.main:app --port 8080` and load `/`. Expected: the page renders, the samples dropdown has 10 entries, and Run with no API key shows a readable error event, not a blank page.
- [ ] **Step 6: Commit** with message `feat: single-page UI`.

### Task 13: Model discovery script

**Files:** Create `scripts/__init__.py` and `scripts/discover_models.py`. Test: `tests/test_discover.py`.

**Interfaces:**
- Produces:
  - `classify(ids: list[str]) -> dict`, returning `{"nemotron": [...], "embedding": [...]}`. Matching is case-insensitive: `"nemotron"` in the id, or `embed`/`e5`/`bge`/`retriever` for embeddings.
  - `probe(client, model) -> dict[str, bool]`. Its keys are `json_schema` and `tools`. Each runs one tiny call; any exception or non-conforming reply counts as False.
  - `recommend(nemotron, probes, embedding) -> dict`, with keys `MODEL_FAST`, `MODEL_REASON` and `MODEL_EMBED`:
    - FAST is the first id containing `nano` or `lightning` that passes json_schema, else any passing id.
    - REASON is the first containing `super`, then `ultra`, that passes json_schema, else FAST.
    - EMBED is the first embedding id containing `nemotron` or `nv`, else the first embedding, else `""`.
    - A model that fails json_schema but passes json_object is still eligible: probe it with json_object too, and record the key `json_object`.
  - `main()` uses `Settings` and prints all ids (marking Nemotron and embedding ones), a probe table and `.env` lines.

- [ ] **Step 1: Write the failing tests**
  - `test_classify`, with canned ids `["nvidia/Nemotron-3-Super-120B", "nvidia/Nemotron-Nano-9B-v2", "BAAI/bge-en-icl", "meta/llama-3"]`.
  - `test_recommend_prefers_nano_and_super`.
  - `test_recommend_falls_back_when_super_fails`: REASON becomes the nano id.
  - `test_recommend_no_embedding_gives_empty`.
  - `test_probe_failure_is_false`: a client that raises gives both values False.
- [ ] **Step 2: Run** `uv run pytest tests/test_discover.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: Token Factory model discovery script`.

### Task 14: Eval harness

**Files:** Create `eval/__init__.py` and `eval/run_eval.py`. Test: `tests/test_eval.py`.

**Interfaces:**
- Consumes: `run_pipeline`, `reviewer.run`, `deterministic_checks`, `merge_flags`, `CaseStore`, `PolicyIndex.build`, `LLMClient` and the Task 11 `expected.json`.
- Produces:
  - `norm(s) -> str` (lowercase, alphanumerics only; `None` becomes `""`).
  - `extraction_accuracy(pred: Denial, expected: dict) -> float`. This is the share of the 9 fields matched, comparing `norm` values. `icd10` compares as a set of normed codes.
  - `criterion_scores(criteria, matches, expected_criteria) -> dict`, with keys `status_acc` and `evidence_precision`. It joins predicted criteria to expected ones by `norm(source_section)`. An unmatched expected criterion counts as wrong. Precision is the share of cited IDs that fall in the expected evidence, and it is 1.0 when nothing was cited.
  - `citation_validity(draft, chart_lines, criteria, denial_text) -> float`. This is the share of `is_factual` sentences with no deterministic flag (1.0 if there are none).
  - `inject_fabrications(draft, chart_lines) -> tuple[AppealDraft, list[str]]`. It appends `F1` ("The patient was last evaluated on 02/29/2019, confirming the findings above." citing `[chart_lines[0].id]`) and `F2` ("The patient also reports intermittent hemoptysis." citing `[chart_lines[0].id]`).
  - `catch_rate(flags, fabricated_ids) -> float`.
  - `async run_case(case_dir, settings, llm, index) -> dict` (one result row).
  - `to_markdown(rows) -> str`, a per-case table plus a mean row with columns Case, Extraction, Status acc, Evidence precision, Citation validity, Catch rate, Latency s, Tokens.
  - `main()` writes `eval/results.md` and prints it. It exits with a readable message if `NEBIUS_API_KEY` or the models are unset.

- [ ] **Step 1: Write the failing tests**
  - `test_extraction_accuracy_normalizes`: `"ACME health"` vs `"Acme Health"` matches, and the icd10 order doesn't matter. One wrong field out of 9 gives `8/9`.
  - `test_criterion_scores`: 2 expected criteria. One has a correct status with evidence `["L1","L5"]`, where L5 isn't expected. The other is missing from the prediction. The result is `status_acc == 0.5` and `evidence_precision == 0.5`.
  - `test_injected_date_caught_by_checks`: `deterministic_checks` on the injected draft flags `F1`.
  - `test_catch_rate`: flags on F1 only give 0.5.
  - `test_to_markdown_has_mean_row`.
- [ ] **Step 2: Run** `uv run pytest tests/test_eval.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat: eval harness`.

### Task 15: Docker, README, feedback

**Files:** Create `Dockerfile`, `.dockerignore`, `requirements.txt`, `README.md` and `FEEDBACK.md`. Test: `tests/test_docs.py`.

**Interfaces:**
- `requirements.txt` is generated with `uv export --no-dev --no-hashes --format requirements-txt -o requirements.txt`.
- `Dockerfile`: `FROM python:3.11-slim`, `WORKDIR /app`, `COPY requirements.txt .`, `RUN pip install --no-cache-dir -r requirements.txt`, `COPY app static data scripts ./…` (keeping the dirs), `ENV PORT=8080`, `EXPOSE 8080`, `CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]`.
- README sections (the headings are exact, because the tests pin them):
  - `## What it is`
  - `## Architecture` (a Mermaid `flowchart` of preprocess → intake[MODEL_FAST] → policy[retrieval + MODEL_FAST] → matcher[MODEL_REASON] → writer[MODEL_REASON] → reviewer[MODEL_REASON] → deterministic checks → human approval)
  - `## How Nebius and NVIDIA are used` (endpoint, model IDs per step, embeddings, structured outputs, the reason for tiering)
  - `## Eval results`
  - `## Setup`
  - `## Disclaimer`
  - `## License`
  - Model IDs and eval numbers stay as clearly marked `TODO (run scripts/discover_models.py)` placeholders until they are run with a key.
- `FEEDBACK.md` lists the Token Factory/Nemotron friction recorded during the build (json_schema support, `<think>` output, etc.).

- [ ] **Step 1: Write the failing tests**
  - `test_license_is_mit`.
  - `test_readme_sections`: every heading above is present, along with ` ```mermaid ` and `api.tokenfactory.nebius.com`.
  - `test_requirements_in_sync`: every `[project].dependencies` name (lowercased, before any `[`, `>` or `=`) appears in `requirements.txt`.
  - `test_dockerfile_cmd`: `python:3.11-slim` and `app.main:app` are in the Dockerfile.
- [ ] **Step 2: Run** `uv run pytest tests/test_docs.py -q`. Expected: FAIL.
- [ ] **Step 3: Write** the files.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `docs: README, Dockerfile, feedback notes`.
