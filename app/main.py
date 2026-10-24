"""FastAPI app and routes."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from app.config import Settings, get_settings
from app.llm import LLMClient
from app.pipeline import run_pipeline
from app.retrieval import PolicyIndex
from app.schemas import Case
from app.store import CaseStore, RateLimiter, assemble_letter, list_samples, load_sample
from app.tools.pdf import extract_text

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
HEARTBEAT_S = 15


class LetterEdit(BaseModel):
    sentence_id: str
    text: str | None = None
    flag_action: Literal["keep", "remove"] | None = None


def _client_ip(request: Request, trust_proxy: bool = False) -> str:
    # X-Forwarded-For is client-controlled; only a trusted proxy's appended hop (the rightmost) counts.
    forwarded = request.headers.get("x-forwarded-for")
    if trust_proxy and forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


async def stream_events(source: AsyncIterator[dict], case: Case, save: Callable[[Case], None],
                        heartbeat_s: float = HEARTBEAT_S) -> AsyncIterator[str]:
    """SSE frames for pipeline events, with comment pings while a stage is silent so proxies keep the stream open.
    If the client disconnects, the run is cancelled and the case is left re-runnable ("error"), never "running"."""
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for event in source:
                queue.put_nowait(event)
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), heartbeat_s)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        task.cancel()
        if case.status == "running":
            case.status = "error"
        save(case)


async def _read_upload(form, field: str) -> str:
    upload = form.get(field)
    if not isinstance(upload, UploadFile):
        raise HTTPException(400, f"Missing file: {field}")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename} is larger than 5 MB")
    try:
        return extract_text(data, upload.filename or field)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception:
        raise HTTPException(400, f"Could not read {upload.filename}") from None


def create_app(settings: Settings | None = None, llm_factory: Callable[[], Any] | None = None) -> FastAPI:
    settings = settings or get_settings()
    llm_factory = llm_factory or (lambda: LLMClient(settings))
    store = CaseStore(dump_dir=settings.cache_dir)
    limiter = RateLimiter(settings.max_runs_per_ip_per_hour)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Built once at startup; embeddings are cached to disk by PolicyIndex.
        app.state.index = PolicyIndex.build(settings, llm_factory() if settings.model_embed else None)
        log.info("Policy index ready (%s, %d chunks)", app.state.index.mode, len(app.state.index.chunks))
        yield

    app = FastAPI(title="Appeal-Pilot", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_case(case_id: str) -> Case:
        try:
            return store.get(case_id)
        except KeyError:
            raise HTTPException(404, "Case not found") from None

    @app.get("/")
    def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/samples")
    def samples():
        return list_samples(settings.data_dir)

    @app.post("/api/cases")
    async def create_case(request: Request):
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            try:
                denial_text, chart_text = load_sample(settings.data_dir, str(body.get("sample_id", "")))
            except KeyError:
                raise HTTPException(404, "Unknown sample case") from None
        else:
            form = await request.form()
            denial_text = await _read_upload(form, "denial")
            chart_text = await _read_upload(form, "chart")
        case = store.create(denial_text, chart_text)
        if not case.chart_lines:
            raise HTTPException(400, "The chart notes contain no text lines")
        return {"case_id": case.id}

    @app.get("/api/cases/{case_id}")
    def read_case(case_id: str):
        return get_case(case_id)

    @app.get("/api/cases/{case_id}/run")
    async def run_case(case_id: str, request: Request):
        case = get_case(case_id)
        if case.status not in ("new", "error"):
            raise HTTPException(409, f"Case is {case.status}; it cannot be run again")
        if not limiter.hit(_client_ip(request, settings.trust_proxy)):
            raise HTTPException(429, "Run limit reached for this hour. Please try again later.")
        case.status = "running"
        source = run_pipeline(case, llm_factory(), settings, request.app.state.index)
        return StreamingResponse(stream_events(source, case, store.save), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.patch("/api/cases/{case_id}/letter")
    def edit_letter(case_id: str, edit: LetterEdit):
        case = get_case(case_id)
        if case.status != "needs_review" or not case.draft:
            raise HTTPException(409, "The letter can only be edited while it awaits review")
        sentence = next((s for s in case.draft.sentences if s.id == edit.sentence_id), None)
        if sentence is None:
            raise HTTPException(404, "Sentence not found")
        if edit.text is not None:
            sentence.text = edit.text.strip()
        if edit.flag_action is not None:
            case.flag_resolutions[edit.sentence_id] = edit.flag_action
        store.save(case)
        return case

    @app.post("/api/cases/{case_id}/approve")
    def approve(case_id: str):
        case = get_case(case_id)
        if case.status != "needs_review":
            raise HTTPException(409, f"Case is {case.status}; only a reviewed draft can be approved")
        flagged = {f.sentence_id for f in case.review.flags} if case.review else set()
        if flagged - case.flag_resolutions.keys():
            raise HTTPException(409, "Resolve all flags before approving")
        case.final_letter = assemble_letter(case)
        case.status = "approved"
        store.save(case)
        return case

    @app.get("/api/cases/{case_id}/export")
    def export(case_id: str, format: Literal["txt", "md"] = "txt"):
        case = get_case(case_id)
        if case.status != "approved" or case.final_letter is None:
            raise HTTPException(409, "Approve the letter before exporting")
        return PlainTextResponse(case.final_letter, media_type="text/markdown" if format == "md" else "text/plain",
                                 headers={"Content-Disposition": f'attachment; filename="appeal-{case.id}.{format}"'})

    return app


app = create_app()
