from typing import Literal

from pydantic import BaseModel


class ChartLine(BaseModel):
    id: str  # "L1", "L2", ...
    text: str


class Denial(BaseModel):
    payer: str
    member_id: str | None
    patient_name: str
    service: str
    cpt_or_hcpcs: str | None
    icd10: list[str]
    denial_reason: str
    denial_date: str | None
    appeal_deadline: str | None


class Criterion(BaseModel):
    id: str
    text: str
    source_doc: str
    source_section: str


class CriterionMatch(BaseModel):
    criterion_id: str
    status: Literal["met", "not_met", "unclear"]
    evidence_line_ids: list[str]
    rationale: str


class LetterSentence(BaseModel):
    id: str
    text: str
    cites: list[str]  # line ids / criterion ids


class AppealDraft(BaseModel):
    sentences: list[LetterSentence]


class ReviewFlag(BaseModel):
    sentence_id: str
    issue: str
    action: Literal["remove", "revise"]


class Review(BaseModel):
    flags: list[ReviewFlag]
    summary: str


class Case(BaseModel):
    id: str
    denial_text: str
    chart_lines: list[ChartLine]
    denial: Denial | None = None
    criteria: list[Criterion] = []
    matches: list[CriterionMatch] = []
    draft: AppealDraft | None = None
    review: Review | None = None
    final_letter: str | None = None
    status: Literal["new", "running", "needs_review", "approved", "error"] = "new"
    usage: list[dict] = []
    # Human decision per flagged sentence id.
    flag_resolutions: dict[str, Literal["keep", "remove"]] = {}
    # Tavily results: display only, never used for matching.
    external_refs: list[dict] = []


# LLM wrapper schemas: structured outputs need an object at the top level.
class CriteriaList(BaseModel):
    criteria: list[Criterion]


class MatchList(BaseModel):
    matches: list[CriterionMatch]
