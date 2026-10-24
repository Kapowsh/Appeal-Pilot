"""Run the pipeline on every bundled case and report extraction, matching, citation and reviewer metrics.

Usage: uv run python eval/run_eval.py      (needs NEBIUS_API_KEY and MODEL_* in .env)
Writes eval/results.md.
"""
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import reviewer  # noqa: E402
from app.checks import deterministic_checks, is_factual, merge_flags  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402
from app.retrieval import PolicyIndex  # noqa: E402
from app.schemas import AppealDraft, ChartLine, Criterion, CriterionMatch, Denial, LetterSentence, ReviewFlag  # noqa: E402
from app.store import CaseStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["payer", "member_id", "patient_name", "service", "cpt_or_hcpcs", "icd10", "denial_reason",
          "denial_date", "appeal_deadline"]
FREE_TEXT = {"service", "denial_reason"}  # paraphrase is fine: scored by token overlap
DATES = {"denial_date", "appeal_deadline"}
_STOP = {"the", "of", "a", "an", "and", "or", "for", "to", "in", "on", "is", "are", "was", "be", "with", "by"}
COLUMNS = [("case", "Case"), ("extraction", "Extraction"), ("status_acc", "Status acc"),
           ("evidence_precision", "Evidence precision"), ("citation_validity", "Citation validity"),
           ("catch_rate", "Catch rate"), ("latency_s", "Latency s"), ("tokens", "Tokens")]


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s is not None else ""


def _date(s) -> str:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return norm(s)


def _overlap(a, b) -> float:
    ta = {w for w in re.findall(r"\w+", str(a or "").lower()) if w not in _STOP}
    tb = {w for w in re.findall(r"\w+", str(b or "").lower()) if w not in _STOP}
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def _field_ok(field, pred, exp) -> bool:
    if field == "icd10":
        return {norm(x) for x in pred or []} == {norm(x) for x in exp or []}
    if field in FREE_TEXT:
        return _overlap(pred, exp) >= 0.5
    if field in DATES:
        return (_date(pred) if pred else "") == (_date(exp) if exp else "")
    return norm(pred) == norm(exp)


def extraction_accuracy(pred: Denial, expected: dict) -> float:
    return sum(_field_ok(f, getattr(pred, f), expected.get(f)) for f in FIELDS) / len(FIELDS)


def criterion_scores(criteria: list[Criterion], matches: list[CriterionMatch], expected_criteria: list[dict]) -> dict:
    by_criterion = {m.criterion_id: m for m in matches}
    by_section = {norm(c.source_section): by_criterion.get(c.id) for c in criteria}
    correct, cited, cited_ok = 0, 0, 0
    for exp in expected_criteria:
        m = by_section.get(norm(exp["section"]))
        if m is None:
            continue
        if m.status in exp.get("accept", [exp["status"]]):
            correct += 1
        cited += len(m.evidence_line_ids)
        cited_ok += sum(e in exp["evidence"] for e in m.evidence_line_ids)
    return {"status_acc": correct / len(expected_criteria) if expected_criteria else 1.0,
            "evidence_precision": cited_ok / cited if cited else 1.0}


def citation_validity(draft: AppealDraft, chart_lines: list[ChartLine], criteria: list[Criterion],
                      denial_text: str) -> float:
    chart_text = " ".join(l.text for l in chart_lines)
    factual = [s for s in draft.sentences if is_factual(s.text, chart_text)]
    if not factual:
        return 1.0
    flagged = {f.sentence_id for f in deterministic_checks(draft, chart_lines, criteria, denial_text)}
    return sum(s.id not in flagged for s in factual) / len(factual)


def inject_fabrications(draft: AppealDraft, chart_lines: list[ChartLine]) -> tuple[AppealDraft, list[str]]:
    """A false date (02/29/2019 does not exist) and a symptom absent from every chart.
    Both cite a mid-chart clinical line, not the L1 "SYNTHETIC RECORD" banner, so catching them is not trivial."""
    cite = [chart_lines[len(chart_lines) // 2].id]
    fabricated = [
        LetterSentence(id="F1", text="The patient was last evaluated on 02/29/2019, confirming the findings above.",
                       cites=cite),
        LetterSentence(id="F2", text="The patient also reports intermittent hemoptysis.", cites=cite),
    ]
    return AppealDraft(sentences=[*draft.sentences, *fabricated]), [s.id for s in fabricated]


def catch_rate(flags: list[ReviewFlag], fabricated_ids: list[str]) -> float:
    flagged = {f.sentence_id for f in flags}
    return sum(i in flagged for i in fabricated_ids) / len(fabricated_ids)


async def run_case(case_dir: Path, settings: Settings, llm, index: PolicyIndex) -> dict:
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    case = CaseStore().create((case_dir / "denial.txt").read_text(encoding="utf-8"),
                              (case_dir / "chart.txt").read_text(encoding="utf-8"))
    started = time.perf_counter()
    events = [e async for e in run_pipeline(case, llm, settings, index)]
    row = {"case": case_dir.name, "latency_s": round(time.perf_counter() - started, 1)}
    if case.status != "needs_review":
        return {**row, "error": events[-1]["data"]["message"], "tokens": sum(u["total_tokens"] for u in llm.usage)}
    row["extraction"] = extraction_accuracy(case.denial, expected["denial"])
    row.update(criterion_scores(case.criteria, case.matches, expected["criteria"]))
    row["citation_validity"] = citation_validity(case.draft, case.chart_lines, case.criteria, case.denial_text)
    injected, ids = inject_fabrications(case.draft, case.chart_lines)
    review = await asyncio.to_thread(reviewer.run, llm, settings, injected, case.chart_lines, case.criteria)
    flags = merge_flags(review.flags, deterministic_checks(injected, case.chart_lines, case.criteria, case.denial_text))
    row["catch_rate"] = catch_rate(flags, ids)
    row["tokens"] = sum(u["total_tokens"] for u in llm.usage)
    return row


def _fmt(value) -> str:
    if isinstance(value, float) and value <= 1.0:
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def to_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    lines = [header, "|" + "---|" * len(COLUMNS)]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['case']} | error: {r['error']} |" + " |" * (len(COLUMNS) - 2))
            continue
        lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k, _ in COLUMNS) + " |")
    ok = [r for r in rows if "error" not in r]
    if ok:
        means = []
        for k, _ in COLUMNS[1:]:
            mean = sum(r[k] for r in ok) / len(ok)
            means.append(str(round(mean)) if k == "tokens" else _fmt(float(mean)))
        lines.append("| **Mean** | " + " | ".join(means) + " |")
    return "\n".join(lines) + "\n"


async def _main(settings: Settings) -> list[dict]:
    index = PolicyIndex.build(settings, LLMClient(settings) if settings.model_embed else None)
    rows = []
    for case_dir in sorted((settings.data_dir / "cases").glob("case_*")):
        print(f"running {case_dir.name}…", flush=True)
        rows.append(await run_case(case_dir, settings, LLMClient(settings), index))
    return rows


def main() -> None:
    settings = Settings()
    if not settings.nebius_api_key or not settings.model_fast or not settings.model_reason:
        sys.exit("Set NEBIUS_API_KEY, MODEL_FAST and MODEL_REASON in .env (see scripts/discover_models.py).")
    rows = asyncio.run(_main(settings))
    md = (f"# Eval results\n\nRun {datetime.now().strftime('%Y-%m-%d %H:%M')} on {len(rows)} synthetic cases.  \n"
          f"MODEL_FAST=`{settings.model_fast}` · MODEL_REASON=`{settings.model_reason}` · "
          f"MODEL_EMBED=`{settings.model_embed or '(BM25 fallback)'}`\n\n" + to_markdown(rows))
    (ROOT / "eval" / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
