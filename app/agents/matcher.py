from app.agents import format_criteria, format_lines
from app.checks import normalize_id
from app.schemas import ChartLine, Criterion, CriterionMatch, MatchList

SYSTEM = """You compare payer coverage criteria against numbered chart lines.
For every criterion return one match with status:
- "met": the chart explicitly documents it. Cite every supporting line ID.
- "not_met": the chart contradicts it. Cite the contradicting line IDs.
- "unclear": the chart is silent or ambiguous. Do not guess.
Cite only line IDs that appear in the chart (e.g. "L3"). Never infer facts that are not written.
rationale: one or two sentences explaining the decision."""


def run(llm, settings, criteria: list[Criterion], chart_lines: list[ChartLine]) -> list[CriterionMatch]:
    user = f"Criteria:\n{format_criteria(criteria)}\n\nChart lines:\n{format_lines(chart_lines)}"
    parsed = llm.chat_json(settings.model_reason, SYSTEM, user, MatchList, step="match", max_tokens=3072)
    valid_lines = {l.id for l in chart_lines}
    by_id: dict[str, CriterionMatch] = {}
    for m in parsed.matches:
        cid = normalize_id(m.criterion_id)
        if cid in by_id:
            continue
        evidence = [e for e in dict.fromkeys(normalize_id(x) for x in m.evidence_line_ids) if e in valid_lines]
        status, rationale = m.status, m.rationale
        if status == "met" and not evidence:
            status, rationale = "unclear", rationale + " (evidence citation invalid)"
        by_id[cid] = CriterionMatch(criterion_id=cid, status=status, evidence_line_ids=evidence, rationale=rationale)
    return [by_id.get(c.id) or CriterionMatch(criterion_id=c.id, status="unclear", evidence_line_ids=[],
                                               rationale="No determination returned.") for c in criteria]
