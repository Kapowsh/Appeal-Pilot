import re

from app.schemas import CriteriaList, Criterion, Denial

SYSTEM = """You read excerpts from payer medical policies and list the coverage criteria the denied service must meet.
- Use only the policy that matches the denied service (prefer the policy ID the denial cites).
- Use only the numbered items under "Coverage Criteria". One criterion per numbered item.
- text: the criterion wording, verbatim. source_doc: the policy ID. source_section: the item number, e.g. "2.1".
- id: same as source_section.
- Never invent or merge criteria."""

_SECTION = re.compile(r"\d+\.\d+")


def run(llm, settings, index, denial: Denial, tavily=None) -> tuple[list[Criterion], list[dict]]:
    query = f"{denial.payer} {denial.service} {denial.denial_reason}"
    chunks = index.search(query, k=4, payer=denial.payer)
    excerpts = "\n\n".join(f"[source_doc={c.source_doc} | {c.section}]\n{c.text}" for c in chunks)
    user = (f"Denied service: {denial.service} (code {denial.cpt_or_hcpcs or 'n/a'})\n"
            f"Payer: {denial.payer}\nDenial reason: {denial.denial_reason}\n\nPolicy excerpts:\n\n{excerpts}")
    parsed = llm.chat_json(settings.model_fast, SYSTEM, user, CriteriaList, step="policy", max_tokens=2048)
    default_doc = chunks[0].source_doc if chunks else ""
    criteria = []
    for i, c in enumerate(parsed.criteria, start=1):
        # Keep the policy's own item number ("2.1") so the eval and UI can key on it.
        section = _SECTION.search(c.source_section)
        criteria.append(Criterion(id=f"C{i}", text=c.text.strip(), source_doc=c.source_doc.strip() or default_doc,
                                  source_section=section.group(0) if section else c.source_section.strip()))
    refs = tavily(f"{denial.service} prior authorization coverage criteria") if tavily else []
    return criteria, refs
