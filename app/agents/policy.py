import re
from collections import Counter

from app.schemas import CriteriaList, Criterion, Denial

SYSTEM = """You read excerpts from payer medical policies and list the coverage criteria the denied service must meet.
- Use only the policy that matches the denied service (prefer the policy ID the denial cites).
- Use only the numbered items under "Coverage Criteria". One criterion per numbered item.
- text: the criterion wording, verbatim. source_doc: the policy ID. source_section: the item number, e.g. "2.1".
- id: same as source_section.
- Never invent or merge criteria.
- If no excerpt covers this service for this payer, return an empty list of criteria."""

_SECTION = re.compile(r"\d+\.\d+")
_POLICY_ID = re.compile(r"\b[A-Z]{2,4}-[A-Z]{2,5}-\d{3}\b")


def run(llm, settings, index, denial: Denial, tavily=None, denial_text: str = "") -> tuple[list[Criterion], list[dict]]:
    query = f"{denial.payer} {denial.service} {denial.denial_reason}"
    cited = _POLICY_ID.search(denial_text)
    chunks = index.search(query, k=4, payer=denial.payer, policy_id=cited.group(0) if cited else None)
    refs = tavily(f"{denial.service} prior authorization coverage criteria") if tavily else []
    if not chunks:
        return [], refs
    excerpts = "\n\n".join(f"[source_doc={c.source_doc} | {c.section}]\n{c.text}" for c in chunks)
    user = (f"Denied service: {denial.service} (code {denial.cpt_or_hcpcs or 'n/a'})\n"
            f"Payer: {denial.payer}\nDenial reason: {denial.denial_reason}\n\nPolicy excerpts:\n\n{excerpts}")
    parsed = llm.chat_json(settings.model_fast, SYSTEM, user, CriteriaList, step="policy", max_tokens=12288)
    default_doc = chunks[0].source_doc
    retrieved = {c.source_doc for c in chunks}
    # Only criteria from a retrieved policy, and from a single policy: never argue from an invented or mixed source.
    kept = [c for c in parsed.criteria if (c.source_doc.strip().upper() or default_doc) in retrieved]
    if kept:
        main_doc = Counter(c.source_doc.strip().upper() or default_doc for c in kept).most_common(1)[0][0]
        kept = [c for c in kept if (c.source_doc.strip().upper() or default_doc) == main_doc]
    criteria = []
    for i, c in enumerate(kept, start=1):
        # Keep the policy's own item number ("2.1") so the eval and UI can key on it.
        section = _SECTION.search(c.source_section)
        criteria.append(Criterion(id=f"C{i}", text=c.text.strip(), source_doc=c.source_doc.strip().upper() or default_doc,
                                  source_section=section.group(0) if section else c.source_section.strip()))
    return criteria, refs
