from app.agents import format_lines
from app.checks import normalize_id
from app.schemas import AppealDraft, ChartLine, Criterion, CriterionMatch, Denial, LetterSentence

SYSTEM = """You draft a formal prior-authorization appeal letter from a treating clinician to a payer.
Return the letter as an ordered list of sentences. Each sentence has text and cites.
Rules:
- cites lists the chart line IDs (L#) and criterion IDs (C#) that support the sentence. Cite "DENIAL" for facts taken from the denial letter (denial date, member ID, policy ID).
- Every sentence that states a clinical fact, date, number, or criterion must cite its support.
- Put IDs only in cites. Never write IDs such as L4 or C2 inside the sentence text.
- Salutation, subject line, closing, and signature lines have empty cites.
- Argue only from criteria whose status is "met", using their evidence lines.
- For criteria that are "not_met" or "unclear", never claim they are met. Instead request a peer-to-peer review with the medical director to discuss them.
- Use only facts written in the chart lines and the denial. Do not invent dates, numbers, names, or findings.
- Sign as "[Treating clinician name]"."""


def run(llm, settings, denial: Denial, criteria: list[Criterion], matches: list[CriterionMatch],
        chart_lines: list[ChartLine]) -> AppealDraft:
    status = {m.criterion_id: m for m in matches}
    crit_text = "\n".join(
        f"{c.id} [{c.source_doc} §{c.source_section}] status={status[c.id].status} "
        f"evidence={','.join(status[c.id].evidence_line_ids) or 'none'}: {c.text}"
        for c in criteria if c.id in status)
    user = (f"Denial:\n{denial.model_dump_json(indent=1)}\n\nCriteria and match results:\n{crit_text}\n\n"
            f"Chart lines:\n{format_lines(chart_lines)}")
    draft = llm.chat_json(settings.model_reason, SYSTEM, user, AppealDraft, step="draft", max_tokens=16384)
    return AppealDraft(sentences=[
        LetterSentence(id=f"S{i}", text=s.text.strip(),
                       cites=list(dict.fromkeys(normalize_id(c) for c in s.cites if c.strip())))
        for i, s in enumerate((s for s in draft.sentences if s.text.strip()), start=1)])
