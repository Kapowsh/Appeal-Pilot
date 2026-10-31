from app.agents import format_criteria, format_lines
from app.checks import normalize_id
from app.schemas import AppealDraft, ChartLine, Criterion, Review, ReviewFlag

SYSTEM = """You are an independent fact-checker for a prior-authorization appeal letter. You did not write it.
For each sentence, check its claims against the chart lines and criteria it cites.
Flag a sentence when it:
- states a fact the cited lines do not support,
- states a fact that is absent from the chart,
- misstates a number or date, or
- claims a criterion is met without supporting evidence.
action "remove" when the claim is false or unsupported; "revise" when it is partly supported or overstated.
Do not flag salutations, closings, or requests for review. summary: one sentence on the letter's reliability."""


def run(llm, settings, draft: AppealDraft, chart_lines: list[ChartLine], criteria: list[Criterion]) -> Review:
    # Fresh context: the reviewer sees the letter and the sources only, never the writer's reasoning.
    sentences = "\n".join(f"{s.id} (cites: {', '.join(s.cites) or 'none'}): {s.text}" for s in draft.sentences)
    user = (f"Letter sentences:\n{sentences}\n\nChart lines:\n{format_lines(chart_lines)}\n\n"
            f"Criteria:\n{format_criteria(criteria)}")
    review = llm.chat_json(settings.model_reason, SYSTEM, user, Review, step="review", max_tokens=32768)
    ids = {s.id for s in draft.sentences}
    flags = [ReviewFlag(sentence_id=normalize_id(f.sentence_id), issue=f.issue, action=f.action)
             for f in review.flags if normalize_id(f.sentence_id) in ids]
    return Review(flags=flags, summary=review.summary)
