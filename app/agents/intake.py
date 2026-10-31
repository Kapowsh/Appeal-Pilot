from app.schemas import Denial

SYSTEM = """You extract structured fields from an insurance prior-authorization denial letter.
Copy values as written in the letter. Use null for any field the letter does not contain.
- payer: the insurance company name.
- service: the denied service in plain words (e.g. "MRI lumbar spine").
- cpt_or_hcpcs: the procedure, device, or drug code, if given.
- icd10: every ICD-10 diagnosis code in the letter.
- denial_reason: one sentence summarizing the payer's stated reason.
- denial_date and appeal_deadline: YYYY-MM-DD."""


def run(llm, settings, denial_text: str) -> Denial:
    return llm.chat_json(settings.model_fast, SYSTEM, f"Denial letter:\n\n{denial_text}", Denial,
                         step="intake", max_tokens=6144)
