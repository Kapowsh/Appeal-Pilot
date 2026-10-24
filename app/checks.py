"""Deterministic (no-LLM) checks run after the reviewer agent."""
import re

from app.schemas import AppealDraft, ChartLine, Criterion, ReviewFlag

_BOILERPLATE = re.compile(
    r"^\s*(dear|sincerely|respectfully submitted|thank you|regards|re:|to whom|cc:|enclosures?)\b", re.IGNORECASE)
_STOPWORDS = {"about", "after", "again", "patient", "which", "their", "there", "these", "those", "would",
              "could", "should", "while", "where", "being", "other", "under", "since", "before", "member"}


def normalize_id(s: str) -> str:
    return s.strip().upper()


def is_boilerplate(text: str) -> bool:
    if _BOILERPLATE.match(text):
        return True
    return len(text.split()) < 4 and not re.search(r"\d", text)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w{5,}", text.lower()) if w not in _STOPWORDS}


def is_factual(text: str, chart_text: str) -> bool:
    if is_boilerplate(text):
        return False
    if re.search(r"\d", text):
        return True
    return len(_content_words(text) & _content_words(chart_text)) >= 2


def _numbers(text: str) -> set[str]:
    # "03/14/2025" -> {"3", "14", "2025"}; leading zeros never decide a match.
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", text)}


def deterministic_checks(draft: AppealDraft, chart_lines: list[ChartLine], criteria: list[Criterion],
                         denial_text: str = "") -> list[ReviewFlag]:
    lines = {l.id: l.text for l in chart_lines}
    crits = {c.id: f"{c.text} {c.source_section} {c.source_doc}" for c in criteria}
    chart_text = " ".join(lines.values())
    flags: list[ReviewFlag] = []
    for s in draft.sentences:
        issues: list[str] = []
        action = "revise"
        cites = [normalize_id(c) for c in s.cites]
        for cid in cites:
            if cid not in lines and cid not in crits:
                issues.append(f"cites unknown ID {cid}")
        if not cites and is_factual(s.text, chart_text):
            issues.append("factual claim without citation")
        if cites:
            sources = " ".join(lines.get(c) or crits.get(c) or "" for c in cites)
        else:
            sources = f"{denial_text} {chart_text}"
        missing = sorted(_numbers(s.text) - _numbers(sources), key=lambda n: (len(n), n))
        if missing:
            issues.append(f"number/date not found in cited sources: {', '.join(missing)}")
            action = "remove"
        if issues:
            flags.append(ReviewFlag(sentence_id=s.id, issue="; ".join(issues), action=action))
    return flags


def merge_flags(*lists: list[ReviewFlag]) -> list[ReviewFlag]:
    merged: dict[str, ReviewFlag] = {}
    for flags in lists:
        for f in flags:
            if f.sentence_id not in merged:
                merged[f.sentence_id] = f.model_copy()
                continue
            cur = merged[f.sentence_id]
            if f.issue not in cur.issue.split("; "):
                cur.issue = f"{cur.issue}; {f.issue}"
            if f.action == "remove":
                cur.action = "remove"
    return list(merged.values())
