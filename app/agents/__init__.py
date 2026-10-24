"""Pipeline agents. Each module exposes SYSTEM and run(...)."""
from app.schemas import ChartLine, Criterion


def format_lines(lines: list[ChartLine]) -> str:
    return "\n".join(f"{l.id}: {l.text}" for l in lines)


def format_criteria(criteria: list[Criterion]) -> str:
    return "\n".join(f"{c.id} [{c.source_doc} §{c.source_section}]: {c.text}" for c in criteria)
