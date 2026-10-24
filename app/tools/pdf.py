import io
import re

from pypdf import PdfReader

from app.schemas import ChartLine

LONG_LINE = 200
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_text(data: bytes, filename: str) -> str:
    """PDF or text bytes -> plain text."""
    if filename.lower().endswith(".pdf") or data.startswith(b"%PDF"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = data.decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"No extractable text in {filename}")
    return text


def split_chart_lines(text: str) -> list[ChartLine]:
    """Number chart text as L1..Ln. Deterministic: expected.json and the eval depend on it."""
    pieces: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) > LONG_LINE:
            pieces.extend(p.strip() for p in _SENTENCE_BREAK.split(line) if p.strip())
        else:
            pieces.append(line)
    return [ChartLine(id=f"L{i}", text=t) for i, t in enumerate(pieces, start=1)]
