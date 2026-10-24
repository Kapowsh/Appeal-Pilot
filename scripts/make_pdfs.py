"""Render the demo cases (case_01..03) to PDF so the upload path can be shown with real files.

Usage: uv run python scripts/make_pdfs.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

CASES_DIR = Path(__file__).resolve().parent.parent / "data" / "cases"
DEMO_CASES = ["case_01", "case_02", "case_03"]
FONT, FONT_SIZE, MARGIN = "Helvetica", 9, 50


def render_text(text: str, out: Path) -> None:
    """One source line -> one printed line (font shrinks to fit), so extracted chart line IDs match the .txt."""
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    usable = width - 2 * MARGIN
    y = height - MARGIN
    for line in text.splitlines():
        if y < MARGIN:
            c.showPage()
            y = height - MARGIN
        size = FONT_SIZE
        text_width = stringWidth(line, FONT, size)
        if text_width > usable:
            size = size * usable / text_width
        c.setFont(FONT, size)
        c.drawString(MARGIN, y, line)
        y -= 14
    c.save()


def render_case(case_dir: Path) -> None:
    for name in ("denial", "chart"):
        render_text((case_dir / f"{name}.txt").read_text(encoding="utf-8"), case_dir / f"{name}.pdf")


def main() -> None:
    for name in DEMO_CASES:
        render_case(CASES_DIR / name)
        print(f"wrote {name}/denial.pdf and {name}/chart.pdf")


if __name__ == "__main__":
    main()
