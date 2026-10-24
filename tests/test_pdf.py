import pytest

from app.tools.pdf import extract_text, split_chart_lines


def test_split_drops_blanks_and_numbers_sequentially():
    lines = split_chart_lines("  A one.\n\n\nB two.\n C three. ")
    assert [(l.id, l.text) for l in lines] == [("L1", "A one."), ("L2", "B two."), ("L3", "C three.")]


def test_short_line_with_abbreviation_not_split():
    assert len(split_chart_lines("Seen by Dr. Lee today. BP 120/80.")) == 1


def test_long_line_sentence_split():
    long = "Patient reports low back pain radiating to the left leg. " * 3 + "Exam shows positive straight leg raise on the left side today."
    assert len(split_chart_lines(long)) == 4


def test_extract_txt():
    assert extract_text("héllo\n".encode(), "chart.txt").strip() == "héllo"


def test_extract_pdf(tmp_path):
    from reportlab.pdfgen import canvas
    p = tmp_path / "x.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(72, 720, "Synthetic chart line")
    c.save()
    assert "Synthetic chart line" in extract_text(p.read_bytes(), "x.pdf")


def test_blank_pdf_raises(tmp_path):
    from reportlab.pdfgen import canvas
    p = tmp_path / "b.pdf"
    c = canvas.Canvas(str(p))
    c.showPage()
    c.save()
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text(p.read_bytes(), "b.pdf")
