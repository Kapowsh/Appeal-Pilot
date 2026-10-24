from app.checks import deterministic_checks, merge_flags
from app.schemas import AppealDraft, ChartLine, Criterion, LetterSentence, ReviewFlag

LINES = [ChartLine(id="L1", text="Low back pain radiating to left leg for 8 weeks."),
         ChartLine(id="L2", text="Completed 6 weeks of physical therapy ending 09/02/2026.")]
CRITERIA = [Criterion(id="C1", text="At least 6 weeks of conservative therapy", source_doc="AH-RAD-012", source_section="2.1")]


def check(text, cites):
    return deterministic_checks(AppealDraft(sentences=[LetterSentence(id="S1", text=text, cites=cites)]), LINES, CRITERIA)


def test_unknown_id_flagged():
    flags = check("Pain radiated to the leg.", ["L9"])
    assert len(flags) == 1 and "unknown ID L9" in flags[0].issue


def test_ids_normalized():
    assert check("The patient completed 6 weeks of physical therapy.", [" l2", "c1 "]) == []


def test_uncited_factual_flagged():
    flags = check("The patient reported radiating pain for 8 weeks.", [])
    assert flags and "without citation" in flags[0].issue


def test_boilerplate_not_flagged():
    assert check("Dear Appeals Committee,", []) == []
    assert check("Sincerely,", []) == []


def test_number_not_in_cited_line_flagged():
    flags = check("Therapy ended 03/14/2025.", ["L2"])
    assert flags and flags[0].action == "remove"


def test_date_leading_zero_ok():
    assert check("Therapy ended 9/2/2026.", ["L2"]) == []


def test_merge_dedupes_remove_wins():
    merged = merge_flags([ReviewFlag(sentence_id="S1", issue="a", action="revise")],
                         [ReviewFlag(sentence_id="S1", issue="b", action="remove"),
                          ReviewFlag(sentence_id="S2", issue="c", action="revise")])
    assert [(f.sentence_id, f.issue, f.action) for f in merged] == [("S1", "a; b", "remove"), ("S2", "c", "revise")]
