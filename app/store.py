"""In-memory case store with a JSON dump to disk, bundled samples, letter assembly, rate limiting."""
import json
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from app.checks import is_boilerplate
from app.schemas import Case
from app.tools.pdf import split_chart_lines

_SAMPLE_ID = re.compile(r"^case_\d{2}$")


class CaseStore:
    def __init__(self, dump_dir: Path | None = None):
        self._cases: dict[str, Case] = {}
        self.dump_dir = dump_dir

    def create(self, denial_text: str, chart_text: str) -> Case:
        case = Case(id=uuid.uuid4().hex[:12], denial_text=denial_text, chart_lines=split_chart_lines(chart_text))
        self._cases[case.id] = case
        self.save(case)
        return case

    def get(self, case_id: str) -> Case:
        return self._cases[case_id]

    def save(self, case: Case) -> None:
        if self.dump_dir:
            out = Path(self.dump_dir) / "cases"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{case.id}.json").write_text(case.model_dump_json(indent=1), encoding="utf-8")


def list_samples(data_dir: Path) -> list[dict]:
    samples = []
    for d in sorted((Path(data_dir) / "cases").glob("case_*")):
        expected = d / "expected.json"
        if _SAMPLE_ID.match(d.name) and expected.exists():
            samples.append({"id": d.name, "title": json.loads(expected.read_text(encoding="utf-8"))["title"]})
    return samples


def load_sample(data_dir: Path, sample_id: str) -> tuple[str, str]:
    if not _SAMPLE_ID.match(sample_id or ""):
        raise KeyError(sample_id)
    d = Path(data_dir) / "cases" / sample_id
    try:
        return (d / "denial.txt").read_text(encoding="utf-8"), (d / "chart.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        raise KeyError(sample_id) from None


def assemble_letter(case: Case) -> str:
    """Final letter text: removed sentences dropped, boilerplate on its own line, body sentences in paragraphs."""
    blocks: list[str] = []
    paragraph: list[str] = []
    for s in case.draft.sentences if case.draft else []:
        if case.flag_resolutions.get(s.id) == "remove":
            continue
        if is_boilerplate(s.text):
            if paragraph:
                blocks.append(" ".join(paragraph))
                paragraph = []
            blocks.append(s.text)
        else:
            paragraph.append(s.text)
    if paragraph:
        blocks.append(" ".join(paragraph))
    return "\n\n".join(blocks) + "\n"


class RateLimiter:
    def __init__(self, max_per_hour: int, clock=time.time):
        self.max_per_hour = max_per_hour
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def hit(self, key: str) -> bool:
        now = self.clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= 3600:
            hits.popleft()
        if len(hits) >= self.max_per_hour:
            return False
        hits.append(now)
        return True
