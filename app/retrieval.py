"""Policy corpus and retrieval index."""
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("policy_id", "payer", "service", "codes")


@dataclass
class PolicyDoc:
    policy_id: str
    payer: str
    service: str
    codes: list[str]
    title: str
    body: str
    filename: str


def parse_policy(text: str, filename: str) -> PolicyDoc:
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError(f"{filename}: missing front matter")
    try:
        header, body = text[4:].split("\n---\n", 1)
    except ValueError:
        raise ValueError(f"{filename}: unterminated front matter") from None
    meta = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    missing = [k for k in REQUIRED_KEYS if not meta.get(k)]
    if missing:
        raise ValueError(f"{filename}: front matter missing {', '.join(missing)}")
    title = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), meta["policy_id"])
    return PolicyDoc(
        policy_id=meta["policy_id"], payer=meta["payer"], service=meta["service"],
        codes=[c.strip() for c in meta["codes"].split(",") if c.strip()],
        title=title, body=body.strip(), filename=filename,
    )


def load_policies(directory: Path) -> list[PolicyDoc]:
    return [parse_policy(p.read_text(encoding="utf-8"), p.name) for p in sorted(Path(directory).glob("*.md"))]
