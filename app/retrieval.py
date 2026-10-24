"""Policy corpus and retrieval index."""
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings

log = logging.getLogger(__name__)

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


# --- index -----------------------------------------------------------------

@dataclass
class Chunk:
    policy_id: str
    payer: str
    service: str
    section: str
    text: str
    source_doc: str


def chunk_policy(doc: PolicyDoc) -> list[Chunk]:
    """One chunk per '## ' section, prefixed with the doc title so each chunk stands alone."""
    chunks: list[Chunk] = []
    header = f"{doc.title}\nPayer: {doc.payer}. Service: {doc.service}. Codes: {', '.join(doc.codes)}."
    for part in re.split(r"^## ", doc.body, flags=re.MULTILINE)[1:]:
        heading, _, body = part.partition("\n")
        chunks.append(Chunk(policy_id=doc.policy_id, payer=doc.payer, service=doc.service,
                            section=heading.strip(), text=f"{header}\n## {heading.strip()}\n{body.strip()}",
                            source_doc=doc.policy_id))
    return chunks


def load_chunks(directory: Path) -> list[Chunk]:
    return [c for doc in load_policies(directory) for c in chunk_policy(doc)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class PolicyIndex:
    def __init__(self, chunks: list[Chunk], embedder: Callable[[list[str]], list[list[float]]] | None = None,
                 cache_dir: Path | None = None, cache_key: str = ""):
        self.chunks = chunks
        self.embedder = embedder
        self.mode: Literal["embeddings", "bm25"] = "bm25"
        self._bm25 = BM25Okapi([_tokens(c.text) for c in chunks])
        self._vectors: np.ndarray | None = None
        if embedder is not None:
            try:
                self._vectors = self._load_vectors(cache_dir, cache_key)
                self.mode = "embeddings"
            except Exception as e:  # any embeddings failure -> BM25 keeps the demo working
                log.warning("Embedding index unavailable, using BM25: %s", e)

    def _load_vectors(self, cache_dir: Path | None, cache_key: str) -> np.ndarray:
        digest = hashlib.sha256((cache_key + "".join(c.text for c in self.chunks)).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"embeddings-{digest}.npy" if cache_dir else None
        if path and path.exists():
            return np.load(path)
        vectors = _normalize(np.asarray(self.embedder([c.text for c in self.chunks]), dtype=np.float32))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors

    def search(self, query: str, k: int = 4, payer: str | None = None, policy_id: str | None = None) -> list[Chunk]:
        """Top-k chunks. A payer that matches no policy returns [] rather than another payer's criteria."""
        candidates = list(range(len(self.chunks)))
        if payer:
            candidates = [i for i in candidates if _payer_matches(payer, self.chunks[i].payer)]
        if policy_id:
            cited = [i for i in candidates if self.chunks[i].source_doc == policy_id.upper()]
            candidates = cited or candidates
        if not candidates:
            return []
        if self.mode == "embeddings":
            q = _normalize(np.asarray(self.embedder([query]), dtype=np.float32))[0]
            scores = self._vectors @ q
        else:
            scores = self._bm25.get_scores(_tokens(query))
            candidates = [i for i in candidates if scores[i] > 0]  # no shared terms -> not a match
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]

    @classmethod
    def build(cls, settings: Settings, llm: Any | None) -> "PolicyIndex":
        embedder = None
        if settings.model_embed and llm is not None:
            embedder = lambda texts: llm.embed(settings.model_embed, texts)  # noqa: E731
        return cls(load_chunks(settings.data_dir / "policies"), embedder=embedder,
                   cache_dir=settings.cache_dir, cache_key=settings.model_embed)


def _payer_matches(query: str, payer: str) -> bool:
    # Tolerates suffixes the intake model may add ("Acme Health Plan", "Northwind Care, Inc.").
    q, p = " ".join(_tokens(query)), " ".join(_tokens(payer))
    return bool(q) and (q == p or q.startswith(p + " ") or p.startswith(q + " "))


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
