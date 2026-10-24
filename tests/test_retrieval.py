import re
import zlib
from pathlib import Path

from app.config import Settings
from app.retrieval import PolicyIndex, load_chunks

QUERY = "Acme Health MRI lumbar spine radiculopathy conservative therapy"


class CountingEmbedder:
    def __init__(self):
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"\w+", t.lower()):
                v[zlib.crc32(w.encode()) % 64] += 1.0
            out.append(v)
        return out


def test_bm25_when_no_embed_model():
    index = PolicyIndex.build(Settings(_env_file=None, model_embed=""), llm=None)
    assert index.mode == "bm25"
    results = index.search(QUERY, k=4, payer="Acme Health")
    assert len(results) == 4
    assert results[0].source_doc == "AH-RAD-012"


def test_payer_filter():
    index = PolicyIndex.build(Settings(_env_file=None, model_embed=""), llm=None)
    assert {c.payer for c in index.search(QUERY, k=4, payer="northwind care")} == {"Northwind Care"}


def test_embedding_path_uses_cache(tmp_path):
    chunks = load_chunks(Path("data/policies"))
    first = CountingEmbedder()
    index = PolicyIndex(chunks, embedder=first, cache_dir=tmp_path, cache_key="m")
    assert index.mode == "embeddings" and first.texts == len(chunks)
    second = CountingEmbedder()
    index2 = PolicyIndex(chunks, embedder=second, cache_dir=tmp_path, cache_key="m")
    assert second.texts == 0
    assert index2.search(QUERY, k=4, payer="Acme Health")[0].source_doc == "AH-RAD-012"
    assert second.texts == 1


def test_embed_failure_falls_back():
    def boom(texts):
        raise RuntimeError("no embeddings")
    index = PolicyIndex(load_chunks(Path("data/policies")), embedder=boom)
    assert index.mode == "bm25"
    assert index.search(QUERY, k=4)
