from scripts.discover_models import classify, probe, recommend

IDS = ["nvidia/Nemotron-3-Super-120B", "nvidia/Nemotron-Nano-9B-v2", "BAAI/bge-en-icl", "meta/llama-3"]
SUPER, NANO = IDS[0], IDS[1]
OK = {"json_schema": True, "json_object": True, "tools": True}
FAIL = {"json_schema": False, "json_object": False, "tools": False}


def test_classify():
    assert classify(IDS) == {"nemotron": [SUPER, NANO], "embedding": ["BAAI/bge-en-icl"]}


def test_recommend_prefers_nano_and_super():
    rec = recommend([SUPER, NANO], {SUPER: OK, NANO: OK}, ["BAAI/bge-en-icl", "nvidia/llama-nemotron-embed-1b"])
    assert rec == {"MODEL_FAST": NANO, "MODEL_REASON": SUPER, "MODEL_EMBED": "nvidia/llama-nemotron-embed-1b"}


def test_recommend_falls_back_when_super_fails():
    rec = recommend([SUPER, NANO], {SUPER: FAIL, NANO: OK}, [])
    assert rec["MODEL_REASON"] == NANO and rec["MODEL_FAST"] == NANO


def test_recommend_json_object_only_is_eligible():
    rec = recommend([SUPER], {SUPER: {"json_schema": False, "json_object": True, "tools": False}}, [])
    assert rec["MODEL_REASON"] == SUPER


def test_recommend_no_embedding_gives_empty():
    assert recommend([NANO], {NANO: OK}, [])["MODEL_EMBED"] == ""


def test_probe_failure_is_false():
    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("down")
    assert probe(Boom(), "m") == FAIL
