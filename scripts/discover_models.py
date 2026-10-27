"""Step 0: discover which Nemotron models Token Factory serves and what they support.

Usage: uv run python scripts/discover_models.py
Prints every model ID, probes each Nemotron chat model for json_schema / json_object structured
output and tool calling, and recommends MODEL_FAST / MODEL_REASON / MODEL_EMBED values for .env.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.llm import _clean  # noqa: E402

_EMBED = re.compile(r"embed|e5|bge|retriever", re.IGNORECASE)
_PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
_PROBE_MESSAGES = [{"role": "system", "content": "Reply with JSON only."},
                   {"role": "user", "content": 'Return {"ok": true}.'}]
_PROBE_TOOL = {"type": "function", "function": {
    "name": "get_policy", "description": "Look up a coverage policy by ID.",
    "parameters": {"type": "object", "properties": {"policy_id": {"type": "string"}}, "required": ["policy_id"]}}}


def classify(ids: list[str]) -> dict:
    embedding = [i for i in ids if _EMBED.search(i)]
    nemotron = [i for i in ids if "nemotron" in i.lower() and i not in embedding]
    return {"nemotron": nemotron, "embedding": embedding}


def _json_ok(client, model, response_format) -> bool:
    try:
        resp = client.chat.completions.create(model=model, messages=_PROBE_MESSAGES, response_format=response_format,
                                              max_tokens=256, temperature=0.1)
        return json.loads(_clean(resp.choices[0].message.content)).get("ok") is True
    except Exception:
        return False


def probe(client, model: str) -> dict[str, bool]:
    result = {
        "json_schema": _json_ok(client, model, {"type": "json_schema", "json_schema": {
            "name": "Probe", "schema": _PROBE_SCHEMA, "strict": False}}),
        "json_object": _json_ok(client, model, {"type": "json_object"}),
    }
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=256, temperature=0.1, tools=[_PROBE_TOOL], tool_choice="auto",
            messages=[{"role": "user", "content": "Look up coverage policy AH-RAD-012."}])
        result["tools"] = bool(resp.choices[0].message.tool_calls)
    except Exception:
        result["tools"] = False
    return result


def recommend(nemotron: list[str], probes: dict[str, dict], embedding: list[str]) -> dict:
    eligible = [m for m in nemotron if probes.get(m, {}).get("json_schema") or probes.get(m, {}).get("json_object")]

    def first(words: list[str]) -> str:
        return next((m for w in words for m in eligible if w in m.lower()), "")

    fast = first(["nano", "lightning"]) or (eligible[0] if eligible else "")
    reason = first(["super", "ultra"]) or fast
    embed = next((e for e in embedding if "nemotron" in e.lower() or "nv" in e.lower()), embedding[0] if embedding else "")
    return {"MODEL_FAST": fast, "MODEL_REASON": reason, "MODEL_EMBED": embed}


def main() -> None:
    from openai import OpenAI

    settings = Settings()
    if not settings.nebius_api_key:
        sys.exit("Set NEBIUS_API_KEY in .env first.")
    client = OpenAI(base_url=settings.nebius_base_url, api_key=settings.nebius_api_key, timeout=60)
    ids = sorted(m.id for m in client.models.list().data)
    groups = classify(ids)
    print(f"{len(ids)} models at {settings.nebius_base_url}\n")
    for i in ids:
        tag = "  <- nemotron" if i in groups["nemotron"] else "  <- embedding" if i in groups["embedding"] else ""
        print(f"  {i}{tag}")

    print("\nProbing Nemotron chat models (json_schema / json_object / tools):")
    probes = {}
    for m in groups["nemotron"]:
        probes[m] = probe(client, m)
        marks = "  ".join(f"{k}={'PASS' if v else 'fail'}" for k, v in probes[m].items())
        print(f"  {m}: {marks}")

    print("\nRecommended .env values:")
    for k, v in recommend(groups["nemotron"], probes, groups["embedding"]).items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
