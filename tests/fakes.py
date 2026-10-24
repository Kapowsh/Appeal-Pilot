"""Test doubles. App code never imports this module."""
from types import SimpleNamespace

import httpx
import openai
from pydantic import BaseModel


def api_error(cls, status: int):
    req = httpx.Request("POST", "http://fake/v1/chat/completions")
    return cls(f"HTTP {status}", response=httpx.Response(status, request=req), body=None)


class _Endpoint:
    def __init__(self, owner, kind):
        self.owner, self.kind = owner, kind

    def create(self, **kw):
        self.owner.calls.append(kw)
        item = self.owner.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if self.kind == "embeddings":
            return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in item],
                                   usage=SimpleNamespace(prompt_tokens=1, total_tokens=1))
        if isinstance(item, str):
            item = {"content": item}
        p, c = item.get("usage", (1, 1))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item["content"]))],
            usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c),
        )


class FakeOpenAI:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_Endpoint(self, "chat"))
        self.embeddings = _Endpoint(self, "embeddings")


class ScriptedLLM:
    """Mimics LLMClient: returns scripted objects per schema, records each call."""

    def __init__(self, by_schema: dict):
        self.by_schema = {k: (list(v) if isinstance(v, list) else [v]) for k, v in by_schema.items()}
        self.calls: list[dict] = []
        self.usage: list[dict] = []

    def chat_json(self, model, system, user, schema, *, step="", max_tokens=2048, temperature=0.1):
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema, "step": step})
        self.usage.append({"step": step, "model": model, "latency_ms": 1, "prompt_tokens": 1,
                           "completion_tokens": 1, "total_tokens": 2})
        queue = self.by_schema[schema]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item.model_copy(deep=True) if isinstance(item, BaseModel) else item

    def embed(self, model, texts):
        return [[float(len(t))] for t in texts]


__all__ = ["FakeOpenAI", "ScriptedLLM", "api_error", "openai"]
