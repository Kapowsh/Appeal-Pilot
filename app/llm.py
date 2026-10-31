"""The one place LLM calls happen: Nebius Token Factory (OpenAI-compatible)."""
import json
import logging
import random
import re
import time
from typing import Any, Callable, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 4
TIMEOUT_S = 120
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(Exception):
    pass


def _clean(content: str) -> str:
    text = _THINK.sub("", content or "")
    if "</think>" in text:  # reasoning without an opening tag
        text = text.split("</think>", 1)[1]
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text.strip()


class LLMClient:
    def __init__(self, settings: Settings, client: Any = None, sleep: Callable[[float], None] = time.sleep):
        if client is None and settings.nebius_api_key:
            client = OpenAI(base_url=settings.nebius_base_url, api_key=settings.nebius_api_key,
                            timeout=TIMEOUT_S, max_retries=0)
        self.client = client
        self._sleep = sleep
        self._missing_key = client is None
        self.json_schema_ok = True
        self.usage: list[dict] = []

    def _with_retries(self, fn: Callable[[], Any]) -> Any:
        err: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return fn()
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                err = e
            except openai.APIStatusError as e:
                if e.status_code < 500:
                    raise
                err = e
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(min(8, 0.5 * 2 ** attempt) + random.uniform(0, 0.25))
        raise LLMError(f"Token Factory request failed after {MAX_ATTEMPTS} attempts: {err}")

    def _complete(self, model, messages, schema, max_tokens, temperature):
        def call(response_format):
            return self._with_retries(lambda: self.client.chat.completions.create(
                model=model, messages=messages, response_format=response_format,
                max_tokens=max_tokens, temperature=temperature))

        if self.json_schema_ok:
            try:
                return call({"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}})
            except openai.BadRequestError as e:
                log.warning("json_schema rejected by %s, falling back to json_object: %s", model, e)
                self.json_schema_ok = False
            except openai.APIStatusError as e:
                raise LLMError(f"Token Factory rejected the request ({e.status_code}): {e}") from e
        try:
            return call({"type": "json_object"})
        except openai.APIStatusError as e:
            raise LLMError(f"Token Factory rejected the request ({e.status_code}): {e}") from e

    def _record(self, step, model, started, usage):
        self.usage.append({
            "step": step, "model": model, "latency_ms": int((time.perf_counter() - started) * 1000),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        })

    def chat_json(self, model: str, system: str, user: str, schema: type[T], *, step: str = "",
                  max_tokens: int = 2048, temperature: float = 0.1) -> T:
        self._check_ready(model)
        system_msg = {"role": "system", "content": system + "\n\nReturn only JSON matching this schema:\n"
                      + json.dumps(schema.model_json_schema())}
        user_content = user
        err: Exception | None = None
        for _ in range(2):  # one validation retry
            messages = [system_msg, {"role": "user", "content": user_content}]
            started = time.perf_counter()
            resp = self._complete(model, messages, schema, max_tokens, temperature)
            self._record(step, model, started, resp.usage)
            try:
                return schema.model_validate_json(_clean(resp.choices[0].message.content))
            except ValidationError as e:
                err = e
                user_content = (user + f"\n\nYour previous reply failed validation: {e}. "
                                "Return only valid JSON matching the schema.")
        raise LLMError(f"{step or model}: model output failed validation twice: {err}")

    def _check_ready(self, model: str) -> None:
        if not model:
            raise LLMError("Model ID not configured: set MODEL_FAST / MODEL_REASON in .env "
                           "(run scripts/discover_models.py).")
        if self._missing_key:
            raise LLMError("NEBIUS_API_KEY is not set. Add it to .env to call Nebius Token Factory.")

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self._check_ready(model)
        started = time.perf_counter()
        resp = self._with_retries(lambda: self.client.embeddings.create(model=model, input=texts))
        self._record("embed", model, started, resp.usage)
        return [d.embedding for d in resp.data]
