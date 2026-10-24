import openai
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm import LLMClient, LLMError
from tests.fakes import FakeOpenAI, api_error


class Out(BaseModel):
    x: int


def make(responses):
    fake = FakeOpenAI(responses)
    sleeps = []
    client = LLMClient(Settings(_env_file=None), client=fake, sleep=sleeps.append)
    return client, fake, sleeps


def test_uses_json_schema_and_low_temperature():
    client, fake, _ = make(['{"x":1}'])
    client.chat_json("m", "sys", "user", Out)
    call = fake.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["temperature"] <= 0.2
    assert call["max_tokens"] == 2048


def test_falls_back_to_json_object_on_400():
    client, fake, _ = make([api_error(openai.BadRequestError, 400), '{"x":1}'])
    assert client.chat_json("m", "sys", "user", Out).x == 1
    assert fake.calls[1]["response_format"] == {"type": "json_object"}
    assert client.json_schema_ok is False


def test_validation_retry_includes_error():
    client, fake, _ = make(['{"x":"nope"}', '{"x":2}'])
    assert client.chat_json("m", "sys", "user", Out).x == 2
    assert "failed validation" in fake.calls[1]["messages"][-1]["content"]


def test_validation_fails_twice_raises():
    client, _, _ = make(['{"x":"a"}', "not json"])
    with pytest.raises(LLMError):
        client.chat_json("m", "sys", "user", Out)


def test_retries_429_then_succeeds():
    client, _, sleeps = make([api_error(openai.RateLimitError, 429),
                              api_error(openai.InternalServerError, 503), '{"x":3}'])
    assert client.chat_json("m", "sys", "user", Out).x == 3
    assert len(sleeps) == 2 and all(s <= 8.25 for s in sleeps)


def test_gives_up_after_4_attempts():
    client, fake, _ = make([api_error(openai.InternalServerError, 500) for _ in range(5)])
    with pytest.raises(LLMError):
        client.chat_json("m", "sys", "user", Out)
    assert len(fake.calls) == 4


def test_records_usage():
    client, _, _ = make([{"content": '{"x":1}', "usage": (10, 5)}])
    client.chat_json("m", "sys", "user", Out, step="intake")
    u = client.usage[0]
    assert {k: u[k] for k in ["step", "model", "prompt_tokens", "completion_tokens", "total_tokens"]} == \
        {"step": "intake", "model": "m", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert u["latency_ms"] >= 0


def test_strips_think_and_fences():
    client, _, _ = make(['<think>hmm</think>\n```json\n{"x":4}\n```'])
    assert client.chat_json("m", "sys", "user", Out).x == 4


def test_missing_model_raises_readable():
    client, fake, _ = make(['{"x":1}'])
    with pytest.raises(LLMError, match="MODEL_FAST / MODEL_REASON"):
        client.chat_json("", "sys", "user", Out)
    assert fake.calls == []


def test_missing_api_key_raises_without_network():
    client = LLMClient(Settings(_env_file=None, nebius_api_key=""))
    with pytest.raises(LLMError, match="NEBIUS_API_KEY"):
        client.chat_json("m", "sys", "user", Out)


def test_auth_error_wrapped():
    client, _, _ = make([api_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMError, match="401"):
        client.chat_json("m", "sys", "user", Out)
