import json
import os
from types import SimpleNamespace

import ollama
import pytest
from google.genai import errors

from scheduler.env import load_dotenv
from scheduler.llm import LLM, FallbackLLM, GeminiLLM, LLMError, OllamaLLM

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
          "additionalProperties": False}


# --- fake SDK clients -------------------------------------------------------------

def gemini_client(result=None, error=None):
    calls = []

    def generate_content(**kw):
        calls.append(kw)
        if error:
            raise error
        return SimpleNamespace(text=json.dumps(result),
                               usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5))

    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)), calls


def ollama_client(result=None, error=None):
    calls = []

    def chat(**kw):
        calls.append(kw)
        if error:
            raise error
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)),
                               prompt_eval_count=10, eval_count=5)

    return SimpleNamespace(chat=chat), calls


# --- Gemini -------------------------------------------------------------------------

def test_gemini_request_uses_json_schema():
    client, calls = gemini_client({"a": "hi"})
    llm = GeminiLLM(model="gemini-test", client=client)
    assert llm.json("sys", "prompt", SCHEMA) == {"a": "hi"}
    cfg = calls[0]["config"]
    assert calls[0]["model"] == "gemini-test"
    assert cfg.system_instruction == "sys"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == SCHEMA
    assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) == (1, 10, 5)


def test_gemini_api_error_becomes_llm_error():
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    client, _ = gemini_client(error=err)
    with pytest.raises(LLMError, match="429"):
        GeminiLLM(client=client).json("s", "p", SCHEMA)


# --- Ollama -------------------------------------------------------------------------

def test_ollama_request_uses_format_schema():
    client, calls = ollama_client({"a": "hi"})
    llm = OllamaLLM(model="qwen3:8b", client=client)
    assert llm.json("sys", "prompt", SCHEMA) == {"a": "hi"}
    call = calls[0]
    assert call["format"] == SCHEMA
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert "think" not in call
    assert llm.usage.calls == 1


def test_ollama_missing_model_hint():
    client, _ = ollama_client(error=ollama.ResponseError("model 'x' not found", 404))
    with pytest.raises(LLMError, match="ollama pull qwen3:4b"):
        OllamaLLM(model="qwen3:4b", client=client).json("s", "p", SCHEMA)


def test_ollama_not_running():
    client, _ = ollama_client(error=ConnectionError("refused"))
    with pytest.raises(LLMError, match="installed and running"):
        OllamaLLM(client=client).json("s", "p", SCHEMA)


def test_invalid_json_is_an_llm_error():
    client = SimpleNamespace(chat=lambda **kw: SimpleNamespace(
        message=SimpleNamespace(content="not json"), prompt_eval_count=1, eval_count=1))
    with pytest.raises(LLMError, match="invalid JSON"):
        OllamaLLM(client=client).json("s", "p", SCHEMA)


# --- fallback chain -----------------------------------------------------------------

class Stub(LLM):
    def __init__(self, name, fail=False):
        super().__init__()
        self.name, self.fail = name, fail

    def json(self, system, prompt, schema):
        if self.fail:
            raise LLMError(f"{self.name} is down")
        self._record(1, 1)
        return {"from": self.name}


def test_fallback_switches_and_stays_on_working_provider():
    switches = []
    chain = FallbackLLM([Stub("gemini", fail=True), Stub("gemini-lite"), Stub("ollama")],
                        on_switch=lambda *a: switches.append(a))
    assert chain.json("s", "p", SCHEMA) == {"from": "gemini-lite"}
    assert chain.json("s", "p", SCHEMA) == {"from": "gemini-lite"}
    assert switches == [("gemini", "gemini is down", "gemini-lite")]
    assert chain.name == "gemini-lite"
    assert chain.usage.calls == 2


class Flaky(Stub):
    """Fails with a temporary error a given number of times, then works."""

    def __init__(self, name, failures):
        super().__init__(name)
        self.failures = failures

    def json(self, system, prompt, schema):
        if self.failures:
            self.failures -= 1
            raise LLMError("503 busy", transient=True)
        return super().json(system, prompt, schema)


def test_transient_errors_are_retried_before_switching():
    waits = []
    chain = FallbackLLM([Flaky("gemini", failures=2), Stub("ollama")], retry_delays=(5, 15), sleep=waits.append)
    assert chain.json("s", "p", SCHEMA) == {"from": "gemini"}
    assert waits == [5, 15]
    assert chain.failures == []


def test_persistent_transient_error_eventually_switches():
    waits = []
    chain = FallbackLLM([Flaky("gemini", failures=9), Stub("ollama")], retry_delays=(1, 2), sleep=waits.append)
    assert chain.json("s", "p", SCHEMA) == {"from": "ollama"}
    assert waits == [1, 2]


def test_permanent_errors_switch_without_waiting():
    waits = []
    chain = FallbackLLM([Stub("gemini", fail=True), Stub("ollama")], sleep=waits.append)
    assert chain.json("s", "p", SCHEMA) == {"from": "ollama"}
    assert waits == []


def test_ollama_out_of_memory_hint():
    err = ollama.ResponseError("llama-server process has terminated\nunable to allocate CPU_REPACK buffer", 500)
    client, _ = ollama_client(error=err)
    with pytest.raises(LLMError, match="not enough free memory"):
        OllamaLLM(client=client).json("s", "p", SCHEMA)


def test_fallback_reports_all_failures():
    chain = FallbackLLM([Stub("a", fail=True), Stub("b", fail=True)])
    with pytest.raises(LLMError, match="All LLM providers failed.*a is down.*b is down"):
        chain.json("s", "p", SCHEMA)


# --- .env loading ---------------------------------------------------------------------

def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_KEY=abc\nEMPTY_KEY=\nKEEP_KEY=from_file\nQUOTED='q'\n", encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("EMPTY_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.setenv("KEEP_KEY", "from_env")
    loaded = load_dotenv(env)
    assert os.environ["FOO_KEY"] == "abc"
    assert os.environ["QUOTED"] == "q"
    assert os.environ["KEEP_KEY"] == "from_env"   # existing values win
    assert "EMPTY_KEY" not in os.environ          # blank template lines are ignored
    assert sorted(loaded) == ["FOO_KEY", "QUOTED"]
