"""LLM providers that return schema-shaped JSON, plus a fallback chain.

Every agent turn is one call to `json(system, prompt, schema) -> dict`.
Providers:
  GeminiLLM  - Google Gemini API (needs GEMINI_API_KEY; free tier available)
  OllamaLLM  - local open model via Ollama (free, offline; needs Ollama running)
  FallbackLLM - tries providers in order; retries temporary errors (busy,
                rate limit, network) before switching to the next provider
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_LITE_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"


class LLMError(Exception):
    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient  # worth retrying the same provider (busy, rate limit, network)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class LLM:
    name = "llm"

    def __init__(self):
        self.usage = Usage()

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        raise NotImplementedError

    def _record(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.usage.calls += 1
        self.usage.input_tokens += input_tokens or 0
        self.usage.output_tokens += output_tokens or 0

    @staticmethod
    def _parse(text: str | None) -> dict:
        if not text:
            raise LLMError("The model returned no text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"The model returned invalid JSON: {text[:200]}") from e


class GeminiLLM(LLM):
    """Google Gemini via the google-genai SDK, with JSON-schema constrained output."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, timeout_s: int = 60, client=None):
        super().__init__()
        from google import genai  # imported lazily: optional dependency
        from google.genai import types

        self.model = model
        self.name = f"gemini:{model}"
        # Without a timeout a busy model can hang the call for minutes.
        self.client = client or genai.Client(http_options=types.HttpOptions(timeout=timeout_s * 1000))

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        from google.genai import errors, types

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
        except errors.APIError as e:
            busy = e.code == 429 or (e.code or 0) >= 500
            raise LLMError(f"Gemini API error {e.code}: {e.message}", transient=busy) from e
        except OSError as e:
            raise LLMError("Could not reach the Gemini API (network error)", transient=True) from e

        meta = resp.usage_metadata
        self._record(meta and meta.prompt_token_count, meta and meta.candidates_token_count)
        return self._parse(resp.text)


class OllamaLLM(LLM):
    """Local open model served by Ollama (http://localhost:11434 by default)."""

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, think: bool | None = None, client=None):
        super().__init__()
        import ollama  # imported lazily: optional dependency

        self.model = model
        self.name = f"ollama:{model}"
        self.think = think
        self.client = client or ollama.Client(host=os.environ.get("OLLAMA_HOST") or None)

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        import httpx
        import ollama

        kwargs = {"think": self.think} if self.think is not None else {}
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt + "\n\nRespond only with JSON matching the schema."}],
                format=schema,
                **kwargs,
            )
        except ollama.ResponseError as e:
            hint = ""
            if e.status_code == 404:
                hint = f" - run 'ollama pull {self.model}'"
            elif "allocate" in str(e.error):
                hint = " - not enough free memory: close other apps or use a smaller model (--ollama-model)"
            raise LLMError(f"Ollama error {e.status_code}: {e.error.splitlines()[0]}{hint}") from e
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMError("Could not reach Ollama - is it installed and running?") from e

        self._record(resp.prompt_eval_count, resp.eval_count)
        return self._parse(resp.message.content)


class FallbackLLM(LLM):
    """Try providers in order; when one fails, switch to the next and stay there.

    Temporary errors (busy server, rate limit, network) are retried on the
    same provider after a short wait before giving up on it.
    """

    def __init__(self, providers: list[LLM], on_switch: Callable[[str, str, str | None], None] | None = None,
                 retry_delays: tuple[float, ...] = (5, 15), sleep: Callable[[float], None] = time.sleep,
                 on_retry: Callable[[str, str, float], None] | None = None):
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self.providers = providers
        self.active = 0
        self.failures: list[tuple[str, str]] = []
        self.on_switch = on_switch
        self.retry_delays = retry_delays
        self.sleep = sleep
        self.on_retry = on_retry

    @property
    def name(self) -> str:
        return self.providers[min(self.active, len(self.providers) - 1)].name

    @property
    def usage(self) -> Usage:
        total = Usage()
        for p in self.providers:
            total.calls += p.usage.calls
            total.input_tokens += p.usage.input_tokens
            total.output_tokens += p.usage.output_tokens
        return total

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        while self.active < len(self.providers):
            provider = self.providers[self.active]
            try:
                return self._with_retries(provider, system, prompt, schema)
            except LLMError as e:
                self.failures.append((provider.name, str(e)))
                self.active += 1
                nxt = self.providers[self.active].name if self.active < len(self.providers) else None
                if self.on_switch:
                    self.on_switch(provider.name, str(e), nxt)
        summary = "; ".join(f"{n}: {msg}" for n, msg in self.failures)
        raise LLMError(f"All LLM providers failed ({summary})")

    def _with_retries(self, provider: LLM, system: str, prompt: str, schema: dict) -> dict:
        for delay in (*self.retry_delays, None):
            try:
                return provider.json(system, prompt, schema)
            except LLMError as e:
                if not e.transient or delay is None:
                    raise
                if self.on_retry:
                    self.on_retry(provider.name, str(e), delay)
                self.sleep(delay)
        raise AssertionError("unreachable")
