"""Thin wrapper around the Claude API returning schema-validated JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"


class LLMError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ClaudeLLM:
    """One JSON-producing call per agent turn.

    Uses structured outputs (output_config.format) so every reply parses, and
    server-side refusal fallbacks so a declined request is retried on another
    model inside the same call.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low",
                 max_tokens: int = 16000, client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()
        self.usage = Usage()

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude API authentication failed - set ANTHROPIC_API_KEY") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude API rate limit reached; try again later") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Claude API (network error)") from e

        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens

        if resp.stop_reason == "refusal":
            raise LLMError("The model declined to answer")
        if resp.stop_reason == "max_tokens":
            raise LLMError("The model ran out of output tokens")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError("The model returned no text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"The model returned invalid JSON: {text[:200]}") from e
