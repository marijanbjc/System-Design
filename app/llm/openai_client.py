"""OpenAI-compatible client, enabled by LLM_PROVIDER=openai. Never used in the demo."""

import json

import httpx

from app.config import get_settings
from app.llm.base import SYSTEM_PROMPT, LLMDraft, LLMUnavailable, build_prompt


class OpenAICompatibleLLM:
    """Structured output enforced by response_format=json_object plus pydantic validation."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise LLMUnavailable("OPENAI_API_KEY is not set")
        self.model_name = settings.llm_model
        self._base_url = settings.openai_base_url
        self._api_key = settings.openai_api_key
        self._timeout = settings.llm_timeout_seconds

    def generate(self, question_scrubbed: str, context_chunks: list[str]) -> tuple[LLMDraft, int]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question_scrubbed, context_chunks)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - any provider failure degrades the same way
            raise LLMUnavailable(str(exc)) from exc

        content = body["choices"][0]["message"]["content"]
        tokens = int(body.get("usage", {}).get("total_tokens", 0))
        try:
            # Validation is part of the defence: a model that returns anything outside
            # the schema (for example an injected "risk": "low") simply fails here.
            return LLMDraft.model_validate(json.loads(content)), tokens
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"malformed structured output: {exc}") from exc


def build_llm():
    """Factory: mock by default, real client only when explicitly configured."""
    from app.llm.mock import MockLLM

    settings = get_settings()
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM()
    return MockLLM()
