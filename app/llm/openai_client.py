"""OpenAI-совместимый клиент. Включается переменной LLM_PROVIDER=openai, в демо не используется."""

import json

import httpx

from app.config import get_settings
from app.llm.base import SYSTEM_PROMPT, LLMDraft, LLMUnavailable, build_prompt


class OpenAICompatibleLLM:
    """Структурированный выход задаётся response_format и проверяется pydantic-схемой."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise LLMUnavailable("не задан OPENAI_API_KEY")
        self.model_name = settings.llm_model
        self._base_url = settings.openai_base_url
        self._api_key = settings.openai_api_key
        self._timeout = settings.llm_timeout_seconds

    def generate(self, question_scrubbed: str, context_chunks: list[str]) -> tuple[LLMDraft, int]:
        """Вызвать провайдера обезличенным текстом и вернуть валидный черновик."""
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
        except Exception as exc:  # noqa: BLE001 — любой отказ провайдера деградирует одинаково
            raise LLMUnavailable(str(exc)) from exc

        content = body["choices"][0]["message"]["content"]
        tokens = int(body.get("usage", {}).get("total_tokens", 0))
        try:
            # Валидация — часть защиты: модель, вернувшая что-то вне схемы (например,
            # подсунутый инъекцией "risk": "low"), просто не пройдёт эту строку.
            return LLMDraft.model_validate(json.loads(content)), tokens
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"структурированный выход не по схеме: {exc}") from exc
