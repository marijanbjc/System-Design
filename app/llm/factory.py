"""Выбор реализации LLM по конфигу."""

from app.config import get_settings
from app.llm.base import LLMClient
from app.llm.mock import MockLLM
from app.llm.openai_client import OpenAICompatibleLLM


def build_llm() -> LLMClient:
    """Мок по умолчанию, реальный клиент — только при явной настройке."""
    if get_settings().llm_provider == "openai":
        return OpenAICompatibleLLM()
    return MockLLM()
