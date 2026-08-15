"""Единственное место, где живут пороги и магические числа.

В логике их быть не должно: любой порог настраивается через .env без правки кода.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- инфраструктура ---
    redis_url: str = "redis://localhost:6379/0"
    audit_db_path: str = "audit.db"

    # --- эмбеддинги и версии моделей (пишутся в аудит каждого тикета) ---
    embed_dim: int = 256
    encoder_version: str = "mock-hash-v1"
    classifier_version: str = "centroid-v1"

    # --- пороги маршрутизации (architecture.md §8.1) ---
    tau_high: float = 0.72  # близость тройки → Tier 1 (типовое обращение)
    tau_kb: float = 0.40  # близость фрагмента БЗ → пре-гейт перед вызовом LLM
    # tau_conf и tau_ctx откалиброваны под детерминированный мок. На реальном
    # провайдере их надо переподобрать на held-out выборке (ml.md §7).
    tau_conf: float = 0.70  # самооценка уверенности модели → авто-отправка
    tau_ctx: float = 0.55  # скор достаточности контекста → авто-отправка
    conf_cls_min: float = 0.45  # ниже — теме не доверяем и уходим к безопасному дефолту
    risk_high_score: float = 0.60  # скор риск-головы → «high»
    risk_medium_score: float = 0.35  # скор риск-головы → «medium»

    # --- детекция всплеска (architecture.md §6.1) ---
    surge_window_minutes: int = 10
    surge_key_ttl_seconds: int = 900
    surge_threshold: int = 5  # намеренно низкий, чтобы демо могло его перебить

    # --- ограничение частоты и бюджет LLM (architecture.md §9) ---
    llm_provider: str = "mock"  # "mock" | "openai"
    llm_model: str = "mock-deterministic-v1"
    llm_prompt_version: str = "v1"
    llm_rate_limit_rps: float = 2.0
    llm_rate_limit_burst: int = 4
    llm_daily_token_budget: int = 200_000
    llm_timeout_seconds: float = 20.0

    # --- OpenAI-совместимый клиент (используется только при llm_provider == "openai") ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- волт ПДН: TTL должен пережить очередь ревью оператором ---
    pii_vault_ttl_seconds: int = 3600

    # --- очереди ---
    stream_gen: str = "stream:gen"
    stream_review: str = "stream:review"
    stream_delivery: str = "stream:delivery"
    queue_depth_alert: int = 50


@lru_cache
def get_settings() -> Settings:
    """Закешированный экземпляр настроек."""
    return Settings()
