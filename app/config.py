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

    # --- предобработка ---
    max_text_length: int = 4000  # обрезка обращения перед моделями

    # --- эмбеддинги и версии моделей (пишутся в аудит каждого тикета) ---
    embed_dim: int = 256
    encoder_trigram_weight: float = 0.5  # вес символьной триграммы относительно слова
    encoder_version: str = "mock-hash-v1"
    classifier_version: str = "rules-v1"

    # --- мок-классификатор (в целевой картине — обученные головы, ml.md §4) ---
    classifier_matched_confidence: float = 0.9  # правило сработало
    classifier_fallback_confidence: float = 0.4  # ни одно правило не подошло → general

    # --- поиск ---
    retrieval_top_k: int = 3  # сколько соседей забираем из векторного индекса

    # --- пороги маршрутизации (architecture.md §8.1) ---
    tau_high: float = 0.72  # близость тройки → Tier 1 (типовое обращение)
    tau_kb: float = 0.40  # близость фрагмента БЗ → пре-гейт перед вызовом LLM
    # tau_conf и tau_ctx откалиброваны под детерминированный мок. На реальном
    # провайдере их надо переподобрать на held-out выборке (ml.md §7).
    tau_conf: float = 0.70  # самооценка уверенности модели → авто-отправка
    tau_ctx: float = 0.55  # скор достаточности контекста → авто-отправка
    conf_cls_min: float = 0.45  # ниже — теме не доверяем и уходим к безопасному дефолту

    # --- детекция всплеска (architecture.md §6.1) ---
    surge_window_minutes: int = 10
    surge_key_ttl_seconds: int = 900
    surge_threshold: int = 5  # намеренно низкий, чтобы демо могло его перебить

    # --- ограничение частоты вызовов LLM (architecture.md §9) ---
    llm_model: str = "mock-deterministic-v1"
    llm_prompt_version: str = "v1"
    llm_rate_limit_rps: float = 2.0
    llm_rate_limit_burst: int = 4
    llm_bucket_ttl_seconds: int = 3600
    llm_acquire_poll_seconds: float = 0.05  # как часто воркер перепроверяет ведро
    llm_timeout_seconds: float = 20.0
    llm_cost_per_token: float = 1e-5  # для учёта стоимости в аудите

    # --- параметры мок-модели: не продовые ручки, а «характер» заглушки ---
    mock_chars_per_token: int = 4
    mock_min_context_for_answer: float = 0.5
    mock_confidence_base: float = 0.55
    mock_confidence_scale: float = 0.4
    mock_confidence_cap: float = 0.95
    mock_low_confidence: float = 0.25

    # --- очереди ---
    stream_gen: str = "stream:gen"
    stream_review: str = "stream:review"
    stream_delivery: str = "stream:delivery"


@lru_cache
def get_settings() -> Settings:
    """Закешированный экземпляр настроек."""
    return Settings()
