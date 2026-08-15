"""Single source of truth for every threshold and magic number in the system."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every threshold lives here, never inline in logic."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- infrastructure ---
    redis_url: str = "redis://localhost:6379/0"
    audit_db_path: str = "audit.db"

    # --- embeddings ---
    embed_dim: int = 256
    encoder_version: str = "mock-hash-v1"
    classifier_version: str = "centroid-v1"

    # --- routing thresholds (architecture.md 8.1) ---
    tau_high: float = 0.72  # triple similarity -> Tier 1 (typical request)
    tau_kb: float = 0.40  # knowledge-base chunk similarity -> pre-gate before LLM
    # tau_conf / tau_ctx are calibrated against the deterministic mock in this PoC.
    # On a real provider they must be re-picked on a held-out set (ml.md 7).
    tau_conf: float = 0.70  # LLM self-reported confidence -> auto-send
    tau_ctx: float = 0.55  # LLM context-sufficiency score -> auto-send
    conf_cls_min: float = 0.45  # below this the topic is not trusted (abstain to human)
    risk_high_score: float = 0.60  # risk head score -> "high"
    risk_medium_score: float = 0.35  # risk head score -> "medium"

    # --- surge detection (architecture.md 6.1) ---
    surge_window_minutes: int = 10
    surge_key_ttl_seconds: int = 900
    surge_threshold: int = 5  # low on purpose so the demo can trigger it

    # --- LLM rate limiting and budget (architecture.md 9) ---
    llm_provider: str = "mock"  # "mock" | "openai"
    llm_model: str = "mock-deterministic-v1"
    llm_prompt_version: str = "v1"
    llm_rate_limit_rps: float = 2.0
    llm_rate_limit_burst: int = 4
    llm_daily_token_budget: int = 200_000
    llm_timeout_seconds: float = 20.0

    # --- OpenAI-compatible client (only used when llm_provider == "openai") ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- PII vault ---
    pii_vault_ttl_seconds: int = 3600  # must outlive the operator review queue

    # --- queues ---
    stream_gen: str = "stream:gen"
    stream_review: str = "stream:review"
    stream_delivery: str = "stream:delivery"
    queue_depth_alert: int = 50


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
