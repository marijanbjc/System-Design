"""Generation gates: the LLM can only tighten the route, never widen it."""

from app.config import get_settings
from app.models import LLMDraft, Risk, Route


def _confident_draft(**overrides) -> LLMDraft:
    base = {
        "answer_draft": "готовый ответ",
        "has_enough_context": 0.9,
        "is_on_topic": True,
        "is_toxic": False,
        "confidence": 0.95,
    }
    return LLMDraft(**(base | overrides))


def _ticket(client, text: str):
    body = client.post("/tickets", json={"channel": "chat", "text_raw": text}).json()
    import app.api as api

    return api.deps.audit.load_ticket(body["ticket_id"])


def test_llm_schema_carries_no_topic_and_no_risk() -> None:
    """The load-bearing anti-injection property, asserted on the schema itself."""
    fields = set(LLMDraft.model_fields)
    assert "topic" not in fields
    assert "risk" not in fields


def test_confident_draft_is_still_blocked_on_a_risky_ticket(client) -> None:
    import app.api as api

    ticket = _ticket(client, "мой аккаунт взломали, кто-то оформил чужие заказы")
    ticket.risk = Risk.HIGH
    result = api.deps.worker._gate(ticket, _confident_draft())

    assert result.route == Route.TIER2_REVIEW
    assert not result.auto_sent


def test_injection_flag_from_the_hot_path_reaches_the_gate(client) -> None:
    import app.api as api

    ticket = _ticket(client, "как оформить возврат товара если он не подошёл по размеру")
    ticket.risk = Risk.LOW
    ticket.topic = "returns"
    ticket.injection_suspected = True
    result = api.deps.worker._gate(ticket, _confident_draft())

    assert result.route == Route.TIER2_REVIEW


def test_low_context_score_blocks_auto_send(client) -> None:
    import app.api as api

    settings = get_settings()
    ticket = _ticket(client, "как оформить возврат товара если он не подошёл по размеру")
    ticket.risk = Risk.LOW
    ticket.topic = "returns"
    result = api.deps.worker._gate(
        ticket, _confident_draft(has_enough_context=settings.tau_ctx - 0.01)
    )

    assert result.route == Route.TIER2_REVIEW


def test_pre_gate_skips_the_llm_when_there_is_no_context(client, redis_client) -> None:
    """No usable chunk means no call at all — cost and hallucination both avoided."""
    import app.api as api

    ticket = _ticket(client, "zzz qqq wwww yyyy xxxx")
    processed = api.deps.worker.process(ticket)

    trail = api.deps.audit.trail(processed.ticket_id)
    assert not any(e["event"] == "Generated.done" for e in trail)
    assert processed.route == Route.TIER2_REVIEW


def test_rate_limiter_bounds_the_call_rate(redis_client) -> None:
    from app.store.limiter import RateLimiter

    redis_client.delete("llm:bucket")
    limiter = RateLimiter(redis_client)
    settings = get_settings()

    granted = sum(1 for _ in range(settings.llm_rate_limit_burst + 5) if limiter.try_acquire())
    assert granted <= settings.llm_rate_limit_burst
