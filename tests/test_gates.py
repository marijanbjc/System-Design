"""Гейты вокруг генерации: модель умеет только ужесточить маршрут, но не разрешить."""

from app.config import get_settings
from app.models import AutomationLevel, LLMDraft, Risk, Route, Status
from app.routing.gates import post_gate, pre_gate
from app.storage.vector_index import Hit


def _confident_draft(**overrides) -> LLMDraft:
    """Черновик, который прошёл бы гейт, если бы локальные условия это позволяли."""
    base = {
        "answer_draft": "готовый ответ",
        "has_enough_context": 0.9,
        "is_on_topic": True,
        "is_toxic": False,
        "confidence": 0.95,
    }
    return LLMDraft(**(base | overrides))


def test_llm_schema_has_no_topic_and_no_risk() -> None:
    """Несущее свойство защиты от инъекций, проверенное на самой схеме."""
    fields = set(LLMDraft.model_fields)

    assert "topic" not in fields
    assert "risk" not in fields


def test_confident_draft_is_blocked_on_risky_ticket() -> None:
    """Даже confidence=0.95 не открывает авто-отправку, если риск не низкий."""
    decision = post_gate(AutomationLevel.AUTO_OK, Risk.HIGH, _confident_draft(), False, False)

    assert not decision.auto_send
    assert decision.reason == "risk_not_low"


def test_topic_policy_outweighs_model_confidence() -> None:
    """REVIEW_REQUIRED означает «всегда через человека», что бы модель ни вернула."""
    decision = post_gate(
        AutomationLevel.REVIEW_REQUIRED, Risk.LOW, _confident_draft(), False, False
    )

    assert not decision.auto_send
    assert decision.reason == "policy_review_required"


def test_hot_path_flags_reach_the_gate() -> None:
    """Подозрение на инъекцию, найденное до вызова модели, блокирует авто-отправку."""
    decision = post_gate(AutomationLevel.AUTO_OK, Risk.LOW, _confident_draft(), True, False)

    assert not decision.auto_send
    assert decision.reason == "hot_path_safety_flag"


def test_model_safety_flags_block_auto_send() -> None:
    """Токсичность и уход с темы модель может отметить сама — этого достаточно."""
    toxic = post_gate(
        AutomationLevel.AUTO_OK, Risk.LOW, _confident_draft(is_toxic=True), False, False
    )
    off_topic = post_gate(
        AutomationLevel.AUTO_OK, Risk.LOW, _confident_draft(is_on_topic=False), False, False
    )

    assert not toxic.auto_send
    assert not off_topic.auto_send
    assert toxic.reason == off_topic.reason == "llm_safety_flag"


def test_low_context_score_blocks_auto_send() -> None:
    """Модель сама сообщила, что контекста мало, — этого достаточно для эскалации."""
    settings = get_settings()
    decision = post_gate(
        AutomationLevel.AUTO_OK,
        Risk.LOW,
        _confident_draft(has_enough_context=settings.tau_ctx - 0.01),
        False,
        False,
    )

    assert not decision.auto_send
    assert decision.reason == "low_context_score"


def test_low_confidence_blocks_auto_send() -> None:
    """Порог уверенности — последняя проверка перед отправкой пользователю."""
    settings = get_settings()
    decision = post_gate(
        AutomationLevel.AUTO_OK,
        Risk.LOW,
        _confident_draft(confidence=settings.tau_conf - 0.01),
        False,
        False,
    )

    assert not decision.auto_send
    assert decision.reason == "low_confidence"


def test_gate_passes_only_when_every_condition_holds() -> None:
    """Позитивный случай: политика разрешает, риск низкий, флаги чистые."""
    decision = post_gate(AutomationLevel.AUTO_OK, Risk.LOW, _confident_draft(), False, False)

    assert decision.auto_send


def test_pre_gate_blocks_llm_call_without_context() -> None:
    """Нет пригодного фрагмента — нет и вызова: экономим и деньги, и риск галлюцинации."""
    settings = get_settings()

    assert not pre_gate([]).auto_send
    assert not pre_gate([Hit("kb:1", settings.tau_kb - 0.01, {})]).auto_send
    assert pre_gate([Hit("kb:1", settings.tau_kb + 0.01, {})]).auto_send


def test_worker_skips_llm_when_context_is_missing(client, redis_client) -> None:
    """Тот же пре-гейт, но проверенный сквозным путём через реальный воркер."""
    from app.api.deps import get_container

    deps = get_container()
    body = client.post("/tickets", json={"channel": "chat", "text_raw": "zzz qqq wwww yyyy"}).json()
    ticket = deps.audit.load_ticket(body["ticket_id"])
    processed = deps.worker.process(ticket)

    trail = deps.audit.trail(processed.ticket_id)
    assert not any(e["event"] == "Generated.done" for e in trail)
    assert processed.route == Route.TIER2_REVIEW


def test_llm_failure_degrades_to_operator(client) -> None:
    """При недоступности провайдера система деградирует, а не падает."""
    from app.api.deps import get_container
    from app.llm.mock import FAIL_MARKER

    deps = get_container()
    body = client.post(
        "/tickets",
        json={
            "channel": "chat",
            "text_raw": f"Перенести дату доставки заказа на другой день можно? {FAIL_MARKER}",
        },
    ).json()
    client.post("/admin/drain-queues")

    ticket = deps.audit.load_ticket(body["ticket_id"])
    trail = deps.audit.trail(body["ticket_id"])

    assert ticket.status in (Status.PENDING_OPERATOR, Status.AWAITING_APPROVAL)
    assert any(e["event"] == "Generated.failed" for e in trail)
    assert any(e.get("reason") == "llm_unavailable" for e in trail)


def test_rate_limiter_bounds_the_call_rate(redis_client) -> None:
    """Всплеск заявок не может выдать больше разрешений, чем размер ведра."""
    from app.queues.limiter import RateLimiter

    settings = get_settings()
    redis_client.delete(settings.rate_limit_bucket_key)
    limiter = RateLimiter(redis_client)

    granted = sum(1 for _ in range(settings.llm_rate_limit_burst + 5) if limiter.try_acquire())

    assert granted <= settings.llm_rate_limit_burst
