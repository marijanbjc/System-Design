"""Гейты вокруг генерации: модель умеет только ужесточить маршрут, но не разрешить."""

from app.config import get_settings
from app.models import AutomationLevel, LLMDraft, Risk, Route
from app.routing.gates import post_gate, pre_gate
from app.storage.vector_index import Hit


def _уверенный_черновик(**overrides) -> LLMDraft:
    """Черновик, который прошёл бы гейт, если бы локальные условия это позволяли."""
    base = {
        "answer_draft": "готовый ответ",
        "has_enough_context": 0.9,
        "is_on_topic": True,
        "is_toxic": False,
        "confidence": 0.95,
    }
    return LLMDraft(**(base | overrides))


def test_в_схеме_llm_нет_полей_темы_и_риска() -> None:
    """Несущее свойство защиты от инъекций, проверенное на самой схеме."""
    fields = set(LLMDraft.model_fields)

    assert "topic" not in fields
    assert "risk" not in fields


def test_уверенный_черновик_блокируется_на_рисковом_тикете() -> None:
    """Даже confidence=0.95 не открывает авто-отправку, если риск не низкий."""
    decision = post_gate(
        AutomationLevel.AUTO_OK, Risk.HIGH, _уверенный_черновик(), False, False
    )

    assert not decision.auto_send
    assert decision.reason == "risk_not_low"


def test_политика_темы_важнее_уверенности_модели() -> None:
    """REVIEW_REQUIRED означает «всегда через человека», что бы модель ни вернула."""
    decision = post_gate(
        AutomationLevel.REVIEW_REQUIRED, Risk.LOW, _уверенный_черновик(), False, False
    )

    assert not decision.auto_send
    assert decision.reason == "policy_review_required"


def test_флаги_горячего_пути_доезжают_до_гейта() -> None:
    """Подозрение на инъекцию, найденное до вызова модели, блокирует авто-отправку."""
    decision = post_gate(
        AutomationLevel.AUTO_OK, Risk.LOW, _уверенный_черновик(), True, False
    )

    assert not decision.auto_send
    assert decision.reason == "hot_path_safety_flag"


def test_низкий_скор_контекста_блокирует_автоотправку() -> None:
    """Модель сама сообщила, что контекста мало, — этого достаточно для эскалации."""
    settings = get_settings()
    decision = post_gate(
        AutomationLevel.AUTO_OK,
        Risk.LOW,
        _уверенный_черновик(has_enough_context=settings.tau_ctx - 0.01),
        False,
        False,
    )

    assert not decision.auto_send
    assert decision.reason == "low_context_score"


def test_гейт_пропускает_только_при_всех_выполненных_условиях() -> None:
    """Позитивный случай: политика разрешает, риск низкий, флаги чистые."""
    decision = post_gate(AutomationLevel.AUTO_OK, Risk.LOW, _уверенный_черновик(), False, False)

    assert decision.auto_send


def test_пре_гейт_не_пускает_в_llm_без_контекста() -> None:
    """Нет пригодного фрагмента — нет и вызова: экономим и деньги, и риск галлюцинации."""
    settings = get_settings()

    assert not pre_gate([]).auto_send
    assert not pre_gate([Hit("kb:1", settings.tau_kb - 0.01, {})]).auto_send
    assert pre_gate([Hit("kb:1", settings.tau_kb + 0.01, {})]).auto_send


def test_воркер_не_зовёт_llm_когда_контекста_нет(client, redis_client) -> None:
    """Тот же пре-гейт, но проверенный сквозным путём через реальный воркер."""
    from app.api.deps import get_container

    deps = get_container()
    body = client.post("/tickets", json={"channel": "chat", "text_raw": "zzz qqq wwww yyyy"}).json()
    ticket = deps.audit.load_ticket(body["ticket_id"])
    processed = deps.worker.process(ticket)

    trail = deps.audit.trail(processed.ticket_id)
    assert not any(e["event"] == "Generated.done" for e in trail)
    assert processed.route == Route.TIER2_REVIEW


def test_ограничитель_частоты_держит_потолок(redis_client) -> None:
    """Всплеск заявок не может выдать больше разрешений, чем размер ведра."""
    from app.queues.limiter import RateLimiter

    redis_client.delete("llm:bucket")
    limiter = RateLimiter(redis_client)
    settings = get_settings()

    granted = sum(1 for _ in range(settings.llm_rate_limit_burst + 5) if limiter.try_acquire())
    assert granted <= settings.llm_rate_limit_burst
