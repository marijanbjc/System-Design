"""Гейты вокруг генерации: что пропускаем пользователю, а что отдаём человеку.

Ключевое свойство — порядок проверок. Локальные условия (политика темы, риск, флаги
горячего пути) проверяются до флагов модели, а сами флаги модели умеют только
ужесточить маршрут. Инъекция, добившаяся от модели «всё хорошо», авто-отправку
не разблокирует.
"""

from dataclasses import dataclass

from app.config import get_settings
from app.models import AutomationLevel, LLMDraft, Risk
from app.storage.vector_index import Hit


@dataclass(frozen=True)
class GateDecision:
    """Решение гейта: пропускать ли ответ пользователю и почему нет."""

    auto_send: bool
    reason: str


def pre_gate(chunks: list[Hit]) -> GateDecision:
    """Проверка до вызова модели: есть ли вообще из чего собирать ответ.

    Если лучший фрагмент базы знаний ниже порога, генерировать не из чего — не тратим
    вызов LLM и сразу отдаём обращение оператору.
    """
    settings = get_settings()
    if not chunks or chunks[0].sim < settings.tau_kb:
        return GateDecision(auto_send=False, reason="pre_gate_no_context")
    return GateDecision(auto_send=True, reason="ok")


def post_gate(
    level: AutomationLevel,
    risk: Risk,
    draft: LLMDraft,
    injection_suspected: bool,
    unsafe_prefilter: bool,
) -> GateDecision:
    """Проверка после вызова модели по инварианту architecture.md §2.3."""
    settings = get_settings()

    if injection_suspected or unsafe_prefilter:
        return GateDecision(auto_send=False, reason="hot_path_safety_flag")
    if level is not AutomationLevel.AUTO_OK:
        return GateDecision(auto_send=False, reason="policy_review_required")
    if risk is not Risk.LOW:
        return GateDecision(auto_send=False, reason="risk_not_low")
    if draft.is_toxic or not draft.is_on_topic:
        return GateDecision(auto_send=False, reason="llm_safety_flag")
    if draft.has_enough_context < settings.tau_ctx:
        return GateDecision(auto_send=False, reason="low_context_score")
    if draft.confidence < settings.tau_conf:
        return GateDecision(auto_send=False, reason="low_confidence")

    return GateDecision(auto_send=True, reason="ok")
