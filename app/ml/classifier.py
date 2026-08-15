"""Мок-классификатор темы и риска: правила по ключевым словам.

В PoC обучать модель избыточно — нужно показать работоспособность маршрутизации, а не
качество классификации. В целевой архитектуре здесь две головы поверх контрастивно
дообученного энкодера (docs/ml.md §3–4); контракт этого модуля при замене не меняется.

Правила упорядочены: рисковые темы проверяются первыми, поэтому «списали деньги за
заказ» уйдёт в `payment_dispute`, а не в `payment`. Неопознанное обращение получает
тему `general` с низкой уверенностью — ниже `conf_cls_min`, то есть роутер не будет
доверять ни теме, ни политике по ней.
"""

from dataclasses import dataclass

from app.config import get_settings
from app.models import Risk


@dataclass(frozen=True)
class Prediction:
    """Результат классификации: тема, риск и уверенность."""

    topic: str
    risk: Risk
    conf_cls: float


# тема → (ключевые слова, риск)
_RULES: list[tuple[str, tuple[str, ...], Risk]] = [
    (
        "payment_dispute",
        ("списали", "списание", "двойн", "верните деньги", "чарджбэк", "chargeback", "в суд"),
        Risk.HIGH,
    ),
    ("account_security", ("взлом", "мошен", "украли", "несанкционирован", "утечк"), Risk.HIGH),
    ("returns", ("возврат", "вернуть", "возврату", "refund", "обмен", "брак"), Risk.LOW),
    (
        "delivery",
        ("доставк", "курьер", "где заказ", "когда приедет", "трек", "посылк", "пункт выдачи"),
        Risk.LOW,
    ),
    ("payment", ("оплат", "платеж", "платёж", "картой", "чек", "сбп"), Risk.LOW),
    ("account", ("аккаунт", "пароль", "личный кабинет", "код подтверждения", "профил"), Risk.LOW),
    ("loyalty", ("бонус", "промокод", "балл", "лояльност", "скидк"), Risk.LOW),
]


class Classifier:
    """Определяет тему и риск обращения по ключевым словам."""

    def predict(self, text: str) -> Prediction:
        """Вернуть тему, риск и уверенность для нормализованного текста обращения."""
        settings = get_settings()
        lowered = text.lower()
        for topic, keywords, risk in _RULES:
            if any(keyword in lowered for keyword in keywords):
                return Prediction(topic=topic, risk=risk, conf_cls=settings.classifier_matched_confidence)
        return Prediction(topic="general", risk=Risk.LOW, conf_cls=settings.classifier_fallback_confidence)
