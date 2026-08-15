"""Обратимое обезличивание персональных данных — единственный текст, который выходит наружу.

Замена именно обратимая. Вопрос «где мой заказ №77-881234», отвеченный без номера
заказа, бесполезен, поэтому плейсхолдеры восстанавливаются обратно после ответа
модели — по карте соответствий из локального волта.
"""

import re

# Порядок важен: более специфичные шаблоны идут раньше, чтобы номер карты не был
# съеден общим правилом на последовательность цифр.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE", re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b")),
    ("ORDER", re.compile(r"\b(?:заказ[а-я]*|order)\s*[№#nN]?\s*([\w-]{4,})", re.IGNORECASE)),
    ("NAME", re.compile(r"\b(?:меня зовут|моё имя|мое имя|я)\s+([А-ЯЁ][а-яё]{2,})\b")),
    ("ADDRESS", re.compile(r"\b(?:ул\.|улица|пр-т|проспект)\s+[А-ЯЁа-яё\w\s.,-]{3,40}")),
]


class PiiMap(dict[str, str]):
    """Карта «плейсхолдер → исходное значение». Живёт в волте и наш контур не покидает."""


def scrub(text: str) -> tuple[str, PiiMap]:
    """Заменить персональные данные на устойчивые плейсхолдеры и вернуть карту замен.

    В целевой системе поверх регулярок работает NER (Natasha); в PoC всё держится на
    регулярках, и это названо упрощением в документации.
    """
    mapping = PiiMap()
    counters: dict[str, int] = {}
    scrubbed = text

    for label, pattern in _PATTERNS:

        def _replace(match: re.Match[str], label: str = label) -> str:
            # Группа 1 — когда шаблону нужен якорь по ключевому слову, иначе всё совпадение.
            value = match.group(1) if match.groups() else match.group(0)
            for placeholder, known in mapping.items():
                if known == value:
                    return match.group(0).replace(value, placeholder)
            counters[label] = counters.get(label, 0) + 1
            placeholder = f"[{label}_{counters[label]}]"
            mapping[placeholder] = value
            return match.group(0).replace(value, placeholder)

        scrubbed = pattern.sub(_replace, scrubbed)

    return scrubbed, mapping


def rehydrate(text: str, mapping: PiiMap) -> str:
    """Вернуть реальные значения на место плейсхолдеров после ответа модели."""
    restored = text
    for placeholder, value in mapping.items():
        restored = restored.replace(placeholder, value)
    return restored


def contains_pii(text: str) -> bool:
    """Проверка «в тексте остались персональные данные».

    Используется тестами, чтобы утверждать инвариант в той самой точке, где его
    нарушение было бы необратимым.
    """
    return any(pattern.search(text) for _, pattern in _PATTERNS)
