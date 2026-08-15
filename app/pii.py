"""Reversible PII pseudonymization: the only text that may cross our perimeter.

The replacement is reversible on purpose. "Where is my order 77-881234?" answered
without the order number is useless, so placeholders are restored after the model
call from a short-lived local vault.
"""

import re

# Order matters: the most specific patterns run first so a card number is not
# eaten by the generic digit-sequence rule.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE", re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b")),
    ("ORDER", re.compile(r"\b(?:заказ[а-я]*|order)\s*[№#nN]?\s*([\w-]{4,})", re.IGNORECASE)),
    ("NAME", re.compile(r"\b(?:меня зовут|моё имя|мое имя|я)\s+([А-ЯЁ][а-яё]{2,})\b")),
    ("ADDRESS", re.compile(r"\b(?:ул\.|улица|пр-т|проспект)\s+[А-ЯЁа-яё\w\s.,-]{3,40}")),
]


class PiiMap(dict[str, str]):
    """Placeholder -> original value. Lives in the vault, never leaves our perimeter."""


def scrub(text: str) -> tuple[str, PiiMap]:
    """Replace personal data with stable placeholders and return the reverse map.

    In the target system this is backed by a NER model (Natasha) on top of the regex
    layer; in the PoC regexes alone carry it, which is stated as a simplification.
    """
    mapping = PiiMap()
    counters: dict[str, int] = {}
    scrubbed = text

    for label, pattern in _PATTERNS:
        def _replace(match: re.Match[str], label: str = label) -> str:
            # Group 1 when the pattern needs a keyword anchor, whole match otherwise.
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
    """Put the real values back after the model answered."""
    restored = text
    for placeholder, value in mapping.items():
        restored = restored.replace(placeholder, value)
    return restored


def contains_pii(text: str) -> bool:
    """Used by tests to assert that nothing personal ever reaches the LLM client."""
    return any(pattern.search(text) for _, pattern in _PATTERNS)
