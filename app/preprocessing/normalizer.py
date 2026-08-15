"""Нормализация текста правилами — горячий путь, без моделей и без LLM.

Обращения приходят из разных каналов и в разном виде: письмо тянет за собой HTML,
подпись и цитату предыдущей переписки, чат — почти чистый текст. Нормализация
приводит всё это к одной форме детерминированно, чтобы эмбеддер и классификатор
работали с сопоставимым входом.
"""

import re

from app.config import get_settings

_HTML_TAG = re.compile(r"<[^>]+>")
_QUOTE_LINE = re.compile(r"^\s*(>+|On .+ wrote:|\d{1,2}\.\d{1,2}\.\d{2,4}.*пишет:).*$", re.MULTILINE)
_SIGNATURE = re.compile(
    r"(?:^|\n)\s*(--\s*$|С уважением|Best regards|Sent from my|Отправлено с)[\s\S]*",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize(text_raw: str) -> str:
    """Срезать разметку, цитаты и подпись, схлопнуть пробелы, обрезать по длине."""
    text = _HTML_TAG.sub(" ", text_raw)
    text = _QUOTE_LINE.sub("", text)
    text = _SIGNATURE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()[: get_settings().max_text_length]
