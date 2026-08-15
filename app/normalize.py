"""Rule-based normalization and regex safety detectors — hot path, no models, no LLM."""

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_QUOTE_LINE = re.compile(r"^\s*(>+|On .+ wrote:|\d{1,2}\.\d{1,2}\.\d{2,4}.*пишет:).*$", re.MULTILINE)
_SIGNATURE = re.compile(
    r"(?:^|\n)\s*(--\s*$|С уважением|Best regards|Sent from my|Отправлено с)[\s\S]*",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

MAX_LENGTH = 4000

# Explicit imperatives and markup traps. Catches the obvious layer only: paraphrases,
# multi-turn attacks and other languages are out of reach and that is stated in the docs.
_INJECTION_PATTERNS = [
    r"игнорируй\w*\s+(все\s+)?(предыдущ\w+|прежн\w+|выше)",
    r"забудь\w*\s+(все\s+)?(предыдущ\w+|инструкц\w+|указан\w+)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior)\s+",
    r"ты\s+(теперь|отныне)\s+\w+",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a|an)\s+",
    r"(system|assistant|user)\s*:\s*",
    r"<\|.*?\|>",
    r"新的指令|new\s+system\s+prompt",
    r"(закрой|close)\s+тикет\s+автоматич",
]

# Toxicity / threats / topics we must never answer automatically.
_UNSAFE_PATTERNS = [
    r"\b(идиот|дебил|тварь|ублюд\w+|сволоч\w+)\b",
    r"\b(убью|прикончу|найду\s+тебя|сожгу\s+ваш)\b",
    r"\b(подам\s+в\s+суд|прокуратур\w+|роспотребнадзор)\b",
    r"\b(взлом\w+\s+аккаунт|украли\s+деньги|мошенник\w*)\b",
    r"\b(чарджбэк|chargeback)\b",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def normalize(text_raw: str) -> str:
    """Strip HTML, quoted replies, signatures and collapse whitespace.

    Deterministic and identical for every channel: that is how heterogeneous input
    is reduced to one shape without summarization and without an LLM.
    """
    text = _HTML_TAG.sub(" ", text_raw)
    text = _QUOTE_LINE.sub("", text)
    text = _SIGNATURE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()[:MAX_LENGTH]


def detect_injection(text: str) -> bool:
    """Cheap first layer against prompt injection.

    A hit routes the ticket to an operator; it never blocks the user. "Please ignore
    my previous letter" is a real support case and must not turn into a refusal.
    """
    return bool(_INJECTION_RE.search(text))


def detect_unsafe(text: str) -> bool:
    """Coarse but terminal safety prefilter: a hit sends the ticket straight to a human."""
    return bool(_UNSAFE_RE.search(text))
