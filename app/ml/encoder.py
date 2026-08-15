"""Локальный энкодер. В PoC это мок — см. ml.md §3.

Хеширует слова и символьные триграммы в вектор фиксированной длины. Это не
семантическая модель: она ловит лексическое пересечение и ничего больше. Для
демонстрации порогов ретрива этого достаточно, а демо при этом запускается офлайн,
без скачивания весов.

В целевой системе здесь контрастивно дообученный sentence-энкодер; остальной
конвейер не меняется — подменяется только этот модуль.
"""

import hashlib
import re

import numpy as np

from app.config import get_settings

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _bucket(token: str, dim: int) -> int:
    """Устойчивый номер измерения для токена."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def embed(text: str) -> np.ndarray:
    """Закодировать текст в L2-нормализованный вектор float32."""
    settings = get_settings()
    dim = settings.embed_dim
    vector = np.zeros(dim, dtype=np.float32)

    for token in _TOKEN_RE.findall(text.lower()):
        vector[_bucket(token, dim)] += 1.0
        # Символьные триграммы дают частичный зачёт морфологическим вариантам
        # («возврат» / «возврата»), который чистый мешок слов потерял бы.
        padded = f"^{token}$"
        for i in range(len(padded) - 2):
            vector[_bucket(padded[i : i + 3], dim)] += settings.encoder_trigram_weight

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
