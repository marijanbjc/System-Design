"""Local encoder. In the PoC it is a deterministic mock — see ml.md 3.

Hashes word unigrams and character trigrams into a fixed-width vector. It is not a
semantic model: it captures lexical overlap only. That is enough to demonstrate the
retrieval thresholds and keeps the demo runnable offline with no model download.
In the target system this is a contrastively fine-tuned sentence encoder; the rest
of the pipeline does not change, only this module is swapped.
"""

import hashlib
import re

import numpy as np

from app.config import get_settings

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def embed(text: str) -> np.ndarray:
    """Encode text into an L2-normalized float32 vector."""
    settings = get_settings()
    dim = settings.embed_dim
    vector = np.zeros(dim, dtype=np.float32)

    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        vector[_bucket(token, dim)] += 1.0
        # Character trigrams give partial credit to morphological variants
        # ("возврат" / "возврата"), which a pure bag of words would miss.
        padded = f"^{token}$"
        for i in range(len(padded) - 2):
            vector[_bucket(padded[i : i + 3], dim)] += 0.5

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors, clamped to [-1, 1]."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))
