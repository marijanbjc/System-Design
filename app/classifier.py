"""Two lightweight heads over one shared embedding: topic and risk (ml.md 4).

Baseline from ml.md 8: the encoder is not trained, only the heads are fitted on our
labels. Here a head is a class centroid — the cheapest thing that still counts as
"trained on our data" and fits in milliseconds on the synthetic dataset.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.embed import embed
from app.models import Risk

_TEMPERATURE = 0.08  # softmax sharpness over centroid similarities


@dataclass(frozen=True)
class Prediction:
    topic: str
    conf_cls: float
    risk: Risk
    conf_risk: float


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = (scores - scores.max()) / _TEMPERATURE
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


class Classifier:
    """Nearest-centroid heads. `fit` runs in well under a second on the demo dataset."""

    def __init__(self) -> None:
        self.topics: list[str] = []
        self.topic_centroids: np.ndarray | None = None
        self.risk_labels: list[str] = []
        self.risk_centroids: np.ndarray | None = None

    def fit(self, samples: list[dict[str, str]]) -> None:
        """Fit both heads from labelled samples: {text, topic, risk}."""
        self.topics, self.topic_centroids = self._fit_head(samples, "topic")
        self.risk_labels, self.risk_centroids = self._fit_head(samples, "risk")

    @staticmethod
    def _fit_head(samples: list[dict[str, str]], key: str) -> tuple[list[str], np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        for sample in samples:
            grouped.setdefault(sample[key], []).append(embed(sample["text"]))
        labels = sorted(grouped)
        centroids = np.vstack(
            [_l2(np.mean(np.vstack(grouped[label]), axis=0)) for label in labels]
        )
        return labels, centroids

    def predict(self, vector: np.ndarray) -> Prediction:
        """Predict topic and risk from a single shared embedding."""
        if self.topic_centroids is None or self.risk_centroids is None:
            raise RuntimeError("classifier is not fitted")

        topic_probs = _softmax(self.topic_centroids @ vector)
        topic_idx = int(np.argmax(topic_probs))

        risk_probs = _softmax(self.risk_centroids @ vector)
        risk_score = self._risk_score(risk_probs)

        return Prediction(
            topic=self.topics[topic_idx],
            conf_cls=float(topic_probs[topic_idx]),
            risk=_risk_from_score(risk_score),
            conf_risk=risk_score,
        )

    def _risk_score(self, risk_probs: np.ndarray) -> float:
        """Collapse the risk head into a single [0,1] score (ml.md 4).

        A score rather than a label: thresholds then live in the config and can be
        moved toward caution without retraining anything.
        """
        weights = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return float(sum(weights.get(label, 0.5) * p for label, p in zip(self.risk_labels, risk_probs)))


def _risk_from_score(score: float) -> Risk:
    settings = get_settings()
    if score >= settings.risk_high_score:
        return Risk.HIGH
    if score >= settings.risk_medium_score:
        return Risk.MEDIUM
    return Risk.LOW


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else (vector / norm).astype(np.float32)


def load_classifier(dataset_path: str | Path = "data/train.json") -> Classifier:
    """Fit the heads from the synthetic dataset shipped with the repository."""
    samples = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    classifier = Classifier()
    classifier.fit(samples)
    return classifier
