"""Две лёгкие головы над одним общим эмбеддингом: тема и риск (ml.md §4).

Baseline из ml.md §8: энкодер не трогаем, обучаем только головы на наших метках.
Голова здесь — центроид класса: самое дешёвое, что всё ещё честно называется
«обучено на наших данных» и укладывается в доли секунды на синтетическом датасете.

Риск вынесен в отдельную голову, потому что он ортогонален теме: внутри одной темы
бывают и безобидные, и рисковые обращения.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.ml.encoder import embed
from app.models import Risk

_TEMPERATURE = 0.08  # резкость softmax по близостям к центроидам


@dataclass(frozen=True)
class Prediction:
    """Результат горячего пути: тема, риск и уверенность по каждой голове."""

    topic: str
    conf_cls: float
    risk: Risk
    conf_risk: float


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = (scores - scores.max()) / _TEMPERATURE
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else (vector / norm).astype(np.float32)


class Classifier:
    """Головы на центроидах. `fit` отрабатывает за доли секунды на демо-датасете."""

    def __init__(self) -> None:
        self.topics: list[str] = []
        self.topic_centroids: np.ndarray | None = None
        self.risk_labels: list[str] = []
        self.risk_centroids: np.ndarray | None = None

    def fit(self, samples: list[dict[str, str]]) -> None:
        """Обучить обе головы на размеченных примерах вида {text, topic, risk}."""
        self.topics, self.topic_centroids = self._fit_head(samples, "topic")
        self.risk_labels, self.risk_centroids = self._fit_head(samples, "risk")

    @staticmethod
    def _fit_head(samples: list[dict[str, str]], key: str) -> tuple[list[str], np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        for sample in samples:
            grouped.setdefault(sample[key], []).append(embed(sample["text"]))
        labels = sorted(grouped)
        centroids = np.vstack([_l2(np.mean(np.vstack(grouped[label]), axis=0)) for label in labels])
        return labels, centroids

    def predict(self, vector: np.ndarray) -> Prediction:
        """Предсказать тему и риск по одному общему эмбеддингу.

        Один forward на обе головы — экономия на горячем пути и согласованное
        представление между классификацией и векторным поиском.
        """
        if self.topic_centroids is None or self.risk_centroids is None:
            raise RuntimeError("классификатор не обучен")

        topic_probs = _softmax(self.topic_centroids @ vector)
        topic_idx = int(np.argmax(topic_probs))
        risk_score = self._risk_score(_softmax(self.risk_centroids @ vector))

        return Prediction(
            topic=self.topics[topic_idx],
            conf_cls=float(topic_probs[topic_idx]),
            risk=_risk_from_score(risk_score),
            conf_risk=risk_score,
        )

    def _risk_score(self, risk_probs: np.ndarray) -> float:
        """Свернуть голову риска в один скор [0,1] вместо бинарной метки (ml.md §4).

        Скор удобнее метки: пороги живут в конфиге и двигаются в сторону осторожности
        без переобучения модели.
        """
        weights = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return float(
            sum(weights.get(label, 0.5) * p for label, p in zip(self.risk_labels, risk_probs))
        )


def _risk_from_score(score: float) -> Risk:
    """Перевести скор риска в уровень по порогам из конфига."""
    settings = get_settings()
    if score >= settings.risk_high_score:
        return Risk.HIGH
    if score >= settings.risk_medium_score:
        return Risk.MEDIUM
    return Risk.LOW


def load_classifier(dataset_path: str | Path = "data/train.json") -> Classifier:
    """Обучить головы на синтетическом датасете, лежащем в репозитории."""
    samples = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    classifier = Classifier()
    classifier.fit(samples)
    return classifier
