from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from typing import Sequence

from .constants import DEFAULT_SEQUENCE_LENGTH
from .dataset import SignSample
from .features import resample_sequence, sequence_summary


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    label: str
    confidence: float
    scores: dict[str, float]


class PrototypeSequenceClassifier:
    """Tiny nearest-prototype classifier used as a runnable baseline.

    No, this is not the final model. It is the smoke-test model: useful while the
    real PSL dataset is still being recorded and cleaned. Once we have enough
    signer diversity, the GRU/LSTM trainer can replace this without changing the
    camera or CLI flow.
    """

    model_type = "prototype-sequence-classifier"

    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH) -> None:
        self.sequence_length = sequence_length
        self.labels: list[str] = []
        self.prototypes: dict[str, list[float]] = {}

    def fit(self, samples: Sequence[SignSample]) -> "PrototypeSequenceClassifier":
        if not samples:
            raise ValueError("cannot train on an empty dataset")

        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for sample in samples:
            normalized = resample_sequence(sample.frames, self.sequence_length)
            features = sequence_summary(normalized)
            if sample.label not in sums:
                sums[sample.label] = [0.0] * len(features)
                counts[sample.label] = 0
            for i, value in enumerate(features):
                sums[sample.label][i] += value
            counts[sample.label] += 1

        self.labels = sorted(sums)
        self.prototypes = {
            label: [value / counts[label] for value in sums[label]] for label in self.labels
        }
        return self

    def predict(self, frames: Sequence[Sequence[float]]) -> ModelPrediction:
        if not self.prototypes:
            raise RuntimeError("model is not trained or loaded")
        normalized = resample_sequence(frames, self.sequence_length)
        features = sequence_summary(normalized)

        distances = {
            label: _euclidean(features, prototype)
            for label, prototype in self.prototypes.items()
        }
        # Softmax over negative distances. Good enough for ranking, not calibrated.
        max_neg = max(-distance for distance in distances.values())
        exp_scores = {
            label: exp((-distance) - max_neg) for label, distance in distances.items()
        }
        total = sum(exp_scores.values()) or 1.0
        scores = {label: score / total for label, score in exp_scores.items()}
        best_label = max(scores, key=scores.get)
        return ModelPrediction(best_label, scores[best_label], dict(sorted(scores.items())))

    def save(self, path: str | Path) -> None:
        if not self.prototypes:
            raise RuntimeError("cannot save an untrained model")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "model_type": self.model_type,
                    "sequence_length": self.sequence_length,
                    "labels": self.labels,
                    "prototypes": self.prototypes,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeSequenceClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"unsupported model type: {payload.get('model_type')}")
        model = cls(sequence_length=int(payload.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)))
        model.labels = [str(label) for label in payload["labels"]]
        model.prototypes = {
            str(label): [float(value) for value in values]
            for label, values in payload["prototypes"].items()
        }
        return model


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same size")
    return sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
