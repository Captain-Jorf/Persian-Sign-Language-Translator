from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dataset import SignSample
from .model import ModelPrediction, PrototypeSequenceClassifier


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    accuracy: float
    sample_count: int
    labels: list[str]
    confusion: list[list[int]]
    per_label_accuracy: dict[str, float]


def evaluate_model(
    model: PrototypeSequenceClassifier,
    samples: Sequence[SignSample],
) -> tuple[EvaluationResult, list[ModelPrediction]]:
    if not samples:
        raise ValueError("cannot evaluate on an empty dataset")

    labels = sorted({sample.label for sample in samples} | set(model.labels))
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = [[0 for _ in labels] for _ in labels]
    predictions: list[ModelPrediction] = []
    correct = 0

    for sample in samples:
        prediction = model.predict(sample.frames)
        predictions.append(prediction)
        true_index = label_to_index[sample.label]
        pred_index = label_to_index[prediction.label]
        confusion[true_index][pred_index] += 1
        if prediction.label == sample.label:
            correct += 1

    per_label_accuracy: dict[str, float] = {}
    for label, index in label_to_index.items():
        total = sum(confusion[index])
        per_label_accuracy[label] = (confusion[index][index] / total) if total else 0.0

    result = EvaluationResult(
        accuracy=correct / len(samples),
        sample_count=len(samples),
        labels=labels,
        confusion=confusion,
        per_label_accuracy=per_label_accuracy,
    )
    return result, predictions
