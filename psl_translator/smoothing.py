from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .constants import DEFAULT_MIN_CONFIDENCE


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    confidence: float


class PredictionSmoother:
    """Majority-vote smoother for jumpy webcam predictions."""

    def __init__(
        self,
        window_size: int = 7,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_votes: int | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_votes = min_votes or max(2, (window_size // 2) + 1)
        self._items: deque[Prediction] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._items.clear()

    def update(self, label: str, confidence: float) -> Prediction | None:
        self._items.append(Prediction(label=label, confidence=float(confidence)))
        confident = [item for item in self._items if item.confidence >= self.min_confidence]
        if len(confident) < self.min_votes:
            return None

        counts = Counter(item.label for item in confident)
        label, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return None
        avg_confidence = sum(item.confidence for item in confident if item.label == label) / votes
        return Prediction(label=label, confidence=avg_confidence)
