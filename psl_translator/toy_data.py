from __future__ import annotations

from math import sin
from random import Random

from .constants import TWO_HAND_VECTOR_SIZE
from .dataset import SignSample


def make_toy_samples(labels: list[str] | None = None, per_label: int = 6, frames: int = 18) -> list[SignSample]:
    """Generate deterministic fake landmark sequences for smoke tests.

    This is not PSL data. It is a guardrail for the pipeline, same way I use a
    two-row CSV before trusting a training script with the real messy file.
    """
    labels = labels or ["سلام", "کمک", "آب"]
    rng = Random(42)
    samples: list[SignSample] = []
    for label_index, label in enumerate(labels):
        base = (label_index + 1) * 0.08
        for sample_index in range(per_label):
            sequence: list[list[float]] = []
            signer = f"toy-signer-{sample_index % 3}"
            noise_seed = rng.random() * 0.01
            for frame_index in range(frames):
                phase = frame_index / max(1, frames - 1)
                frame = [0.0] * TWO_HAND_VECTOR_SIZE
                for feature_index in range(TWO_HAND_VECTOR_SIZE):
                    if feature_index % 9 == 0:
                        frame[feature_index] = base + phase * (0.04 + label_index * 0.01) + noise_seed
                    elif feature_index % 9 == 1:
                        frame[feature_index] = base * 0.5 + sin(phase * 3.14) * 0.03
                    elif feature_index % 9 == 2:
                        frame[feature_index] = base * 0.25
                sequence.append(frame)
            samples.append(
                SignSample(
                    label=label,
                    frames=sequence,
                    signer_id=signer,
                    source="synthetic-toy-data",
                    meta={"sample_index": sample_index},
                )
            )
    return samples
