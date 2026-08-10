from __future__ import annotations

from pathlib import Path

from .dataset import dataset_stats, load_jsonl
from .model import PrototypeSequenceClassifier


def train_prototype_model(dataset_path: str | Path, model_out: str | Path) -> dict:
    samples = load_jsonl(dataset_path)
    model = PrototypeSequenceClassifier().fit(samples)
    model.save(model_out)
    stats = dataset_stats(samples)
    return {
        "samples": stats.sample_count,
        "labels": stats.label_counts,
        "signers": stats.signer_counts,
        "model_out": str(model_out),
    }
