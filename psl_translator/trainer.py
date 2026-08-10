from __future__ import annotations

from pathlib import Path

from .dataset import SignSample, dataset_stats, load_jsonl
from .metrics import evaluate_model
from .model import PrototypeSequenceClassifier
from .reporting import write_evaluation_report


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


def train_evaluate_report(
    train_dataset: str | Path,
    eval_dataset: str | Path,
    model_out: str | Path,
    report_dir: str | Path,
) -> dict:
    train_samples = load_jsonl(train_dataset)
    eval_samples = load_jsonl(eval_dataset)
    model = PrototypeSequenceClassifier().fit(train_samples)
    model.save(model_out)
    result, _ = evaluate_model(model, eval_samples)
    report_paths = write_evaluation_report(result, report_dir, title="Persian Sign Language Model Report")
    train_stats = dataset_stats(train_samples)
    return {
        "train_samples": train_stats.sample_count,
        "eval_samples": result.sample_count,
        "accuracy": result.accuracy,
        "model_out": str(model_out),
        "report": report_paths,
    }


def split_samples_by_ratio(samples: list[SignSample], eval_ratio: float = 0.25) -> tuple[list[SignSample], list[SignSample]]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    train: list[SignSample] = []
    eval_: list[SignSample] = []
    per_label_seen: dict[str, int] = {}
    for sample in samples:
        count = per_label_seen.get(sample.label, 0)
        per_label_seen[sample.label] = count + 1
        # deterministic split, no random seed ceremony
        if count % round(1 / eval_ratio) == 0:
            eval_.append(sample)
        else:
            train.append(sample)
    return train, eval_
