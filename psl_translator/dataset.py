from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .constants import DEFAULT_SEQUENCE_LENGTH, TWO_HAND_VECTOR_SIZE
from .features import resample_sequence


@dataclass(slots=True)
class SignSample:
    label: str
    frames: list[list[float]]
    signer_id: str | None = None
    source: str | None = None
    meta: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def normalized(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH) -> "SignSample":
        return SignSample(
            label=self.label,
            frames=resample_sequence(self.frames, sequence_length),
            signer_id=self.signer_id,
            source=self.source,
            meta=dict(self.meta),
        )

    def validate(self) -> None:
        if not self.label.strip():
            raise ValueError("sample label cannot be empty")
        if not self.frames:
            raise ValueError("sample must contain at least one frame")
        for frame_index, frame in enumerate(self.frames):
            if len(frame) != TWO_HAND_VECTOR_SIZE:
                raise ValueError(
                    f"frame {frame_index} has {len(frame)} values, expected {TWO_HAND_VECTOR_SIZE}"
                )

    def to_json(self) -> str:
        self.validate()
        return json.dumps(
            {
                "label": self.label,
                "signer_id": self.signer_id,
                "source": self.source,
                "frames": self.frames,
                "meta": self.meta,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "SignSample":
        sample = cls(
            label=str(payload["label"]),
            frames=[[float(v) for v in frame] for frame in payload["frames"]],
            signer_id=payload.get("signer_id"),
            source=payload.get("source"),
            meta=dict(payload.get("meta") or {}),
        )
        sample.validate()
        return sample


@dataclass(slots=True)
class DatasetStats:
    sample_count: int
    label_counts: dict[str, int]
    signer_counts: dict[str, int]

    @property
    def label_count(self) -> int:
        return len(self.label_counts)


def write_jsonl(samples: Iterable[SignSample], path: str | Path, append: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json())
            handle.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[SignSample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean = line.strip()
            if not clean:
                continue
            try:
                yield SignSample.from_dict(json.loads(clean))
            except Exception as exc:  # pragma: no cover - message matters more than branch count
                raise ValueError(f"bad sample at {path}:{line_number}: {exc}") from exc


def load_jsonl(path: str | Path, sequence_length: int = DEFAULT_SEQUENCE_LENGTH) -> list[SignSample]:
    return [sample.normalized(sequence_length) for sample in iter_jsonl(path)]


def dataset_stats(samples: Sequence[SignSample]) -> DatasetStats:
    label_counts: dict[str, int] = {}
    signer_counts: dict[str, int] = {}
    for sample in samples:
        label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
        signer = sample.signer_id or "unknown"
        signer_counts[signer] = signer_counts.get(signer, 0) + 1
    return DatasetStats(
        sample_count=len(samples),
        label_counts=dict(sorted(label_counts.items())),
        signer_counts=dict(sorted(signer_counts.items())),
    )


def split_by_signer(
    samples: Sequence[SignSample], holdout_signers: set[str]
) -> tuple[list[SignSample], list[SignSample]]:
    """Split without leaking one signer's motion style into both sides."""
    train: list[SignSample] = []
    test: list[SignSample] = []
    for sample in samples:
        if sample.signer_id in holdout_signers:
            test.append(sample)
        else:
            train.append(sample)
    return train, test
