from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .dataset import SignSample, write_jsonl

OPENPOSE_BODY_POINTS = 25
OPENPOSE_HAND_POINTS = 21
OPENPOSE_VECTOR_SIZE = (OPENPOSE_BODY_POINTS + OPENPOSE_HAND_POINTS * 2) * 3


@dataclass(frozen=True, slots=True)
class ISLR101Facts:
    name: str = "ISLR101"
    arxiv: str = "2503.12451"
    title: str = "ISLR101: an Iranian Word-Level Sign Language Recognition Dataset"
    samples: int = 4614
    classes: int = 101
    signers: int = 10
    fps: int = 25
    resolution: str = "800x600"
    skeleton: str = "OpenPose: 25 body + 21 left hand + 21 right hand keypoints"
    access_note: str = "Dataset is available from the Social & Cognitive Robotics Laboratory archive, upon request for academic/research use."


def facts() -> dict:
    return asdict(ISLR101Facts())


def convert_manifest_to_jsonl(
    manifest_path: str | Path,
    root: str | Path,
    out_path: str | Path,
    split: str | None = None,
) -> int:
    """Convert an ISLR101-style manifest to this repo's JSONL sequence format.

    Expected CSV columns are intentionally flexible:
    - label/gloss/class/sign
    - skeleton_path/keypoints_path/openpose_path/path
    - signer/signer_id/subject
    - split/subset (optional)

    The skeleton path can point to a directory of OpenPose frame JSON files or a
    single JSON file containing a `frames` list. Real datasets rarely arrive in
    the exact shape you want. This importer tries to be strict about features,
    not strict about column naming.
    """
    root_path = Path(root)
    rows = _read_manifest(manifest_path)
    samples: list[SignSample] = []
    for row in rows:
        row_split = _pick(row, "split", "subset", "set")
        if split and row_split and row_split.lower() != split.lower():
            continue

        label = _pick(row, "label", "gloss", "class", "sign", "word")
        skeleton_rel = _pick(row, "skeleton_path", "keypoints_path", "openpose_path", "pose_path", "path")
        if not label or not skeleton_rel:
            raise ValueError("manifest rows need label/gloss and skeleton/keypoints path columns")

        skeleton_path = Path(skeleton_rel)
        if not skeleton_path.is_absolute():
            skeleton_path = root_path / skeleton_path
        frames = load_openpose_sequence(skeleton_path)
        samples.append(
            SignSample(
                label=label,
                frames=frames,
                signer_id=_pick(row, "signer", "signer_id", "subject", "participant") or None,
                source=str(skeleton_path),
                meta={"split": row_split or split or "unknown", "dataset": "ISLR101"},
            )
        )

    write_jsonl(samples, out_path)
    return len(samples)


def load_openpose_sequence(path: str | Path) -> list[list[float]]:
    source = Path(path)
    if source.is_dir():
        frames = [_load_openpose_frame(frame_path) for frame_path in sorted(source.glob("*.json"))]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if "frames" in payload:
            frames = [_frame_from_payload(frame) for frame in payload["frames"]]
        else:
            frames = [_frame_from_payload(payload)]

    frames = [frame for frame in frames if frame]
    if not frames:
        raise ValueError(f"no OpenPose frames found in {source}")
    for index, frame in enumerate(frames):
        if len(frame) != OPENPOSE_VECTOR_SIZE:
            raise ValueError(f"frame {index} in {source} has {len(frame)} values, expected {OPENPOSE_VECTOR_SIZE}")
    return frames


def _load_openpose_frame(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _frame_from_payload(payload)


def _frame_from_payload(payload: dict) -> list[float]:
    # Common OpenPose output: {"people": [{"pose_keypoints_2d": [...]}]}
    if "people" in payload:
        people = payload.get("people") or []
        person = people[0] if people else {}
    else:
        person = payload

    pose = _fixed(person.get("pose_keypoints_2d"), OPENPOSE_BODY_POINTS * 3)
    left = _fixed(person.get("hand_left_keypoints_2d"), OPENPOSE_HAND_POINTS * 3)
    right = _fixed(person.get("hand_right_keypoints_2d"), OPENPOSE_HAND_POINTS * 3)
    return pose + left + right


def _fixed(values: Iterable[float] | None, size: int) -> list[float]:
    if values is None:
        return [0.0] * size
    clean = [float(value) for value in values]
    if len(clean) < size:
        clean.extend([0.0] * (size - len(clean)))
    return clean[:size]


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(k).strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def _pick(row: dict[str, str], *names: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return ""
