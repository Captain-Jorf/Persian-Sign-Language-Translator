from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

from .constants import (
    HAND_LANDMARK_COUNT,
    HAND_VECTOR_SIZE,
    POINT_DIMS,
    TWO_HAND_VECTOR_SIZE,
)

Point = tuple[float, float, float]
FrameVector = list[float]
SequenceVector = list[FrameVector]


def _as_point(raw: Sequence[float] | object) -> Point:
    """Accept tuple/list points or MediaPipe landmark-ish objects."""
    if hasattr(raw, "x") and hasattr(raw, "y") and hasattr(raw, "z"):
        return float(raw.x), float(raw.y), float(raw.z)
    if len(raw) < 2:  # type: ignore[arg-type]
        raise ValueError("landmark point must have at least x and y")
    x = float(raw[0])  # type: ignore[index]
    y = float(raw[1])  # type: ignore[index]
    z = float(raw[2]) if len(raw) > 2 else 0.0  # type: ignore[arg-type,index]
    return x, y, z


def normalize_hand_landmarks(landmarks: Sequence[Sequence[float] | object] | None) -> FrameVector:
    """Normalize one hand to a translation/scale-tolerant 63-number vector.

    The wrist is the origin. Scale is max distance from wrist. This is not magic;
    it just removes the most annoying camera-distance noise before the sequence
    model sees the data.
    """
    if landmarks is None:
        return [0.0] * HAND_VECTOR_SIZE
    if len(landmarks) != HAND_LANDMARK_COUNT:
        raise ValueError(f"expected {HAND_LANDMARK_COUNT} hand landmarks, got {len(landmarks)}")

    points = [_as_point(point) for point in landmarks]
    wrist = points[0]
    centered: list[Point] = [
        (x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in points
    ]

    scale = max(sqrt(x * x + y * y + z * z) for x, y, z in centered)
    if scale <= 1e-8:
        scale = 1.0

    vector: FrameVector = []
    for x, y, z in centered:
        vector.extend([x / scale, y / scale, z / scale])
    return vector


def make_frame_vector(
    left_hand: Sequence[Sequence[float] | object] | None = None,
    right_hand: Sequence[Sequence[float] | object] | None = None,
) -> FrameVector:
    """Create the canonical two-hand vector: left hand first, right hand second."""
    vector = normalize_hand_landmarks(left_hand) + normalize_hand_landmarks(right_hand)
    if len(vector) != TWO_HAND_VECTOR_SIZE:
        raise AssertionError("feature vector size drifted; check constants")
    return vector


def resample_sequence(frames: Sequence[Sequence[float]], target_length: int) -> SequenceVector:
    """Linearly resample a variable-length sequence to a fixed length.

    Works without NumPy on purpose. The CLI and tests stay light, while the real
    training stack can still use NumPy/Torch when installed.
    """
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if not frames:
        return [[0.0] * TWO_HAND_VECTOR_SIZE for _ in range(target_length)]

    frame_size = len(frames[0])
    if frame_size == 0:
        raise ValueError("frames cannot be empty vectors")
    for index, frame in enumerate(frames):
        if len(frame) != frame_size:
            raise ValueError(f"frame {index} has size {len(frame)}, expected {frame_size}")

    if len(frames) == 1:
        return [list(map(float, frames[0])) for _ in range(target_length)]
    if target_length == 1:
        return [list(map(float, frames[0]))]

    output: SequenceVector = []
    source_last = len(frames) - 1
    for out_i in range(target_length):
        position = out_i * source_last / (target_length - 1)
        left_index = int(position)
        right_index = min(left_index + 1, source_last)
        ratio = position - left_index
        left = frames[left_index]
        right = frames[right_index]
        output.append([
            (float(left[j]) * (1.0 - ratio)) + (float(right[j]) * ratio)
            for j in range(frame_size)
        ])
    return output


def flatten_sequence(frames: Sequence[Sequence[float]]) -> FrameVector:
    flat: FrameVector = []
    for frame in frames:
        flat.extend(float(value) for value in frame)
    return flat


def sequence_summary(frames: Sequence[Sequence[float]]) -> FrameVector:
    """Cheap feature summary for the baseline classifier.

    It stores mean, standard deviation, first frame, last frame, and simple motion
    delta. GRU/LSTM is the serious route; this baseline exists so the project is
    runnable before we collect a proper PSL dataset.
    """
    if not frames:
        return [0.0] * (TWO_HAND_VECTOR_SIZE * 5)
    frame_size = len(frames[0])
    count = len(frames)

    means = [0.0] * frame_size
    for frame in frames:
        if len(frame) != frame_size:
            raise ValueError("all frames must have the same size")
        for i, value in enumerate(frame):
            means[i] += float(value)
    means = [value / count for value in means]

    variances = [0.0] * frame_size
    for frame in frames:
        for i, value in enumerate(frame):
            diff = float(value) - means[i]
            variances[i] += diff * diff
    stds = [sqrt(value / count) for value in variances]

    first = [float(value) for value in frames[0]]
    last = [float(value) for value in frames[-1]]
    delta = [last[i] - first[i] for i in range(frame_size)]
    return means + stds + first + last + delta


def chunked(values: Iterable[float], size: int) -> list[list[float]]:
    values_list = list(values)
    if size <= 0:
        raise ValueError("size must be positive")
    if len(values_list) % size != 0:
        raise ValueError("values length must be divisible by chunk size")
    return [values_list[i : i + size] for i in range(0, len(values_list), size)]
