from __future__ import annotations

from collections import deque
from pathlib import Path
from time import time
from typing import Iterable

from psl_translator.constants import DEFAULT_SEQUENCE_LENGTH, LEFT_HAND, RIGHT_HAND
from psl_translator.dataset import SignSample, write_jsonl
from psl_translator.features import make_frame_vector
from psl_translator.model import PrototypeSequenceClassifier
from psl_translator.smoothing import PredictionSmoother


def _require_camera_stack():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        import mediapipe as mp

        try:
            hands_module = mp.solutions.hands
        except AttributeError:
            from mediapipe.python.solutions import hands as hands_module
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return cv2, hands_module


def _hands_from_results(results) -> tuple[Iterable | None, Iterable | None]:
    left_hand = None
    right_hand = None

    if not getattr(results, "multi_hand_landmarks", None):
        return left_hand, right_hand

    handedness = getattr(results, "multi_handedness", None) or []

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        label = None
        if index < len(handedness):
            label = handedness[index].classification[0].label

        points = hand_landmarks.landmark

        if label == LEFT_HAND:
            left_hand = points
        elif label == RIGHT_HAND:
            right_hand = points
        elif left_hand is None:
            left_hand = points
        else:
            right_hand = points

    return left_hand, right_hand


def collect_samples(
    label: str,
    out_path: str | Path,
    signer_id: str | None = None,
    seconds: float = 3.0,
    camera_index: int = 0,
) -> int:
    cv2, mp_hands = _require_camera_stack()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")

    frames: list[list[float]] = []
    deadline = time() + seconds
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    try:
        while time() < deadline:
            ok, image = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            left, right = _hands_from_results(results)
            frames.append(make_frame_vector(left, right))

            cv2.putText(
                image,
                f"Recording: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("PSL Collector", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    sample = SignSample(label=label, frames=frames, signer_id=signer_id, source="webcam")
    write_jsonl([sample], out_path, append=True)
    return len(frames)


def run_realtime_translation(
    model_path: str | Path,
    camera_index: int = 0,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> None:
    cv2, mp_hands = _require_camera_stack()

    model = PrototypeSequenceClassifier.load(model_path)
    smoother = PredictionSmoother()
    buffer: deque[list[float]] = deque(maxlen=sequence_length)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    current_text = "..."

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            left, right = _hands_from_results(results)
            buffer.append(make_frame_vector(left, right))

            if len(buffer) == sequence_length:
                prediction = model.predict(list(buffer))
                stable = smoother.update(prediction.label, prediction.confidence)
                if stable:
                    current_text = f"{stable.label} ({stable.confidence:.2f})"

            cv2.putText(
                image,
                current_text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (50, 220, 50),
                2,
            )

            cv2.imshow("Persian Sign Language Translator - Camera", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()