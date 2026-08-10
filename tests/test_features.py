from __future__ import annotations

import unittest

from psl_translator.constants import HAND_VECTOR_SIZE, TWO_HAND_VECTOR_SIZE
from psl_translator.features import make_frame_vector, normalize_hand_landmarks, resample_sequence


class FeatureTests(unittest.TestCase):
    def test_missing_hand_becomes_zero_vector(self) -> None:
        vector = normalize_hand_landmarks(None)
        self.assertEqual(len(vector), HAND_VECTOR_SIZE)
        self.assertTrue(all(value == 0.0 for value in vector))

    def test_make_frame_vector_has_two_hands(self) -> None:
        vector = make_frame_vector(None, None)
        self.assertEqual(len(vector), TWO_HAND_VECTOR_SIZE)

    def test_normalization_removes_wrist_translation(self) -> None:
        hand = [(10 + i, 20 + i * 0.5, 1.0) for i in range(21)]
        vector = normalize_hand_landmarks(hand)
        self.assertEqual(vector[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(max(abs(value) for value in vector), 0.8944271909999159)

    def test_resample_sequence_keeps_endpoints(self) -> None:
        frames = [[0.0, 10.0], [10.0, 20.0]]
        out = resample_sequence(frames, 5)
        self.assertEqual(out[0], [0.0, 10.0])
        self.assertEqual(out[-1], [10.0, 20.0])
        self.assertEqual(out[2], [5.0, 15.0])


if __name__ == "__main__":
    unittest.main()
