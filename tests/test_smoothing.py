from __future__ import annotations

import unittest

from psl_translator.smoothing import PredictionSmoother


class SmoothingTests(unittest.TestCase):
    def test_waits_for_majority(self) -> None:
        smoother = PredictionSmoother(window_size=5, min_confidence=0.6, min_votes=3)
        self.assertIsNone(smoother.update("سلام", 0.8))
        self.assertIsNone(smoother.update("آب", 0.9))
        self.assertIsNone(smoother.update("سلام", 0.7))
        stable = smoother.update("سلام", 0.95)
        self.assertIsNotNone(stable)
        assert stable is not None
        self.assertEqual(stable.label, "سلام")


if __name__ == "__main__":
    unittest.main()
