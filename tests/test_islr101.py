from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from psl_translator.islr101 import OPENPOSE_VECTOR_SIZE, convert_manifest_to_jsonl, load_openpose_sequence
from psl_translator.dataset import load_jsonl


class ISLR101Tests(unittest.TestCase):
    def test_load_openpose_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.json"
            payload = {
                "people": [
                    {
                        "pose_keypoints_2d": [1, 2, 0.9] * 25,
                        "hand_left_keypoints_2d": [3, 4, 0.8] * 21,
                        "hand_right_keypoints_2d": [5, 6, 0.7] * 21,
                    }
                ]
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            frames = load_openpose_sequence(path)
            self.assertEqual(len(frames), 1)
            self.assertEqual(len(frames[0]), OPENPOSE_VECTOR_SIZE)

    def test_convert_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "sample01"
            frame_dir.mkdir()
            (frame_dir / "000.json").write_text(json.dumps({"people": [{}]}), encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "signer", "skeleton_path", "split"])
                writer.writeheader()
                writer.writerow({"label": "سلام", "signer": "s01", "skeleton_path": "sample01", "split": "train"})
            out = root / "out.jsonl"
            count = convert_manifest_to_jsonl(manifest, root, out, split="train")
            self.assertEqual(count, 1)
            samples = load_jsonl(out)
            self.assertEqual(samples[0].label, "سلام")
            self.assertEqual(len(samples[0].frames[0]), OPENPOSE_VECTOR_SIZE)


if __name__ == "__main__":
    unittest.main()
