from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from psl_translator.dataset import dataset_stats, load_jsonl, write_jsonl
from psl_translator.model import PrototypeSequenceClassifier
from psl_translator.toy_data import make_toy_samples
from psl_translator.trainer import train_prototype_model


class ModelPipelineTests(unittest.TestCase):
    def test_toy_dataset_roundtrip_and_training(self) -> None:
        samples = make_toy_samples(per_label=3, frames=8)
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "toy.jsonl"
            model_path = Path(tmp) / "model.json"
            write_jsonl(samples, dataset_path)
            loaded = load_jsonl(dataset_path)
            stats = dataset_stats(loaded)
            self.assertEqual(stats.sample_count, 9)
            self.assertEqual(stats.label_count, 3)

            report = train_prototype_model(dataset_path, model_path)
            self.assertEqual(report["samples"], 9)
            self.assertTrue(model_path.exists())

            model = PrototypeSequenceClassifier.load(model_path)
            prediction = model.predict(samples[0].frames)
            self.assertEqual(prediction.label, samples[0].label)
            self.assertGreater(prediction.confidence, 0.3)


if __name__ == "__main__":
    unittest.main()
