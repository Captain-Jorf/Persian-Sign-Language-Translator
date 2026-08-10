from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psl_translator.dataset import write_jsonl
from psl_translator.toy_data import make_toy_samples
from psl_translator.trainer import train_evaluate_report


def main() -> int:
    data_dir = Path("data/demo")
    report_dir = Path("reports/demo")
    model_path = Path("models/prototype_model.json")
    data_dir.mkdir(parents=True, exist_ok=True)

    train = make_toy_samples(per_label=8, frames=20)
    eval_ = make_toy_samples(per_label=3, frames=16)
    write_jsonl(train, data_dir / "toy_train.jsonl")
    write_jsonl(eval_, data_dir / "toy_eval.jsonl")

    report = train_evaluate_report(
        train_dataset=data_dir / "toy_train.jsonl",
        eval_dataset=data_dir / "toy_eval.jsonl",
        model_out=model_path,
        report_dir=report_dir,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
