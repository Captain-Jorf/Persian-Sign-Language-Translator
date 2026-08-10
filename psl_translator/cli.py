from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import __version__
from .camera import collect_samples, run_realtime_translation
from .dataset import dataset_stats, load_jsonl, write_jsonl
from .dictionary import known_words, sentence_to_sign_tokens
from .islr101 import convert_manifest_to_jsonl, facts as islr101_facts
from .metrics import evaluate_model
from .model import PrototypeSequenceClassifier
from .reporting import write_evaluation_report
from .toy_data import make_toy_samples
from .trainer import train_evaluate_report, train_prototype_model


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_text_to_sign(args: argparse.Namespace) -> int:
    tokens = sentence_to_sign_tokens(args.text)
    _print_json([asdict(token) for token in tokens])
    return 0


def cmd_words(_: argparse.Namespace) -> int:
    _print_json(known_words())
    return 0


def cmd_make_toy_data(args: argparse.Namespace) -> int:
    samples = make_toy_samples(per_label=args.per_label, frames=args.frames)
    write_jsonl(samples, args.out)
    print(f"wrote {len(samples)} synthetic samples to {args.out}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.dataset)
    stats = dataset_stats(samples)
    _print_json(
        {
            "samples": stats.sample_count,
            "label_count": stats.label_count,
            "labels": stats.label_counts,
            "signers": stats.signer_counts,
        }
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    report = train_prototype_model(args.dataset, args.model_out)
    _print_json(report)
    return 0


def cmd_train_report(args: argparse.Namespace) -> int:
    report = train_evaluate_report(args.train_dataset, args.eval_dataset, args.model_out, args.report_dir)
    _print_json(report)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = PrototypeSequenceClassifier.load(args.model)
    samples = load_jsonl(args.dataset)
    result, _ = evaluate_model(model, samples)
    paths = write_evaluation_report(result, args.report_dir, title=args.title)
    _print_json({"accuracy": result.accuracy, "sample_count": result.sample_count, "report": paths})
    return 0


def cmd_islr101_info(_: argparse.Namespace) -> int:
    _print_json(islr101_facts())
    return 0


def cmd_convert_islr101(args: argparse.Namespace) -> int:
    count = convert_manifest_to_jsonl(args.manifest, args.root, args.out, split=args.split)
    _print_json({"converted_samples": count, "out": args.out})
    return 0


def cmd_predict_json(args: argparse.Namespace) -> int:
    model = PrototypeSequenceClassifier.load(args.model)
    payload = json.loads(Path(args.frames_json).read_text(encoding="utf-8"))
    frames = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    prediction = model.predict(frames)
    _print_json(
        {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "scores": prediction.scores,
        }
    )
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    frame_count = collect_samples(
        label=args.label,
        out_path=args.out,
        signer_id=args.signer_id,
        seconds=args.seconds,
        camera_index=args.camera,
    )
    print(f"saved {frame_count} frames for {args.label} to {args.out}")
    return 0


def cmd_camera(args: argparse.Namespace) -> int:
    run_realtime_translation(model_path=args.model, camera_index=args.camera)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psl-translator",
        description="Persian Sign Language data collection, training, and translation CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_to_sign = subparsers.add_parser("text-to-sign", help="convert Persian text to known PSL gloss tokens")
    text_to_sign.add_argument("text")
    text_to_sign.set_defaults(func=cmd_text_to_sign)

    words = subparsers.add_parser("words", help="list the first target vocabulary")
    words.set_defaults(func=cmd_words)

    toy = subparsers.add_parser("make-toy-data", help="create deterministic synthetic samples for smoke tests")
    toy.add_argument("--out", default="data/demo/toy_samples.jsonl")
    toy.add_argument("--per-label", type=int, default=6)
    toy.add_argument("--frames", type=int, default=18)
    toy.set_defaults(func=cmd_make_toy_data)

    stats = subparsers.add_parser("stats", help="show dataset label/signer counts")
    stats.add_argument("dataset")
    stats.set_defaults(func=cmd_stats)

    train = subparsers.add_parser("train", help="train the runnable prototype classifier")
    train.add_argument("--dataset", required=True)
    train.add_argument("--model-out", default="models/prototype_model.json")
    train.set_defaults(func=cmd_train)

    train_report = subparsers.add_parser("train-report", help="train, evaluate, and write visual report files")
    train_report.add_argument("--train-dataset", required=True)
    train_report.add_argument("--eval-dataset", required=True)
    train_report.add_argument("--model-out", default="models/prototype_model.json")
    train_report.add_argument("--report-dir", default="reports/latest")
    train_report.set_defaults(func=cmd_train_report)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved model and write SVG/HTML report")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--report-dir", default="reports/latest")
    evaluate.add_argument("--title", default="Persian Sign Language Model Report")
    evaluate.set_defaults(func=cmd_evaluate)

    islr_info = subparsers.add_parser("islr101-info", help="show ISLR101 dataset facts from arXiv:2503.12451")
    islr_info.set_defaults(func=cmd_islr101_info)

    convert_islr = subparsers.add_parser("convert-islr101", help="convert ISLR101 OpenPose skeleton files to JSONL")
    convert_islr.add_argument("--manifest", required=True)
    convert_islr.add_argument("--root", required=True)
    convert_islr.add_argument("--out", required=True)
    convert_islr.add_argument("--split")
    convert_islr.set_defaults(func=cmd_convert_islr101)

    predict = subparsers.add_parser("predict-json", help="predict from a JSON file containing frame vectors")
    predict.add_argument("--model", required=True)
    predict.add_argument("--frames-json", required=True)
    predict.set_defaults(func=cmd_predict_json)

    collect = subparsers.add_parser("collect", help="record one labeled webcam sample")
    collect.add_argument("--label", required=True)
    collect.add_argument("--out", default="data/raw/psl_samples.jsonl")
    collect.add_argument("--signer-id")
    collect.add_argument("--seconds", type=float, default=3.0)
    collect.add_argument("--camera", type=int, default=0)
    collect.set_defaults(func=cmd_collect)

    camera = subparsers.add_parser("camera", help="run webcam translation")
    camera.add_argument("--model", required=True)
    camera.add_argument("--camera", type=int, default=0)
    camera.set_defaults(func=cmd_camera)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
