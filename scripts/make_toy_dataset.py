from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psl_translator.dataset import write_jsonl
from psl_translator.toy_data import make_toy_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo/toy_samples.jsonl")
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--frames", type=int, default=18)
    args = parser.parse_args()

    samples = make_toy_samples(per_label=args.per_label, frames=args.frames)
    write_jsonl(samples, args.out)
    print(f"wrote {len(samples)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
