# Persian Sign Language Translator

A practical starter system for translating Persian Sign Language (PSL) gestures into Persian text, with a reverse text-to-sign gloss mode for the first 50 high-frequency words.

I built this repo around the boring parts that usually get ignored: dataset format, landmark normalization, repeatable training commands, tests, and packaging. The model can only become serious after the dataset becomes serious. Validation score without signer diversity is just decoration.

## Why this project exists

Iran has hundreds of thousands of deaf and hard-of-hearing people, but Persian Sign Language tooling is still thin. Live interpreters are expensive, public PSL datasets are almost nonexistent, and most academic work is scattered in papers instead of usable code.

So the first win is not only the model. The first win is a clean pipeline for collecting PSL samples and publishing a dataset other people can actually train on.

## What is inside now

- Webcam sample collection with **OpenCV + MediaPipe Hands**
- Two-hand landmark normalization: `21 landmarks × 3 coordinates × 2 hands = 126 features/frame`
- Temporal resampling to fixed-length sequences
- Runnable baseline classifier so the pipeline can be tested before the real dataset is ready
- Hooks for GRU/LSTM training with PyTorch
- Persian text-to-sign gloss conversion for 50 target words
- CLI commands for data, training, prediction, and camera mode
- Unit tests
- PyInstaller build script and GitHub Actions workflow for a Windows `.exe`

## Current honesty check

This repo does **not** ship a real public PSL dataset yet. The included toy data is synthetic and only exists to test the code path. Do not report toy-data accuracy as model accuracy. That would be nonsense, and honestly a bit shameful.

Target for the first real release:

- 50 common PSL words
- 10 different signers
- signer-wise train/test split
- over 90% top-1 accuracy on held-out signers for the core vocabulary
- public curated landmark dataset

## Install

```bash
git clone https://github.com/Captain-Jorf/Persian-Sign-Language-Translator.git
cd Persian-Sign-Language-Translator
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For GRU/LSTM experiments:

```bash
pip install -r requirements-ml.txt
```

For tests and executable builds:

```bash
pip install -r requirements-dev.txt
```

## Quick smoke test

This uses fake landmark data. It proves the pipeline runs, nothing more.

```bash
python -m psl_translator make-toy-data --out data/demo/toy_samples.jsonl
python -m psl_translator stats data/demo/toy_samples.jsonl
python -m psl_translator train --dataset data/demo/toy_samples.jsonl --model-out models/prototype_model.json
python -m unittest discover -s tests
```

Text-to-sign gloss mode:

```bash
python -m psl_translator text-to-sign "سلام لطفا آب"
```

Example output:

```json
[
  {
    "word": "سلام",
    "gloss": "PSL_001_SIGN_001",
    "asset_hint": "assets/signs/PSL_001_SIGN_001.mp4",
    "known": true
  }
]
```

## Collect PSL samples from webcam

```bash
python -m psl_translator collect \
  --label "سلام" \
  --signer-id s01 \
  --seconds 3 \
  --out data/raw/psl_samples.jsonl
```

Press `q` to stop early.

Then train the baseline:

```bash
python -m psl_translator train \
  --dataset data/raw/psl_samples.jsonl \
  --model-out models/prototype_model.json
```

Run real-time translation:

```bash
python -m psl_translator camera --model models/prototype_model.json
```

## Dataset format

Each line is one labeled sequence:

```json
{
  "label": "سلام",
  "signer_id": "s01",
  "source": "webcam",
  "frames": [[0.0, 0.1, 0.2]],
  "meta": {"camera": "laptop"}
}
```

A frame must contain 126 values. Missing hands are zero-filled. Raw videos should stay out of git unless they are tiny curated examples; they get heavy fast.

## Build executable

### Local build

```bash
pip install -r requirements-dev.txt
python scripts/build_exe.py
```

On Windows this creates:

```text
release/PersianSignLanguageTranslator.exe
```

On Linux/macOS it creates a native binary for that OS instead. PyInstaller does not cross-compile a real Windows executable from Linux. I know, annoying, but that is the toolchain reality.

### GitHub Actions Windows build

A ready workflow template is in `docs/workflows/build-windows-exe.yml`. Copy it to `.github/workflows/build-windows-exe.yml` in GitHub, run the workflow, and download the artifact named `PersianSignLanguageTranslator-windows-exe`.

## Commands

```bash
python -m psl_translator --help
python -m psl_translator words
python -m psl_translator text-to-sign "سلام دکتر کمک"
python -m psl_translator make-toy-data --out data/demo/toy_samples.jsonl
python -m psl_translator stats data/demo/toy_samples.jsonl
python -m psl_translator train --dataset data/demo/toy_samples.jsonl --model-out models/prototype_model.json
python -m psl_translator predict-json --model models/prototype_model.json --frames-json sample_frames.json
python -m psl_translator collect --label "آب" --signer-id s02
python -m psl_translator camera --model models/prototype_model.json
```

## Tech stack

Python, OpenCV, MediaPipe Hands, sequence preprocessing, GRU/LSTM hooks with PyTorch, Persian NLP normalization, dataset curation, model evaluation discipline.

## Project status

Early but usable. The scaffolding is clean enough to build on. The painful part is next: recording real PSL data with enough signer variation, bad lighting, camera angles, and all the ugly little things that make a model survive outside a notebook.

Like Saadi said in a different context, the road matters. Here the road is the dataset.
