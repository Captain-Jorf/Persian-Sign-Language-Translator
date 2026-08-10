# Architecture Notes

This project is split into boring pieces on purpose.

```text
webcam/video
   -> MediaPipe Hands landmarks
   -> normalization (wrist origin, scale cleanup)
   -> temporal resampling (30 frames)
   -> classifier (prototype now, GRU/LSTM later)
   -> smoother
   -> Persian label
```

Reverse mode is separate:

```text
Persian sentence -> normalization/tokenization -> PSL gloss list -> sign assets
```

## Why the prototype classifier exists

A real PSL model needs signer diversity. Training a GRU on three friends in one
room gives a nice validation score and then fails on the first new person. Seen
that movie too many times. The prototype classifier is not pretending to be the
research result; it keeps data collection, CI, packaging, and the CLI executable
working while the dataset grows.

## Dataset format

JSONL, one sample per line:

```json
{
  "label": "سلام",
  "signer_id": "s01",
  "source": "webcam",
  "frames": [[0.0, 0.1, 0.2]],
  "meta": {"camera": "laptop"}
}
```

Each frame is a 126-value vector: 21 landmarks * 3 coordinates * 2 hands.
Missing hands are zero-filled. Raw videos should stay outside git; landmarks are
small enough to review and version when they are curated.

## Next engineering steps

1. Record 50 high-frequency words with at least 10 signers.
2. Split by signer, not random frame/sample. Random split lies.
3. Train GRU/LSTM with augmentation: horizontal jitter, temporal crop, speed
   perturbation, and small landmark noise.
4. Add face-landmark features only after hand-only baseline is measured. More
   features first, discipline later, and suddenly debugging becomes a swamp.
