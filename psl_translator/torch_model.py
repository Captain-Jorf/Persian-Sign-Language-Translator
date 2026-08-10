from __future__ import annotations

"""Optional GRU/LSTM model definitions.

This file imports torch lazily inside the factory so normal CLI tasks and tests do
not need a 700 MB dependency. For real training, install requirements-ml.txt.
"""


def build_sequence_gru(input_size: int, hidden_size: int, num_classes: int, layers: int = 2):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local machine
        raise RuntimeError(
            "PyTorch is not installed. Install it with: pip install -r requirements-ml.txt"
        ) from exc

    class GRUClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layers,
                batch_first=True,
                dropout=0.2 if layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, num_classes),
            )

        def forward(self, x):
            _, hidden = self.gru(x)
            return self.head(hidden[-1])

    # Small sanity call so linters do not think torch is unused in generated docs.
    _ = torch.float32
    return GRUClassifier()
