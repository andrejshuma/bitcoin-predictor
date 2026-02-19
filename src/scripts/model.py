"""
Model Architecture for BTC Price Prediction
============================================
CNN+LSTM hybrid for multi-timeframe feature extraction and temporal modeling.

Architecture:
  1. CNN layers extract local patterns across features
  2. LSTM layers capture temporal dependencies
  3. Multi-head output: classification + auxiliary regression heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────
#  CNN + LSTM BACKBONE
# ─────────────────────────────────────────────

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM architecture for time series classification.

    Architecture flow:
      Input (batch, seq_len, features)
        → Permute to (batch, features, seq_len) for Conv1d
        → CNN feature extraction
        → Permute back to (batch, seq_len, cnn_features)
        → LSTM temporal modeling
        → Output heads (classification + optional regression)
    """

    def __init__(
            self,
            input_features: int = 161,
            sequence_length: int = 60,

            # CNN config
            cnn_channels: list = [64, 128, 256],
            cnn_kernel_size: int = 3,
            cnn_dropout: float = 0.2,

            # LSTM config
            lstm_hidden_size: int = 256,
            lstm_num_layers: int = 2,
            lstm_dropout: float = 0.3,

            # Output config
            num_classes: int = 3,  # -1, 0, 1 → 0, 1, 2
            use_regression_head: bool = False,
    ):
        super().__init__()

        self.input_features = input_features
        self.sequence_length = sequence_length
        self.use_regression_head = use_regression_head

        # ── CNN Feature Extractor ──────────────────────────────────────────
        # Operates on (batch, features, seq_len)
        # Extracts local patterns like "RSI crossed 30 while volume spiked"

        cnn_layers = []
        in_channels = input_features

        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2,  # 'same' padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(cnn_dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_channels[-1]

        # ── LSTM Temporal Modeling ─────────────────────────────────────────
        # Operates on (batch, seq_len, cnn_features)
        # Captures how patterns evolve over time

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,  # causal: only look at past
        )

        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # ── Classification Head ────────────────────────────────────────────
        # Predicts which direction price will move

        self.fc_classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # ── Regression Head (optional) ─────────────────────────────────────
        # Predicts magnitude of move (in ATR units)

        if use_regression_head:
            self.fc_regression = nn.Sequential(
                nn.Linear(lstm_hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),  # single value: expected move size
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, features)

        Returns:
            logits: (batch, num_classes) - class logits for CrossEntropyLoss
            magnitude: (batch, 1) - predicted move magnitude (if regression head enabled)
        """
        batch_size = x.size(0)

        # ── CNN Feature Extraction ─────────────────────────────────────────
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.cnn(x)  # (batch, cnn_channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, cnn_channels)

        # ── LSTM Temporal Modeling ─────────────────────────────────────────
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)

        # Take the last hidden state from the last LSTM layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        last_hidden = self.lstm_dropout(last_hidden)

        # ── Output Heads ───────────────────────────────────────────────────
        logits = self.fc_classifier(last_hidden)  # (batch, num_classes)

        magnitude = None
        if self.use_regression_head:
            magnitude = self.fc_regression(last_hidden)  # (batch, 1)

        return logits, magnitude


# ─────────────────────────────────────────────
#  LIGHTWEIGHT VARIANT (for faster iteration)
# ─────────────────────────────────────────────

class LightCNNLSTM(nn.Module):
    """
    Smaller, faster variant for quick experiments.
    ~1/4 the parameters of the full model.
    """

    def __init__(
            self,
            input_features: int = 170,
            sequence_length: int = 60,
            num_classes: int = 3,
    ):
        super().__init__()

        # Simplified CNN: single layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        # LSTM
        _, (h_n, _) = self.lstm(x)

        # Classifier
        logits = self.fc(h_n[-1])

        return logits, None  # no regression head


# ─────────────────────────────────────────────
#  MODEL FACTORY
# ─────────────────────────────────────────────

def create_model(
        config: str = "default",
        input_features: int = 161,
        sequence_length: int = 60,
        num_classes: int = 3,
) -> nn.Module:
    """
    Factory function to create different model configurations.

    Args:
        config: "default", "light", "heavy", or "with_regression"
        input_features: number of input features
        sequence_length: length of input sequences
        num_classes: number of output classes

    Returns:
        model: initialized PyTorch model
    """

    configs = {
        "light": {
            "class": LightCNNLSTM,
            "params": {
                "input_features": input_features,
                "sequence_length": sequence_length,
                "num_classes": num_classes,
            }
        },

        "default": {
            "class": CNNLSTMModel,
            "params": {
                "input_features": input_features,
                "sequence_length": sequence_length,
                "cnn_channels": [64, 128, 256],
                "cnn_kernel_size": 3,
                "cnn_dropout": 0.2,
                "lstm_hidden_size": 256,
                "lstm_num_layers": 2,
                "lstm_dropout": 0.3,
                "num_classes": num_classes,
                "use_regression_head": False,
            }
        },

        "heavy": {
            "class": CNNLSTMModel,
            "params": {
                "input_features": input_features,
                "sequence_length": sequence_length,
                "cnn_channels": [128, 256, 512],
                "cnn_kernel_size": 5,
                "cnn_dropout": 0.25,
                "lstm_hidden_size": 512,
                "lstm_num_layers": 3,
                "lstm_dropout": 0.35,
                "num_classes": num_classes,
                "use_regression_head": False,
            }
        },

        "with_regression": {
            "class": CNNLSTMModel,
            "params": {
                "input_features": input_features,
                "sequence_length": sequence_length,
                "cnn_channels": [64, 128, 256],
                "cnn_kernel_size": 3,
                "cnn_dropout": 0.2,
                "lstm_hidden_size": 256,
                "lstm_num_layers": 2,
                "lstm_dropout": 0.3,
                "num_classes": num_classes,
                "use_regression_head": True,  # adds magnitude prediction
            }
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys())}")

    model_class = configs[config]["class"]
    model_params = configs[config]["params"]

    model = model_class(**model_params)

    return model


# ─────────────────────────────────────────────
#  MODEL SUMMARY
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_shape: Tuple[int, int, int]):
    """
    Print model architecture and parameter count.

    Args:
        model: PyTorch model
        input_shape: (batch, seq_len, features)
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)

    total_params = count_parameters(model)
    print(f"  Total parameters      : {total_params:,}")
    print(f"  Model size (approx)   : {total_params * 4 / 1024 ** 2:.2f} MB")
    print(f"  Input shape           : {input_shape}")

    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)

    try:
        with torch.no_grad():
            logits, magnitude = model(dummy_input)
        print(f"  Output (logits) shape : {logits.shape}")
        if magnitude is not None:
            print(f"  Output (magnitude)    : {magnitude.shape}")
    except Exception as e:
        print(f"  ⚠️  Forward pass failed: {e}")


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test all model variants
    configs = ["light", "default", "heavy", "with_regression"]

    for config in configs:
        print(f"\n\n{'#' * 60}")
        print(f"Testing config: {config}")
        print(f"{'#' * 60}")

        model = create_model(
            config=config,
            input_features=170,
            sequence_length=60,
            num_classes=3,
        )

        print_model_summary(model, input_shape=(32, 60, 170))