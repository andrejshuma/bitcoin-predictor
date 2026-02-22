import torch
import torch.nn as nn
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
        → Permute back to (batch, seq_len, cnn_features) for LSTM
        → LSTM temporal modeling
        → Output heads (classification + optional regression)
    """

    def __init__(
            self,
            input_features: int = 161,

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

        self.use_regression_head = use_regression_head

        # ── CNN Feature Extractor ──────────────────────────────────────────
        # Operates on features to extract local patterns
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
        cnn_out_channels = cnn_channels[-1]

        # ── LSTM Temporal Modeling ─────────────────────────────────────────
        # Operates on time stamps to capture how patterns evolve over time

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,  # only look at past
        )

        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # ── Classification Head ────────────────────────────────────────────
        # Predicts trade/no-trade or long/short signal

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
        # x = (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features/cnn_channels, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features/cnn_channels)

        # ── LSTM Temporal Modeling ─────────────────────────────────────────
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: This is the LSTM's output at each of the seq_len (60) timesteps
        # lstm_out: (batch, seq_len, hidden_size)
        # `lstm_out[i, t, :]` = the hidden state at timestep `t` for sample `i`

        # h_n: This is the final state of each LSTM layer after processing all seq_len (60) timesteps
        # h_n: (num_layers, batch, hidden_size)

        # Take the last hidden state from the last LSTM layer
        last_hidden = h_n[-1]  # or lstm_out[:, -1, :]. (batch, hidden_size)
        last_hidden = self.lstm_dropout(last_hidden)

        # ── Output Heads ───────────────────────────────────────────────────
        logits = self.fc_classifier(last_hidden)  # (batch, num_classes)

        magnitude = None
        if self.use_regression_head:
            magnitude = self.fc_regression(last_hidden)  # (batch, 1)

        return logits, magnitude



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
    input_features = 161
    seq_len = 30
    batch_size = 32

    model = CNNLSTMModel(
        input_features=input_features,
        cnn_channels=[64, 128, 256],
        cnn_kernel_size=3,
        cnn_dropout=0.2,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        num_classes=2,
        use_regression_head=False
    )

    print_model_summary(model, input_shape=(batch_size, seq_len, input_features))