"""
Two-Stage Training Pipeline for BTC Price Prediction
====================================================
Stage 1: Trade Detection (trade vs no-trade)
Stage 2: Direction Prediction (long vs short) + magnitude regression

Both stages use the same CNN+LSTM architecture but different labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from model import CNNLSTMModel, count_parameters
from preprocesing import preprocess_data


# ═════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════

class ModelConfig:
    """Model architecture configuration."""

    # CNN
    cnn_channels = [64, 128, 256]
    cnn_kernel_size = 3
    cnn_dropout = 0.2

    # LSTM
    lstm_hidden_size = 256
    lstm_num_layers = 2
    lstm_dropout = 0.3

    # Output
    num_classes = 2  # binary classification


class TrainingConfig:
    """Training hyperparameters and settings."""

    # Training
    num_epochs = 100
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-5
    gradient_clip = 1.0

    # Data
    sequence_length = 60
    train_pct = 0.70
    val_pct = 0.15
    num_workers = 10

    # Optimizer
    optimizer_type = "AdamW"

    # Scheduler
    scheduler_patience = 7
    scheduler_factor = 0.5

    # Early stopping
    early_stop_patience = 15
    early_stop_min_delta = 1e-4

    # Logging
    log_interval = 50
    save_interval = 5

    # Regression loss weight (if using regression head)
    regression_loss_weight = 0.3  # classification loss weight will be 1.0

    def __init__(self, stage_name: str):
        """
        Args:
            stage_name: "trade" or "direction"
        """
        self.stage_name = stage_name

        # Paths
        self.output_dir = Path(f"../models/{stage_name}")
        self.log_dir = Path(f"../logs/{stage_name}")
        self.checkpoint_dir = self.output_dir / "checkpoints"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = self._get_device()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def to_dict(self):
        return {
            "stage": self.stage_name,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "device": str(self.device),
        }

    def print_config(self):
        print("\n" + "═" * 70)
        print(f"TRAINING CONFIGURATION - {self.stage_name.upper()} MODEL")
        print("═" * 70)
        print(f"  Device          : {self.device}")
        print(f"  Epochs          : {self.num_epochs}")
        print(f"  Batch Size      : {self.batch_size}")
        print(f"  Learning Rate   : {self.learning_rate}")
        print(f"  Sequence Length : {self.sequence_length}")
        print(f"  Early Stop      : {self.early_stop_patience} epochs")
        print("═" * 70 + "\n")


# ═════════════════════════════════════════════
#  EARLY STOPPING
# ═════════════════════════════════════════════

class EarlyStopping:
    """Stops training when validation loss plateaus."""

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
            return False

        if val_loss < (self.best_score - self.min_delta):
            self.best_score = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ═════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════

def compute_metrics(predictions, targets, returns=None):
    """
    Compute accuracy and optional P&L metrics.

    Args:
        predictions: (N,) predicted class (0 or 1)
        targets: (N,) true class (0 or 1)
        returns: (N,) trade returns (optional)
    """
    accuracy = (predictions == targets).mean()

    # Per-class accuracy
    class_0_mask = targets == 0
    class_1_mask = targets == 1

    class_0_acc = (predictions[class_0_mask] == targets[class_0_mask]).mean() if class_0_mask.sum() > 0 else 0
    class_1_acc = (predictions[class_1_mask] == targets[class_1_mask]).mean() if class_1_mask.sum() > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "class_0_acc": class_0_acc,
        "class_1_acc": class_1_acc,
    }

    # P&L (if returns provided)
    if returns is not None:
        correct_mask = predictions == targets
        pnl = returns[correct_mask].sum()

        metrics["total_pnl"] = pnl
        metrics["avg_pnl"] = returns[correct_mask].mean() if correct_mask.sum() > 0 else 0
        metrics["win_rate"] = correct_mask.mean()

    return metrics


# ═════════════════════════════════════════════
#  TRAINING & VALIDATION
# ═════════════════════════════════════════════

def train_epoch(model, loader, criterion_cls, criterion_reg, optimizer, device, config, epoch, use_regression):
    """Single training epoch."""
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", ncols=100)

    for batch_idx, (X, y, meta) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits, magnitude = model(X)

        # Classification loss
        cls_loss = criterion_cls(logits, y)
        loss = cls_loss

        # Regression loss (if enabled)
        if use_regression and magnitude is not None:
            # Target: absolute value of trade return (how big was the move?)
            returns_tensor = torch.FloatTensor([float(r) for r in meta["return"]]).to(device)
            target_magnitude = torch.abs(returns_tensor).unsqueeze(1)  # (batch, 1)

            reg_loss = criterion_reg(magnitude, target_magnitude)
            loss = cls_loss + config.regression_loss_weight * reg_loss

            running_reg_loss += reg_loss.item()

        loss.backward()

        if config.gradient_clip > 0:
            clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)
    avg_cls_loss = running_cls_loss / len(loader)

    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    metrics["loss"] = avg_loss
    metrics["cls_loss"] = avg_cls_loss

    if use_regression:
        metrics["reg_loss"] = running_reg_loss / len(loader)

    return metrics


def validate_epoch(model, loader, criterion_cls, criterion_reg, device, epoch, use_regression):
    """Single validation epoch."""
    model.eval()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    all_preds = []
    all_targets = []
    all_returns = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Val]  ", ncols=100)

    with torch.no_grad():
        for X, y, meta in pbar:
            X, y = X.to(device), y.to(device)

            logits, magnitude = model(X)

            # Classification loss
            cls_loss = criterion_cls(logits, y)
            loss = cls_loss

            # Regression loss (if enabled)
            if use_regression and magnitude is not None:
                returns_tensor = torch.FloatTensor([float(r) for r in meta["return"]]).to(device)
                target_magnitude = torch.abs(returns_tensor).unsqueeze(1)

                reg_loss = criterion_reg(magnitude, target_magnitude)
                loss = cls_loss + 0.3 * reg_loss

                running_reg_loss += reg_loss.item()

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
            all_returns.extend([float(r) for r in meta["return"]])

    avg_loss = running_loss / len(loader)
    avg_cls_loss = running_cls_loss / len(loader)

    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_returns)
    )
    metrics["loss"] = avg_loss
    metrics["cls_loss"] = avg_cls_loss

    if use_regression:
        metrics["reg_loss"] = running_reg_loss / len(loader)

    return metrics


# ═════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═════════════════════════════════════════════

def train_stage(
    stage_name: str,
    labels_file: str,
    train_config: TrainingConfig,
    model_config: ModelConfig,
    use_regression_head: bool = False
):
    """
    Train one stage (either trade detection or direction prediction).

    Args:
        stage_name: "trade" or "direction"
        labels_file: "labels_trade_1h.parquet" or "labels_direction_1h.parquet"
        train_config: TrainingConfig instance
        model_config: ModelConfig instance
        use_regression_head: whether to predict magnitude alongside direction

    Returns:
        model, history, test_metrics
    """

    train_config.print_config()

    # ── Load Data ──────────────────────────────────────────────────────────
    print(f"[1/5] Loading data for {stage_name} stage...")
    data = preprocess_data(
        features_file_name="features_1h.parquet",
        labels_file_name=labels_file,
        sequence_length=train_config.sequence_length,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_weights = data["class_weights"].to(train_config.device)

    print(f"  Features      : {data['num_features']}")
    print(f"  Train samples : {data['metadata']['train_size']:,}")
    print(f"  Val samples   : {data['metadata']['val_size']:,}")
    print(f"  Test samples  : {data['metadata']['test_size']:,}")
    print(f"  Regression    : {use_regression_head}")

    # ── Create Model ───────────────────────────────────────────────────────
    print(f"\n[2/5] Creating {stage_name} model...")
    model = CNNLSTMModel(
        input_features=data["num_features"],
        cnn_channels=model_config.cnn_channels,
        cnn_kernel_size=model_config.cnn_kernel_size,
        cnn_dropout=model_config.cnn_dropout,
        lstm_hidden_size=model_config.lstm_hidden_size,
        lstm_num_layers=model_config.lstm_num_layers,
        lstm_dropout=model_config.lstm_dropout,
        num_classes=model_config.num_classes,
        use_regression_head=use_regression_head,
    )
    model = model.to(train_config.device)

    print(f"  Parameters : {count_parameters(model):,}")

    # ── Training Components ────────────────────────────────────────────────
    print(f"\n[3/5] Setting up training components...")

    # Classification loss
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights: {class_weights}")

    # Regression loss (MSE for magnitude prediction)
    criterion_reg = nn.MSELoss() if use_regression_head else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
    )

    early_stopping = EarlyStopping(
        patience=train_config.early_stop_patience,
        min_delta=train_config.early_stop_min_delta,
    )

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(train_config.log_dir / f"run_{timestamp}")

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\n[4/5] Training {stage_name} model...")
    print("=" * 70)

    best_val_loss = float('inf')
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_pnl": [],
    }

    for epoch in range(1, train_config.num_epochs + 1):

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer,
            train_config.device, train_config, epoch, use_regression_head
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion_cls, criterion_reg,
            train_config.device, epoch, use_regression_head
        )

        # Scheduler
        scheduler.step(val_metrics["loss"])

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_pnl"].append(val_metrics.get("total_pnl", 0))

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

        if use_regression_head:
            writer.add_scalar("Loss/train_reg", train_metrics["reg_loss"], epoch)
            writer.add_scalar("Loss/val_reg", val_metrics["reg_loss"], epoch)

        # Print
        print(f"\nEpoch {epoch}/{train_config.num_epochs}")
        print(f"  Train → Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   → Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Class Acc → 0: {val_metrics['class_0_acc']:.3f}  1: {val_metrics['class_1_acc']:.3f}")

        if "total_pnl" in val_metrics:
            print(f"  Val PnL: {val_metrics['total_pnl']:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "config": train_config.to_dict(),
        }

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(checkpoint, train_config.output_dir / "best_model.pt")
            print(f"  ✓ Saved best model")

        # Periodic checkpoint
        if epoch % train_config.save_interval == 0:
            torch.save(checkpoint, train_config.checkpoint_dir / f"epoch_{epoch}.pt")

        # Early stopping
        if early_stopping(val_metrics["loss"], epoch):
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

    # ── Test Evaluation ────────────────────────────────────────────────────
    print(f"\n[5/5] Final evaluation on test set...")

    checkpoint = torch.load(train_config.output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate_epoch(
        model, test_loader, criterion_cls, criterion_reg,
        train_config.device, -1, use_regression_head
    )

    print("\n" + "=" * 70)
    print(f"TEST RESULTS - {stage_name.upper()} MODEL")
    print("=" * 70)
    print(f"  Accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"  Loss         : {test_metrics['loss']:.4f}")
    print(f"  Class 0 Acc  : {test_metrics['class_0_acc']:.3f}")
    print(f"  Class 1 Acc  : {test_metrics['class_1_acc']:.3f}")

    if "total_pnl" in test_metrics:
        print(f"  Total PnL    : {test_metrics['total_pnl']:.4f}")
        print(f"  Win Rate     : {test_metrics['win_rate']:.2%}")

    print("=" * 70 + "\n")

    # Save history
    with open(train_config.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()

    return model, history, test_metrics


# ═════════════════════════════════════════════
#  ENTRYPOINT
# ═════════════════════════════════════════════

if __name__ == "__main__":

    # Shared model config
    model_config = ModelConfig()

    print("\n" + "█" * 70)
    print("TWO-STAGE TRAINING PIPELINE")
    print("█" * 70)

    # ═══ STAGE 1: TRADE DETECTION ═══
    print("\n" + "▓" * 70)
    print("STAGE 1: TRADE DETECTION (trade vs no-trade)")
    print("▓" * 70)

    trade_config = TrainingConfig(stage_name="trade")
    model_trade, history_trade, test_trade = train_stage(
        stage_name="trade",
        labels_file="labels_trade_1h.parquet",
        train_config=trade_config,
        model_config=model_config,
        use_regression_head=False  # No regression for trade detection
    )

    # ═══ STAGE 2: DIRECTION PREDICTION ═══
    print("\n" + "▓" * 70)
    print("STAGE 2: DIRECTION PREDICTION (long vs short + magnitude)")
    print("▓" * 70)

    direction_config = TrainingConfig(stage_name="direction")
    model_direction, history_direction, test_direction = train_stage(
        stage_name="direction",
        labels_file="labels_direction_1h.parquet",
        train_config=direction_config,
        model_config=model_config,
        use_regression_head=True  # Predict magnitude for direction model
    )

    # ═══ SUMMARY ═══
    print("\n" + "█" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("█" * 70)
    print(f"\nTrade Model (Stage 1):")
    print(f"  Test Accuracy : {test_trade['accuracy']:.4f}")
    print(f"  Saved to      : {trade_config.output_dir / 'best_model.pt'}")

    print(f"\nDirection Model (Stage 2):")
    print(f"  Test Accuracy : {test_direction['accuracy']:.4f}")
    print(f"  Saved to      : {direction_config.output_dir / 'best_model.pt'}")

    print("\n" + "█" * 70 + "\n")