"""
Training Pipeline for BTC Price Prediction
===========================================
Trains the CNN+LSTM model with proper logging, checkpointing,
early stopping, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Assuming you have these from previous scripts
from model import create_model, count_parameters
from preprocesing import preprocess_data


# ─────────────────────────────────────────────
#  TRAINING CONFIGURATION
# ─────────────────────────────────────────────

class TrainingConfig:
    """Centralized training configuration."""

    def __init__(self):
        # Paths
        self.output_dir = Path("../models")
        self.log_dir = Path("../logs")
        self.checkpoint_dir = self.output_dir / "checkpoints"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model config
        self.model_config = "default"  # "light", "default", "heavy", "with_regression"

        # Training hyperparameters
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 256

        # Optimizer & scheduler
        self.optimizer_type = "Adam"  # "Adam" or "AdamW"
        self.scheduler_type = "ReduceLROnPlateau"  # "ReduceLROnPlateau" or "CosineAnnealing"
        self.scheduler_patience = 7
        self.scheduler_factor = 0.5

        # Early stopping
        self.early_stop_patience = 15
        self.early_stop_min_delta = 1e-4

        # Device
        self.device = self._get_device()

        # Gradient clipping
        self.gradient_clip = 1.0

        # Logging
        self.log_interval = 50  # log every N batches
        self.save_interval = 5  # save checkpoint every N epochs

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "model_config": self.model_config,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "optimizer_type": self.optimizer_type,
            "scheduler_type": self.scheduler_type,
            "device": str(self.device),
        }


# ─────────────────────────────────────────────
#  EARLY STOPPING
# ─────────────────────────────────────────────

class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ─────────────────────────────────────────────
#  METRICS COMPUTATION
# ─────────────────────────────────────────────

def compute_metrics(predictions, targets, returns=None):
    """
    Compute classification metrics and trading P&L.

    Args:
        predictions: (N,) array of predicted classes (0, 1, 2)
        targets: (N,) array of true classes (0, 1, 2)
        returns: (N,) array of log returns (optional, for P&L calculation)

    Returns:
        dict of metrics
    """
    # Basic accuracy
    accuracy = (predictions == targets).mean()

    # Per-class accuracy
    class_acc = {}
    for cls in [0, 1, 2]:
        mask = targets == cls
        if mask.sum() > 0:
            class_acc[cls] = (predictions[mask] == targets[mask]).mean()
        else:
            class_acc[cls] = 0.0

    # Trading P&L (if returns provided)
    pnl_metrics = {}
    if returns is not None:
        # Map predictions to trade directions: 0→short, 1→skip, 2→long
        trade_returns = np.zeros_like(returns)

        # Long trades (predicted class 2)
        long_mask = predictions == 2
        trade_returns[long_mask] = returns[long_mask]

        # Short trades (predicted class 0) — flip the sign
        short_mask = predictions == 0
        trade_returns[short_mask] = -returns[short_mask]

        # Neutral trades (predicted class 1) — no position, 0 return
        # already zero

        # Calculate P&L metrics
        total_trades = (predictions != 1).sum()  # exclude neutral
        winning_trades = (trade_returns > 0).sum()
        losing_trades = (trade_returns < 0).sum()

        pnl_metrics = {
            "total_pnl": trade_returns.sum(),
            "avg_pnl": trade_returns.mean(),
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "avg_win": trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0,
            "avg_loss": trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else 0,
        }

        # Profit factor
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        pnl_metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        "accuracy": accuracy,
        "class_accuracy": class_acc,
        **pnl_metrics,
    }


# ─────────────────────────────────────────────
#  TRAINING EPOCH
# ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, config, epoch):
    """Single training epoch."""
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (X, y, meta) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        # Forward
        optimizer.zero_grad()
        logits, magnitude = model(X)

        # Loss (classification only for now)
        loss = criterion(logits, y)

        # Backward
        loss.backward()

        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        # Metrics
        running_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Epoch metrics
    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    metrics["loss"] = avg_loss

    return metrics


# ─────────────────────────────────────────────
#  VALIDATION EPOCH
# ─────────────────────────────────────────────

def validate_epoch(model, loader, criterion, device, epoch):
    """Single validation epoch."""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_returns = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ")

    with torch.no_grad():
        for X, y, meta in pbar:
            X, y = X.to(device), y.to(device)

            logits, magnitude = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
            all_returns.extend([float(r) for r in meta["return"]])

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_returns)
    )
    metrics["loss"] = avg_loss

    return metrics


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train_model(config: TrainingConfig):
    """Main training loop."""

    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_config}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    data = preprocess_data(
        sequence_length=60,
        batch_size=config.batch_size,
        num_workers=0,  # set to 4-8 on Linux with good CPU
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_weights = data["class_weights"].to(config.device)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")

    # ── Create model ───────────────────────────────────────────────────────
    print("\n[2/5] Creating model...")
    model = create_model(
        config=config.model_config,
        input_features=data["num_features"],
        sequence_length=data["sequence_length"],
        num_classes=3,
    )
    model = model.to(config.device)

    print(f"  Parameters: {count_parameters(model):,}")

    # ── Loss, optimizer, scheduler ─────────────────────────────────────────
    print("\n[3/5] Setting up training components...")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if config.optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    elif config.scheduler_type == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
        )

    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
    )

    # ── TensorBoard ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(config.log_dir / f"run_{timestamp}")

    # ── Training loop ──────────────────────────────────────────────────────
    print("\n[4/5] Starting training...")
    print("=" * 70)

    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_pnl": [],
    }

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, config.device, config, epoch
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, config.device, epoch
        )

        # Scheduler step
        if config.scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step()

        # Log metrics
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_pnl"].append(val_metrics.get("total_pnl", 0))

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        if "total_pnl" in val_metrics:
            writer.add_scalar("PnL/val", val_metrics["total_pnl"], epoch)
            writer.add_scalar("WinRate/val", val_metrics["win_rate"], epoch)

        # Print summary
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}  |  Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f}  |  Acc: {val_metrics['accuracy']:.4f}")

        if "total_pnl" in val_metrics:
            print(
                f"Val PnL: {val_metrics['total_pnl']:.4f}  |  Win Rate: {val_metrics['win_rate']:.2%}  |  Profit Factor: {val_metrics['profit_factor']:.2f}")

        print(
            f"Class Acc - Short: {val_metrics['class_accuracy'][0]:.3f}  Neutral: {val_metrics['class_accuracy'][1]:.3f}  Long: {val_metrics['class_accuracy'][2]:.3f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "config": config.to_dict(),
                "feature_cols": data["feature_cols"],
            }

            torch.save(checkpoint, config.output_dir / "best_model.pt")
            print(f"✓ Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % config.save_interval == 0:
            torch.save(checkpoint, config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

        # Early stopping
        if early_stopping(val_metrics["loss"], epoch):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            print(f"   Best epoch was {early_stopping.best_epoch} with val_loss={early_stopping.best_score:.4f}")
            break

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n[5/5] Final evaluation on test set...")

    # Load best model
    checkpoint = torch.load(config.output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate_epoch(model, test_loader, criterion, config.device, epoch=-1)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"Loss         : {test_metrics['loss']:.4f}")
    if "total_pnl" in test_metrics:
        print(f"Total PnL    : {test_metrics['total_pnl']:.4f}")
        print(f"Avg PnL/trade: {test_metrics['avg_pnl']:.6f}")
        print(f"Win Rate     : {test_metrics['win_rate']:.2%}")
        print(f"Profit Factor: {test_metrics['profit_factor']:.2f}")
        print(f"Total Trades : {test_metrics['total_trades']}")

    # Save history
    history_path = config.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete! Best model saved to {config.output_dir / 'best_model.pt'}")

    writer.close()

    return model, history, test_metrics


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

def plot_training_history(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Val")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Val PnL
    axes[1, 0].plot(history["val_pnl"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cumulative PnL")
    axes[1, 0].set_title("Validation PnL")
    axes[1, 0].grid(alpha=0.3)

    # Hide empty subplot
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved plot to {save_path}")

    plt.show()


# ─────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = TrainingConfig()

    # Train
    model, history, test_metrics = train_model(config)

    # Plot
    plot_training_history(history, save_path=config.output_dir / "training_curves.png")