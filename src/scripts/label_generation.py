"""
Triple Barrier Labeling for BTC Price Prediction
=================================================
For each bar, places three barriers:
  - Upper barrier  : entry + (tp_atr_mult × ATR)  → label  1  (long)
  - Lower barrier  : entry - (sl_atr_mult × ATR)  → label -1  (short)
  - Vertical barrier: entry + max_holding_bars     → label  0  (neutral/timeout)

Whichever barrier is hit first determines the label.

Output: ../data/processed/labels_1h.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit   # pip install numba — makes the loop ~100x faster

PROCESSED_DIR = Path("../../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  CORE BARRIER LOGIC
# ─────────────────────────────────────────────

@njit
def _triple_barrier_loop(
    close: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    atr:   np.ndarray,
    tp_mult:  float,
    sl_mult:  float,
    max_bars: int,
) -> tuple:
    """
    Pure numpy loop compiled by numba for speed.
    Iterates over every bar and looks forward to find which barrier is hit first.

    Returns arrays of:
      labels     : -1, 0, or 1
      hit_bars   : how many bars until the barrier was hit
      returns    : actual log return when barrier was hit
    """
    n      = len(close)
    labels   = np.zeros(n, dtype=np.int8)
    hit_bars = np.zeros(n, dtype=np.int32)
    returns  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entry     = close[i]
        tp_price  = entry + tp_mult * atr[i]   # upper barrier
        sl_price  = entry - sl_mult * atr[i]   # lower barrier
        end_bar   = min(i + max_bars, n - 1)   # vertical barrier

        label   = 0   # default: time barrier hit
        bar_hit = max_bars
        exit_price = close[min(i + max_bars, n - 1)]  # default: time barrier

        for j in range(i + 1, end_bar + 1):
            # Check high for TP hit, low for SL hit
            # Using high/low instead of close catches intrabar touches
            if high[j] >= tp_price:
                label   = 1
                bar_hit = j - i
                exit_price = tp_price
                break
            if low[j] <= sl_price:
                label   = -1
                bar_hit = j - i
                exit_price = sl_price
                break

        # Actual log return at exit bar
        ret            = np.log(exit_price / entry)

        labels[i]      = label
        hit_bars[i]    = bar_hit
        returns[i]     = ret

    return labels, hit_bars, returns


# ─────────────────────────────────────────────
#  LABEL BUILDER
# ─────────────────────────────────────────────

def build_labels(
    features_path: str | Path  = PROCESSED_DIR / "features_1h.parquet",
    atr_col: str               = "atr_14",       # which ATR to use for barrier sizing
    tp_mult: float             = 1.0,            # TP = entry + tp_mult × ATR
    sl_mult: float             = 1.0,            # SL = entry - sl_mult × ATR
    max_bars: int              = 48,             # vertical barrier = 48h max hold
    min_atr_pct: float         = 0.003,          # skip bars where ATR < 0.3% of price (dead market)
) -> pd.DataFrame:
    """
    Generates triple barrier labels for every bar in the features file.

    Parameters
    ----------
    features_path : Path to the features parquet (must contain OHLCV + ATR columns)
    atr_col       : Which ATR column to use for barrier distances
    tp_mult       : Take profit multiplier (TP distance = tp_mult × ATR)
    sl_mult       : Stop loss multiplier  (SL distance = sl_mult × ATR)
    max_bars      : Maximum bars to hold before forced exit (vertical barrier)
    min_atr_pct   : Minimum ATR as % of price — bars below this are labeled 0 (no trade)

    Returns
    -------
    DataFrame with columns:
        label        : -1 (short), 0 (neutral), 1 (long)
        hit_bars     : bars until barrier was hit
        log_return   : actual log return at exit
        tp_price     : where the TP barrier was placed
        sl_price     : where the SL barrier was placed
        atr_used     : ATR value used for this bar
        reward_risk  : tp_mult / sl_mult ratio (constant but useful to have)
    """
    print(f"Loading {features_path} ...")
    df = pd.read_parquet(features_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Verify required columns exist
    required = ["open", "high", "low", "close", "volume", atr_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")

    close = df["close"].to_numpy(dtype=np.float64)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)
    atr   = df[atr_col].to_numpy(dtype=np.float64)

    print(f"Running triple barrier on {len(df):,} bars ...")
    print(f"  TP mult     : {tp_mult}×  |  SL mult : {sl_mult}×  |  R:R = {tp_mult/sl_mult:.1f}")
    print(f"  Max hold    : {max_bars} bars ({max_bars}h)")
    print(f"  Min ATR pct : {min_atr_pct:.1%}")

    labels, hit_bars, log_returns = _triple_barrier_loop(
        close, high, low, atr,
        tp_mult, sl_mult, max_bars
    )

    # ── Build output DataFrame ─────────────────────────────────────────────
    out = pd.DataFrame(index=df.index)
    out["label"]       = labels
    out["hit_bars"]    = hit_bars
    out["log_return"]  = log_returns
    out["tp_price"]    = close + tp_mult * atr
    out["sl_price"]    = close - sl_mult * atr
    out["atr_used"]    = atr
    out["reward_risk"] = tp_mult / sl_mult

    # ── Filter dead-market bars ────────────────────────────────────────────
    # Where ATR is tiny relative to price, there's no tradeable move
    # Force these to label 0 regardless of what the barrier hit
    atr_pct = atr / close
    dead_market = atr_pct < min_atr_pct
    out.loc[dead_market, "label"] = 0
    print(f"  Dead market bars filtered : {dead_market.sum():,}")

    # ── Drop the last max_bars rows ────────────────────────────────────────
    # These bars don't have enough forward data for the vertical barrier
    # to be meaningful — their labels are unreliable
    out = out.iloc[:-max_bars]

    # ── Print distribution ─────────────────────────────────────────────────
    _print_label_stats(out)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "labels_1h.parquet"
    out.to_parquet(out_path)
    print(f"\n✓  Saved → {out_path}")

    return out


# ─────────────────────────────────────────────
#  DIAGNOSTICS
# ─────────────────────────────────────────────

def _print_label_stats(labels: pd.DataFrame) -> None:
    total  = len(labels)
    counts = labels["label"].value_counts().sort_index()
    pcts   = counts / total * 100

    print(f"\n{'─'*45}")
    print(f"  LABEL DISTRIBUTION  (n={total:,})")
    print(f"{'─'*45}")
    print(f"  Short  (-1) : {counts.get(-1, 0):>7,}  ({pcts.get(-1, 0):>5.1f}%)")
    print(f"  Neutral ( 0) : {counts.get( 0, 0):>7,}  ({pcts.get( 0, 0):>5.1f}%)")
    print(f"  Long   ( 1) : {counts.get( 1, 0):>7,}  ({pcts.get( 1, 0):>5.1f}%)")
    print(f"{'─'*45}")

    print(f"\n  AVG BARS TO EXIT")
    for lbl, name in [(-1, "Short"), (0, "Neutral"), (1, "Long")]:
        mask = labels["label"] == lbl
        if mask.sum() > 0:
            avg = labels.loc[mask, "hit_bars"].mean()
            print(f"  {name:<10} : {avg:.1f} bars")

    print(f"\n  AVG LOG RETURN AT EXIT")
    for lbl, name in [(-1, "Short"), (0, "Neutral"), (1, "Long")]:
        mask = labels["label"] == lbl
        if mask.sum() > 0:
            avg = labels.loc[mask, "log_return"].mean() * 100
            print(f"  {name:<10} : {avg:+.3f}%")


def plot_label_distribution(labels: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Add a year-month column, group manually
    tmp = labels[["label"]].copy()
    tmp["month"] = tmp.index.to_period("M").to_timestamp()

    monthly = tmp.groupby(["month", "label"]).size().reset_index(name="count")

    for ax, (lbl, name, color) in zip(axes, [
        ( 1, "Long  (+1)", "green"),
        ( 0, "Neutral (0)", "gray"),
        (-1, "Short (-1)", "red"),
    ]):
        data = monthly[monthly["label"] == lbl]
        ax.bar(data["month"], data["count"], color=color, alpha=0.7, width=20)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_title("Label Distribution Over Time (monthly)", fontsize=12)
    plt.tight_layout()
    plt.savefig("../../data/processed/label_distribution.png", dpi=150)
    plt.show()
    print("Saved → ../../data/processed/label_distribution.png")


# ─────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    labels = build_labels(
        features_path = PROCESSED_DIR / "features_1h.parquet",
        atr_col       = "atr_14",
        tp_mult       = 1.0,    # TP = 1× ATR away
        sl_mult       = 1.0,    # SL = 1× ATR away  →  2:1 reward/risk
        max_bars      = 48,     # max 48h hold
        min_atr_pct   = 0.003,  # skip if ATR < 0.3% of price
    )
    plot_label_distribution(labels)