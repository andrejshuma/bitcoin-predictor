import pandas as pd
import numpy as np

def buy_col_ranged(row):
    if pd.isna(row.next_high) or pd.isna(row.next_low):
        return np.nan
    if row.next_low >= row.high:
        return 2
    elif row.next_high > row.high:
        return 1
    elif row.next_high > row.low:
        return 0
    else:
        return -1
def sell_col_ranged(row):
    if pd.isna(row.next_high) or pd.isna(row.next_low):
        return np.nan
    if row.next_high <= row.low:
        return 2
    elif row.next_low < row.low:
        return 1
    elif row.next_low < row.high:
        return 0
    else:
        return -1

def buy_col(row):
    if pd.isna(row.next_high) or pd.isna(row.next_low):
        return np.nan
    return (row.next_high - row.high) / row.high

def sell_col(row):
    if pd.isna(row.next_high) or pd.isna(row.next_low):
        return np.nan
    return (row.low - row.next_low) / row.low

def define_activity(row):
    if row.signal_scaled < 0.2:
        return 'Strong Sell'
    elif row.signal_scaled < 0.4:
        return 'Sell'
    elif row.signal_scaled < 0.6:
        return 'Hold'
    elif row.signal_scaled < 0.8:
        return 'Buy'
    else:
        return 'Strong Buy'