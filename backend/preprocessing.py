"""
THE-LAG — Preprocessing Pipeline
Splits ET/FT rows, merges coverage onto ET data, engineers lag_proxy, creates binary target.

Key insight: In the ΤΑΧΙΤΑΡΙ dataset, each user reads DIFFERENT texts with ET vs FT devices.
Therefore, we average FT coverage per word position (gid, sid, tid) across all users,
then merge this averaged coverage onto every ET row.
"""

import pandas as pd
import numpy as np
from config import (
    ET_TIMING_COLS, FEATURES, TARGET_COL, CLEAN_DATA_PATH
)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load .xlsx or .csv file.

    For .xlsx: try the fast Rust-based 'calamine' engine first
    (10-20x faster than openpyxl for large files). Fall back to openpyxl
    if calamine is unavailable or fails.
    """
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext == "xlsx":
        df = None
        try:
            df = pd.read_excel(filepath, engine="calamine")
            print("[preprocessing] Loaded xlsx with engine=calamine (fast path).")
        except Exception as e:
            print(f"[preprocessing] calamine failed ({e}); falling back to openpyxl.")
            df = pd.read_excel(filepath, engine="openpyxl")
    elif ext == "csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")
    print(f"[preprocessing] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def split_et_ft(df: pd.DataFrame):
    """
    Split dataset into Eye-Tracking and Finger-Tracking subsets.
    idDevice == 'ET' → eye tracking rows
    idDevice == 1    → finger tracking rows
    """
    et = df[df["idDevice"] == "ET"].copy()
    ft = df[df["idDevice"] == 1].copy()
    print(f"[preprocessing] ET rows: {len(et)}, FT rows: {len(ft)}")
    return et, ft


def merge_coverage(et: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """
    Average FT coverage per word position (gid, sid, tid) across all users,
    then merge onto ET rows. This handles the fact that users read different
    texts with ET vs FT devices.
    """
    ft_avg = ft.groupby(["gid", "sid", "tid"])["coverage"].mean().reset_index()
    ft_avg.rename(columns={"coverage": "coverage"}, inplace=True)
    print(f"[preprocessing] FT coverage averaged: {len(ft_avg)} word positions.")

    # ET may already have a coverage column (all NaN for ET rows) — drop it
    if "coverage" in et.columns:
        et = et.drop(columns=["coverage"])

    merged = pd.merge(et, ft_avg, on=["gid", "sid", "tid"], how="left")

    coverage_found = merged["coverage"].notna().sum()
    print(f"[preprocessing] Coverage merged: {coverage_found}/{len(merged)} ET rows got coverage.")

    # Fill remaining NaN coverage with mean
    mean_cov = merged["coverage"].mean()
    merged["coverage"] = merged["coverage"].fillna(mean_cov)

    return merged


def clean_et_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Fix ET timing columns: convert to numeric, coerce corrupted strings."""
    for col in ET_TIMING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[preprocessing] ET timing columns cleaned: {ET_TIMING_COLS}")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with absurdly large ET values or NaN TRT."""
    before = len(df)

    # Remove extreme outliers
    for col in ET_TIMING_COLS:
        if col in df.columns:
            df = df[df[col].notna() & (df[col].abs() < 1e10)]

    removed = before - len(df)
    print(f"[preprocessing] Removed {removed} rows (outliers + NaN ET). Remaining: {len(df)}")
    return df


def fill_missing_et(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate remaining missing ET values."""
    for col in ET_TIMING_COLS:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                df[col] = df[col].fillna(0)
                print(f"[preprocessing] Interpolated {missing} values in {col}.")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize len and freq to [0, 1]."""
    for col in ["len", "freq"]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0.0
            print(f"[preprocessing] Normalized {col} to [0, 1].")
    return df


def engineer_lag_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    lag_proxy = TRT_normalized − coverage
    
    High lag_proxy → eye spends more time than finger covers → harder word.
    Low/negative → finger tracks comfortably alongside eye → easier word.
    """
    if "TRT" not in df.columns:
        raise ValueError("Column 'TRT' is required to compute lag_proxy.")

    trt_min = df["TRT"].min()
    trt_max = df["TRT"].max()
    if trt_max > trt_min:
        df["TRT_normalized"] = (df["TRT"] - trt_min) / (trt_max - trt_min)
    else:
        df["TRT_normalized"] = 0.0

    df["coverage"] = df["coverage"].fillna(0.0)
    df["lag_proxy"] = df["TRT_normalized"] - df["coverage"]

    print(f"[preprocessing] lag_proxy: mean={df['lag_proxy'].mean():.4f}, "
          f"std={df['lag_proxy'].std():.4f}, "
          f"min={df['lag_proxy'].min():.4f}, max={df['lag_proxy'].max():.4f}")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if TRT > mean AND isReg == 1.
    """
    if "TRT" not in df.columns or "isReg" not in df.columns:
        raise ValueError("Columns 'TRT' and 'isReg' required for target.")

    trt_mean = df["TRT"].mean()
    df["isReg"] = pd.to_numeric(df["isReg"], errors="coerce").fillna(0)
    df[TARGET_COL] = ((df["TRT"] > trt_mean) & (df["isReg"] == 1)).astype(int)

    pos = df[TARGET_COL].sum()
    neg = len(df) - pos
    ratio = pos / len(df) * 100
    print(f"[preprocessing] Target: Positive={pos}, Negative={neg} ({ratio:.1f}%)")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select features + target, drop NaN rows."""
    needed = FEATURES + [TARGET_COL]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        print(f"[preprocessing] WARNING: Missing columns (skipped): {missing_cols}")
        needed = [c for c in needed if c in df.columns]

    df_out = df[needed].copy()
    before = len(df_out)
    df_out = df_out.dropna()
    dropped = before - len(df_out)
    if dropped > 0:
        print(f"[preprocessing] Dropped {dropped} rows with NaN.")

    print(f"[preprocessing] Final: {len(df_out)} rows, {len(df_out.columns)} columns.")
    return df_out


def run_preprocessing(filepath: str) -> str:
    """Full preprocessing pipeline."""
    print("=" * 50)
    print("[preprocessing] Starting pipeline...")
    print("=" * 50)

    df = load_raw_data(filepath)
    et, ft = split_et_ft(df)
    df = merge_coverage(et, ft)
    df = clean_et_timing(df)
    df = remove_outliers(df)
    df = fill_missing_et(df)
    df = normalize_features(df)
    df = engineer_lag_proxy(df)
    df = create_target(df)
    df = select_features(df)

    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"[preprocessing] Saved to {CLEAN_DATA_PATH}")
    print("[preprocessing] Done.\n")

    return CLEAN_DATA_PATH


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_preprocessing(sys.argv[1])
    else:
        print("Usage: python preprocessing.py <path_to_file>")