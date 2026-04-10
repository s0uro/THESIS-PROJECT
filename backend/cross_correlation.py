"""
THE-LAG — Cross-Correlation Analysis
Computes cross-correlation between eye-tracking TRT and finger-tracking coverage
to examine temporal alignment patterns.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

from config import CLEAN_DATA_PATH, XCORR_PLOT_PATH


def run_cross_correlation() -> dict:
    """
    Compute and plot cross-correlation between TRT_normalized and coverage.
    Returns dict with plot path and peak lag info.
    """
    print("=" * 50)
    print("[xcorr] Starting cross-correlation analysis...")
    print("=" * 50)

    df = pd.read_csv(CLEAN_DATA_PATH)

    # We need lag_proxy components
    if "lag_proxy" not in df.columns:
        print("[xcorr] WARNING: lag_proxy not found. Skipping.")
        return {}

    # Use coverage as FT signal proxy
    # Reconstruct TRT_normalized from lag_proxy + coverage
    # lag_proxy = TRT_normalized - coverage => TRT_normalized = lag_proxy + coverage
    if "coverage" in df.columns:
        trt_norm = df["lag_proxy"] + df["coverage"]
        ft_signal = df["coverage"].values
    else:
        print("[xcorr] WARNING: coverage column not found. Skipping.")
        return {}

    et_signal = trt_norm.values

    # Remove NaN
    mask = ~(np.isnan(et_signal) | np.isnan(ft_signal))
    et_signal = et_signal[mask]
    ft_signal = ft_signal[mask]

    if len(et_signal) < 10:
        print("[xcorr] WARNING: Too few valid samples for cross-correlation.")
        return {}

    # Normalize signals (zero mean, unit variance)
    et_signal = (et_signal - np.mean(et_signal)) / (np.std(et_signal) + 1e-8)
    ft_signal = (ft_signal - np.mean(ft_signal)) / (np.std(ft_signal) + 1e-8)

    # Compute cross-correlation
    max_lag = min(100, len(et_signal) // 4)
    correlation = signal.correlate(et_signal, ft_signal, mode="full")
    correlation = correlation / len(et_signal)  # Normalize
    lags = signal.correlation_lags(len(et_signal), len(ft_signal), mode="full")

    # Trim to max_lag range
    center = len(lags) // 2
    lag_range = slice(center - max_lag, center + max_lag + 1)
    lags_trimmed = lags[lag_range]
    corr_trimmed = correlation[lag_range]

    # Find peak
    peak_idx = np.argmax(np.abs(corr_trimmed))
    peak_lag = lags_trimmed[peak_idx]
    peak_corr = corr_trimmed[peak_idx]

    print(f"[xcorr] Peak correlation: {peak_corr:.4f} at lag={peak_lag}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags_trimmed, corr_trimmed, color="#1a1a1a", linewidth=1.2)
    ax.axvline(x=0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color="#999999", linestyle="-", linewidth=0.5, alpha=0.4)

    # Mark peak
    ax.plot(peak_lag, peak_corr, "o", color="#dc2626", markersize=8, zorder=5)
    ax.annotate(
        f"Peak: lag={peak_lag}, r={peak_corr:.3f}",
        xy=(peak_lag, peak_corr),
        xytext=(peak_lag + max_lag * 0.15, peak_corr + 0.05),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#666"),
        color="#333",
    )

    ax.set_xlabel("Lag (word positions)", fontsize=11)
    ax.set_ylabel("Cross-Correlation", fontsize=11)
    ax.set_title("Cross-Correlation: Eye-Tracking (TRT) vs Finger-Tracking (Coverage)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(XCORR_PLOT_PATH, dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"[xcorr] Plot saved to {XCORR_PLOT_PATH}")
    print("[xcorr] Done.\n")

    return {
        "plot_path": XCORR_PLOT_PATH,
        "peak_lag": int(peak_lag),
        "peak_correlation": round(float(peak_corr), 4),
    }


if __name__ == "__main__":
    run_cross_correlation()