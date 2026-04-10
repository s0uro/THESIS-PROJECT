"""
THE-LAG — SHAP Explainability
Generates SHAP summary and dependence plots for XGBoost.
"""

import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from config import (
    XGB_MODEL_PATH, SPLIT_INFO_PATH,
    SHAP_SUMMARY_PATH, SHAP_DEPENDENCE_PATH,
    SHAP_SAMPLE_SIZE, SHAP_TOP_FEATURES,
)


def run_shap() -> dict:
    """
    Generate SHAP plots for XGBoost model.
    Returns dict with paths to saved images.
    """
    print("=" * 50)
    print("[shap] Starting SHAP explainability...")
    print("=" * 50)

    # Load model and test data
    xgb_model = joblib.load(XGB_MODEL_PATH)
    split_info = joblib.load(SPLIT_INFO_PATH)
    X_test = split_info["X_test"]
    feature_names = split_info["feature_names"]

    # Sample for performance
    n_samples = min(SHAP_SAMPLE_SIZE, len(X_test))
    indices = np.random.RandomState(42).choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[indices]

    print(f"[shap] Computing SHAP values for {n_samples} samples...")

    # TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    print("[shap] SHAP values computed.")

    # ── Summary Plot (dot plot, top features) ──
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=SHAP_TOP_FEATURES,
        show=False,
        plot_size=(10, 7),
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[shap] Summary plot saved to {SHAP_SUMMARY_PATH}")

    # ── Dependence Plot for lag_proxy (if present) ──
    if "lag_proxy" in feature_names:
        lag_idx = feature_names.index("lag_proxy")
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            lag_idx,
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(SHAP_DEPENDENCE_PATH, dpi=100, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"[shap] Dependence plot saved to {SHAP_DEPENDENCE_PATH}")
    else:
        print("[shap] WARNING: lag_proxy not in features, skipping dependence plot.")

    print("[shap] Done.\n")

    return {
        "summary_path": SHAP_SUMMARY_PATH,
        "dependence_path": SHAP_DEPENDENCE_PATH,
    }


if __name__ == "__main__":
    run_shap()