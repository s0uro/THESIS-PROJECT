"""
THE-LAG — Model Evaluation
Evaluates XGBoost and MLP, saves metrics.json.
"""

import json
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import (
    XGB_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH,
    SPLIT_INFO_PATH, METRICS_PATH,
)


def run_evaluation() -> dict:
    """
    Evaluate both models on test set. Returns metrics dict and saves to JSON.
    """
    print("=" * 50)
    print("[evaluation] Starting evaluation pipeline...")
    print("=" * 50)

    # Load models and test data
    xgb_model = joblib.load(XGB_MODEL_PATH)
    mlp_model = joblib.load(MLP_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    split_info = joblib.load(SPLIT_INFO_PATH)

    X_test = split_info["X_test"]
    y_test = split_info["y_test"]
    feature_names = split_info["feature_names"]

    print(f"[evaluation] Test set: {len(X_test)} samples.")

    # ── XGBoost predictions ──
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
    xgb_cm = confusion_matrix(y_test, xgb_pred).tolist()
    xgb_report = classification_report(y_test, xgb_pred, zero_division=0)

    print(f"[evaluation] XGBoost — Accuracy: {xgb_acc:.4f}, F1: {xgb_f1:.4f}")

    # ── MLP predictions ──
    X_test_scaled = scaler.transform(X_test)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred, zero_division=0)
    mlp_cm = confusion_matrix(y_test, mlp_pred).tolist()
    mlp_report = classification_report(y_test, mlp_pred, zero_division=0)

    print(f"[evaluation] MLP    — Accuracy: {mlp_acc:.4f}, F1: {mlp_f1:.4f}")

    # ── Build metrics dict ──
    metrics = {
        "xgb_accuracy": round(xgb_acc, 4),
        "xgb_f1": round(xgb_f1, 4),
        "xgb_cm": xgb_cm,
        "xgb_report": xgb_report,
        "mlp_accuracy": round(mlp_acc, 4),
        "mlp_f1": round(mlp_f1, 4),
        "mlp_cm": mlp_cm,
        "mlp_report": mlp_report,
        "test_size": len(X_test),
        "feature_names": feature_names,
    }

    # Save
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[evaluation] Metrics saved to {METRICS_PATH}")
    print("[evaluation] Done.\n")

    return metrics


if __name__ == "__main__":
    run_evaluation()