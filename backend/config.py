"""
THE-LAG — Central Configuration
All paths, feature lists, and shared settings in one place.
"""

import os
import sys

# ── UTF-8 stdout fix (Windows + Greek paths) ──
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── File paths ──
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "data_clean.csv")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "model_xgb.joblib")
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "model_mlp.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
SPLIT_INFO_PATH = os.path.join(MODELS_DIR, "split_info.joblib")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
SHAP_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "shap_summary.png")
SHAP_DEPENDENCE_PATH = os.path.join(OUTPUT_DIR, "shap_dependence.png")
XCORR_PLOT_PATH = os.path.join(OUTPUT_DIR, "cross_correlation.png")

# ── Feature Configuration ──
# Eye-tracking timing fields (may have corrupted strings)
ET_TIMING_COLS = ["FFD", "FPD", "TRT", "RPD"]

# All features used for ML training
# IMPORTANT: isReg is NOT here — it's part of the target (data leakage)
FEATURES = [
    # Linguistic
    "len", "freq", "orthoSyllablesCount",
    # Session
    "idSession",
    # WAIS cognitive scores
    "WAIS Coding", "WAIS Digit Span Ascending", "WAIS Vocabulary",
    # Eye-tracking
    "FFD", "FPD", "TRT", "RPD", "fixNum",
    # Finger-tracking
    "coverage",
    # Engineered
    "lag_proxy",
]

# Target: binary classification
# 1 = high cognitive load (TRT > mean AND isReg == 1)
# 0 = normal processing
TARGET_COL = "target"

# ── Model Hyperparameters ──
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

MLP_HIDDEN_LAYERS = (128, 64, 32)
MLP_MAX_ITER = 300
MLP_RANDOM_STATE = 42

# ── Train/Test Split ──
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── SHAP ──
SHAP_SAMPLE_SIZE = 200
SHAP_TOP_FEATURES = 15

print("[config] Configuration loaded successfully.")