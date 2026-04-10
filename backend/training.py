"""
THE-LAG — Model Training
Trains XGBoost and MLP classifiers with SMOTE oversampling.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import (
    CLEAN_DATA_PATH, FEATURES, TARGET_COL,
    XGB_PARAMS, MLP_HIDDEN_LAYERS, MLP_MAX_ITER, MLP_RANDOM_STATE,
    TEST_SIZE, RANDOM_STATE,
    XGB_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH, SPLIT_INFO_PATH,
)


def load_clean_data() -> pd.DataFrame:
    """Load the preprocessed dataset."""
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"[training] Loaded clean data: {len(df)} rows.")
    return df


def prepare_splits(df: pd.DataFrame):
    """
    Split data into train/test sets (stratified).
    Returns X_train, X_test, y_train, y_test, feature_names.
    """
    # Use only features that exist in the dataframe
    available_features = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[training] WARNING: Missing features (skipped): {missing}")

    X = df[available_features].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"[training] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[training] Train class distribution: "
          f"0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")

    return X_train, X_test, y_train, y_test, available_features


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to balance classes."""
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)

    if pos_count < 2:
        print("[training] WARNING: Too few positive samples for SMOTE. Skipping.")
        return X_train, y_train

    print(f"[training] Before SMOTE: 0={neg_count}, 1={pos_count}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"[training] After SMOTE:  0={np.sum(y_res == 0)}, 1={np.sum(y_res == 1)}")
    return X_res, y_res


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    print("[training] Training XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    print("[training] XGBoost training complete.")
    return model


def train_mlp(X_train, y_train, scaler):
    """Train MLP classifier with scaled features."""
    print("[training] Training MLP...")
    X_scaled = scaler.transform(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        max_iter=MLP_MAX_ITER,
        random_state=MLP_RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_scaled, y_train)
    print("[training] MLP training complete.")
    return model


def run_training() -> dict:
    """
    Full training pipeline. Returns dict with model paths.
    """
    print("=" * 50)
    print("[training] Starting training pipeline...")
    print("=" * 50)

    df = load_clean_data()
    X_train, X_test, y_train, y_test, feature_names = prepare_splits(df)

    # SMOTE oversampling
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Scaler for MLP
    scaler = StandardScaler()
    scaler.fit(X_train_res)

    # Train models
    xgb_model = train_xgboost(X_train_res, y_train_res)
    mlp_model = train_mlp(X_train_res, y_train_res, scaler)

    # Save everything
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    joblib.dump(mlp_model, MLP_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }, SPLIT_INFO_PATH)

    print(f"[training] Models saved to {XGB_MODEL_PATH}, {MLP_MODEL_PATH}")
    print(f"[training] Scaler saved to {SCALER_PATH}")
    print(f"[training] Split info saved to {SPLIT_INFO_PATH}")
    print("[training] Done.\n")

    return {
        "xgb_model_path": XGB_MODEL_PATH,
        "mlp_model_path": MLP_MODEL_PATH,
        "scaler_path": SCALER_PATH,
    }


if __name__ == "__main__":
    run_training()