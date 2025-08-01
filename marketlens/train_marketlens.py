import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

# -----------------------------
# Project Root and Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_features.parquet")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_labels.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "marketlens", "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_marketlens_models():
    """
    Train MarketLens models:
    1. XGBoost Classifier -> Acceptance likelihood
    2. XGBoost Regressor -> Expected loss ratio
    Saves models into marketlens/models/
    """
    # -----------------------------
    # 1. Load Features & Labels
    # -----------------------------
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "❌ Missing processed features or labels. Run `marketlens/preprocess.py` first."
        )

    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)

    print(f"✅ Loaded features {X.shape} and labels {y.shape}")

    # -----------------------------
    # 2. Train Acceptance Likelihood Model (Classification)
    # -----------------------------
    print("\n🎯 Training acceptance likelihood model...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y["acceptance"], test_size=0.2, random_state=42
    )

    model_acceptance = xgb.XGBClassifier(
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"  # Efficient training for large datasets
    )

    model_acceptance.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model_acceptance.predict_proba(X_val)[:, 1])
    print(f"✅ Acceptance Model AUC: {auc:.3f}")

    acceptance_model_path = os.path.join(MODEL_DIR, "xgb_acceptance.pkl")
    joblib.dump(model_acceptance, acceptance_model_path)
    print(f"💾 Saved acceptance model to {acceptance_model_path}")

    # -----------------------------
    # 3. Train Expected Loss Ratio Model (Regression)
    # -----------------------------
    print("\n🎯 Training expected loss ratio model...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y["loss_ratio"], test_size=0.2, random_state=42
    )

    model_lossratio = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    model_lossratio.fit(X_train, y_train)
    mse = mean_squared_error(y_val, model_lossratio.predict(X_val))
    print(f"✅ Loss Ratio Model MSE: {mse:.3f}")

    lossratio_model_path = os.path.join(MODEL_DIR, "xgb_lossratio.pkl")
    joblib.dump(model_lossratio, lossratio_model_path)
    print(f"💾 Saved loss ratio model to {lossratio_model_path}")

    print("\n🎉 Training complete! MarketLens models ready for use in the YC demo.")


if __name__ == "__main__":
    train_marketlens_models()
