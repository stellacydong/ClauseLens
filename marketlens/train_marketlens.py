import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from math import sqrt

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_features.parquet")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_labels.parquet")

MODEL_DIR = os.path.join(PROJECT_ROOT, "marketlens", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)

ACCEPTANCE_MODEL_PKL = os.path.join(MODEL_DIR, "xgb_acceptance.pkl")
LOSS_MODEL_PKL = os.path.join(MODEL_DIR, "xgb_lossratio.pkl")

# Optional JSON boosters for model inspection
ACCEPTANCE_JSON = os.path.join(MODEL_DIR, "xgb_acceptance.json")
LOSS_JSON = os.path.join(MODEL_DIR, "xgb_lossratio.json")

# Demo copies for Streamlit
ACCEPTANCE_MODEL_PKL_DEMO = os.path.join(DEMO_DIR, "xgb_acceptance.pkl")


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    """Load preprocessed features and labels for MarketLens."""
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("❌ Preprocessed features/labels not found. Run preprocess.py first.")

    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)

    if "acceptance" not in y.columns or "expected_loss_ratio" not in y.columns:
        raise KeyError("❌ Labels must include 'acceptance' and 'expected_loss_ratio' columns.")

    return X, y["acceptance"], y["expected_loss_ratio"]


# -----------------------------
# Training Function
# -----------------------------
def train_xgb_models():
    print("🚀 Training MarketLens models...")

    X, y_accept, y_loss = load_data()

    # -----------------------------
    # 1. Acceptance Model (Classifier)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y_accept, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc"
    )
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"✅ Acceptance Model AUC: {auc:.4f}")

    # Save model for SHAP and fairness audit
    joblib.dump(clf, ACCEPTANCE_MODEL_PKL)
    joblib.dump(clf, ACCEPTANCE_MODEL_PKL_DEMO)
    print(f"💾 Acceptance model saved to {ACCEPTANCE_MODEL_PKL} and demo copy to {ACCEPTANCE_MODEL_PKL_DEMO}")

    # Optional: save JSON booster
    clf.get_booster().save_model(ACCEPTANCE_JSON)

    # -----------------------------
    # 2. Loss Ratio Model (Regressor)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y_loss, test_size=0.2, random_state=42)

    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    # Compute RMSE (manual sqrt for compatibility)
    rmse = sqrt(mean_squared_error(y_test, preds))
    print(f"✅ Loss Ratio Model RMSE: {rmse:.4f}")

    # Save model
    joblib.dump(reg, LOSS_MODEL_PKL)
    print(f"💾 Loss ratio model saved to {LOSS_MODEL_PKL}")

    # Optional: save JSON booster
    reg.get_booster().save_model(LOSS_JSON)

    print("\n🎉 Training complete! Models are ready for SHAP, fairness audit, and MarketLens dashboard.")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_xgb_models()
