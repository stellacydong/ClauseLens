import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_features.parquet")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_labels.parquet")

MODEL_DIR = os.path.join(PROJECT_ROOT, "marketlens", "models")
ACCEPTANCE_MODEL = os.path.join(MODEL_DIR, "xgb_acceptance.pkl")

# Primary outputs in models folder
SHAP_PLOT_PATH = os.path.join(MODEL_DIR, "shap_summary.png")
FAIRNESS_CSV_PATH = os.path.join(MODEL_DIR, "fairness_audit.csv")

# Extra outputs for Streamlit demo
DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)
SHAP_PLOT_DEMO = os.path.join(DEMO_DIR, "shap_summary.png")
FAIRNESS_CSV_DEMO = os.path.join(DEMO_DIR, "fairness_audit.csv")


def load_data_and_model():
    """Load MarketLens features, labels, and trained acceptance model."""
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("❌ Run marketlens/preprocess.py first to generate features and labels.")

    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)

    if not os.path.exists(ACCEPTANCE_MODEL):
        raise FileNotFoundError("❌ Acceptance model not found. Run marketlens/train_marketlens.py first.")

    model_acceptance = joblib.load(ACCEPTANCE_MODEL)
    return X, y, model_acceptance


def run_shap_explainability(model, X, max_display=15):
    """Generates SHAP values and saves summary plots to both model dir and demo dir."""
    print("\n🔹 Running SHAP explainability...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()

    # Save to both paths
    for path in [SHAP_PLOT_PATH, SHAP_PLOT_DEMO]:
        plt.savefig(path, dpi=200)
        print(f"✅ SHAP summary plot saved to {path}")

    return shap_values


def run_fairness_audit(model, X):
    """Computes group-level acceptance probabilities and saves results."""
    print("\n🔹 Running fairness audit...")

    # Model predictions
    pred_probs = model.predict_proba(X)[:, 1]

    # Auto-detect group columns
    group_cols = [c for c in X.columns if c.startswith("line_of_business_") or c.startswith("region_")]
    if not group_cols:
        print("⚠️ No group columns found for fairness audit. Skipping.")
        return pd.DataFrame()

    # Compute mean predictions per group
    group_results = {}
    for col in group_cols:
        idx = X[col] == 1
        if idx.sum() > 0:
            group_results[col] = pred_probs[idx].mean()

    fairness_df = pd.DataFrame(list(group_results.items()), columns=["group", "mean_acceptance_prob"])
    fairness_df.sort_values("mean_acceptance_prob", ascending=False, inplace=True)

    # Save to both paths
    for path in [FAIRNESS_CSV_PATH, FAIRNESS_CSV_DEMO]:
        fairness_df.to_csv(path, index=False)
        print(f"✅ Fairness audit results saved to {path}")

    print("📊 Acceptance Probability by Group:")
    print(fairness_df)

    return fairness_df


if __name__ == "__main__":
    print("🚀 MarketLens Fairness Audit starting...")

    X, y, model_acceptance = load_data_and_model()
    print(f"✅ Loaded data: {X.shape}, labels: {y.shape}")

    # 1. SHAP explainability
    run_shap_explainability(model_acceptance, X, max_display=15)

    # 2. Fairness audit
    run_fairness_audit(model_acceptance, X)

    print("\n🎉 Fairness audit complete! Review results in both `marketlens/models/` and `data/demo/`")

