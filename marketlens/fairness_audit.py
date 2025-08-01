import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Project Root and Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_features.parquet")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_labels.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "marketlens", "models")
ACCEPTANCE_MODEL = os.path.join(MODEL_DIR, "xgb_acceptance.pkl")

SHAP_PLOT_PATH = os.path.join(MODEL_DIR, "shap_summary.png")
FAIRNESS_CSV_PATH = os.path.join(MODEL_DIR, "fairness_audit.csv")


def load_data_and_model():
    """Load MarketLens feature matrix, labels, and acceptance model."""
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("❌ Run marketlens/preprocess.py first to generate features and labels.")

    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)

    if not os.path.exists(ACCEPTANCE_MODEL):
        raise FileNotFoundError("❌ Acceptance model not found. Run marketlens/train_marketlens.py first.")

    model_acceptance = joblib.load(ACCEPTANCE_MODEL)
    return X, y, model_acceptance


def run_shap_explainability(model, X, max_display=15, save_plot=True):
    """
    Generates SHAP values and summary plot for the acceptance model.
    """
    print("\n🔹 Running SHAP explainability...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()

    if save_plot:
        plt.savefig(SHAP_PLOT_PATH, dpi=200)
        print(f"✅ SHAP summary plot saved to {SHAP_PLOT_PATH}")
    else:
        plt.show()

    return shap_values


def run_fairness_audit(model, X, group_prefixes=("line_of_business_", "region_")):
    """
    Simple fairness audit: Computes average predicted acceptance probability by group.
    """
    print("\n🔹 Running fairness audit...")

    # Model predictions
    pred_probs = model.predict_proba(X)[:, 1]

    # Detect groups by column prefix
    group_cols = [col for col in X.columns if col.startswith(group_prefixes)]
    if not group_cols:
        # Also handle tuple of prefixes properly
        group_cols = [col for col in X.columns if col.startswith("line_of_business_") or col.startswith("region_")]

    if not group_cols:
        print("⚠️ No group columns found for fairness audit. Skipping.")
        return

    # Compute mean predictions per group
    group_results = {}
    for col in group_cols:
        idx = X[col] == 1
        if idx.sum() > 0:
            group_results[col] = pred_probs[idx].mean()

    # Sort and display results
    print("📊 Acceptance Probability by Group:")
    for group, prob in sorted(group_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group:<25} {prob:.3f}")

    # Save as CSV
    fairness_df = pd.DataFrame(list(group_results.items()), columns=["group", "mean_acceptance_prob"])
    fairness_df.to_csv(FAIRNESS_CSV_PATH, index=False)
    print(f"✅ Fairness audit results saved to {FAIRNESS_CSV_PATH}")


if __name__ == "__main__":
    print("🚀 MarketLens Fairness Audit starting...")

    X, y, model_acceptance = load_data_and_model()
    print(f"✅ Loaded data: {X.shape}, labels: {y.shape}")

    # 1. SHAP explainability
    run_shap_explainability(model_acceptance, X, max_display=15, save_plot=True)

    # 2. Fairness audit
    run_fairness_audit(model_acceptance, X)

    print("\n🎉 Fairness audit complete! Review results in marketlens/models/")
