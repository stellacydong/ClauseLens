import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "treaties_raw.csv")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed", "treaties_synthetic.csv")

FEATURES_OUT = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_features.parquet")
LABELS_OUT = os.path.join(PROJECT_ROOT, "data", "processed", "marketlens_labels.parquet")

# Demo subset for Streamlit visualization
DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)
DEMO_FEATURES = os.path.join(DEMO_DIR, "sample_marketlens.parquet")
DEMO_LABELS = os.path.join(DEMO_DIR, "sample_marketlens_labels.parquet")


def preprocess_marketlens():
    """Load treaty data, generate features & labels, and save parquet files for MarketLens."""
    # -----------------------------
    # 1. Load Data
    # -----------------------------
    if os.path.exists(DATA_PROCESSED):
        df = pd.read_csv(DATA_PROCESSED)
        print(f"✅ Loaded {len(df)} synthetic treaties from {DATA_PROCESSED}")
    elif os.path.exists(DATA_RAW):
        df = pd.read_csv(DATA_RAW)
        print(f"✅ Loaded {len(df)} raw treaties from {DATA_RAW}")
    else:
        raise FileNotFoundError(f"No treaty data found at {DATA_PROCESSED} or {DATA_RAW}")

    # -----------------------------
    # 2. Auto-detect/rename columns
    # -----------------------------
    # Ensure numeric columns
    expected_numeric = ["limit", "premium", "attachment"]
    rename_map = {"attachment_point": "attachment", "att_point": "attachment"}
    df.rename(columns=rename_map, inplace=True)

    for col in expected_numeric:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found. Creating dummy zeros.")
            df[col] = 0.0

    # -----------------------------
    # 3. Generate Features
    # -----------------------------
    # Encode line_of_business and region if present
    cat_cols = [c for c in ["line_of_business", "region"] if c in df.columns]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    X_cat = pd.DataFrame()
    if cat_cols:
        X_cat = pd.DataFrame(
            encoder.fit_transform(df[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols)
        )

    X_num = df[expected_numeric].copy()
    X = pd.concat([X_num, X_cat], axis=1)

    # -----------------------------
    # 4. Generate Labels
    # -----------------------------
    if "acceptance" not in df.columns:
        print("⚠️ 'acceptance' not found. Generating synthetic labels...")
        df["acceptance"] = (df["premium"] / (df["limit"] + 1e-6)).apply(lambda x: 1 if x < 0.8 else 0)

    if "expected_loss_ratio" not in df.columns:
        print("⚠️ 'expected_loss_ratio' not found. Generating synthetic ratios...")
        df["expected_loss_ratio"] = (df["limit"] / (df["premium"] + 1e-6)) * 0.05
        df["expected_loss_ratio"] = df["expected_loss_ratio"].clip(0, 3.0)

    labels = df[["acceptance", "expected_loss_ratio"]]

    # -----------------------------
    # 5. Save Outputs
    # -----------------------------
    os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)
    X.to_parquet(FEATURES_OUT, index=False)
    labels.to_parquet(LABELS_OUT, index=False)

    print(f"✅ Features saved to {FEATURES_OUT} ({X.shape})")
    print(f"✅ Labels saved to {LABELS_OUT} ({labels.shape})")

    # -----------------------------
    # 6. Demo Subset for Streamlit
    # -----------------------------
    X_demo = X.sample(min(1000, len(X)), random_state=42)
    y_demo = labels.loc[X_demo.index]

    X_demo.to_parquet(DEMO_FEATURES, index=False)
    y_demo.to_parquet(DEMO_LABELS, index=False)
    print(f"🎯 Demo subset saved:\n   - {DEMO_FEATURES} ({X_demo.shape})\n   - {DEMO_LABELS} ({y_demo.shape})")


if __name__ == "__main__":
    preprocess_marketlens()
