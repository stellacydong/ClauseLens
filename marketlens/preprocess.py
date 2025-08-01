import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Project Root and Paths
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
    """
    Prepares MarketLens features and labels from synthetic treaty data.
    - Auto-renames attachment columns
    - Skips missing numeric columns gracefully
    - Prints warnings if schema is incomplete
    """
    # -----------------------------
    # 1. Load Data
    # -----------------------------
    if os.path.exists(DATA_PROCESSED):
        df = pd.read_csv(DATA_PROCESSED)
        print(f"✅ Loaded {len(df)} synthetic treaties from {DATA_PROCESSED}")
    elif os.path.exists(DATA_RAW):
        df = pd.read_csv(DATA_RAW)
        print(f"⚠️ Processed file not found, using raw data: {DATA_RAW}")
    else:
        raise FileNotFoundError(
            f"No treaty data found at:\n"
            f"  - {DATA_PROCESSED}\n"
            f"  - {DATA_RAW}\n\n"
            f"👉 Run scripts/generate_synthetic_treaties.py first."
        )

    # -----------------------------
    # 2. Auto-Rename Attachment Column
    # -----------------------------
    attachment_aliases = ["attachment_point", "att_point", "att", "attach"]
    for alias in attachment_aliases:
        if alias in df.columns and "attachment" not in df.columns:
            df.rename(columns={alias: "attachment"}, inplace=True)
            print(f"🔄 Renamed '{alias}' to 'attachment'")
            break

    # -----------------------------
    # 3. Define Columns
    # -----------------------------
    cat_cols = ["line_of_business", "region"]
    num_cols = ["limit", "attachment", "premium"]

    # Check for missing columns
    missing_cats = [c for c in cat_cols if c not in df.columns]
    missing_nums = [c for c in num_cols if c not in df.columns]

    if missing_cats:
        print(f"⚠️ Warning: Missing categorical columns {missing_cats}. Filling with 'Unknown'.")
        for c in missing_cats:
            df[c] = "Unknown"

    if missing_nums:
        print(f"⚠️ Warning: Missing numeric columns {missing_nums}. Filling with zeros.")
        for c in missing_nums:
            df[c] = 0.0

    # -----------------------------
    # 4. Feature Engineering
    # -----------------------------
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_features = enc.fit_transform(df[cat_cols])
    X = pd.DataFrame(cat_features, columns=enc.get_feature_names_out(cat_cols))
    X[num_cols] = df[num_cols].reset_index(drop=True)

    # -----------------------------
    # 5. Label Engineering
    # -----------------------------
    # Acceptance likelihood: low premium-to-limit ratio = more likely to be accepted
    with pd.option_context('mode.chained_assignment', None):
        df["limit"] = df["limit"].replace(0, 1)  # Avoid div by zero

    y_acceptance = (df["premium"] / df["limit"] < 0.05).astype(int)
    y_lossratio = (df["limit"] / (df["attachment"] + 1)).clip(0, 5)

    y = pd.DataFrame({
        "acceptance": y_acceptance,
        "loss_ratio": y_lossratio
    })

    # -----------------------------
    # 6. Save Full Data
    # -----------------------------
    X.to_parquet(FEATURES_OUT)
    y.to_parquet(LABELS_OUT)

    print(f"✅ Features saved to {FEATURES_OUT} ({X.shape})")
    print(f"✅ Labels saved to {LABELS_OUT} ({y.shape})")

    # -----------------------------
    # 7. Save 1k-row Demo Subset
    # -----------------------------
    demo_X = X.sample(min(1000, len(X)), random_state=42)
    demo_y = y.loc[demo_X.index]

    demo_X.to_parquet(DEMO_FEATURES)
    demo_y.to_parquet(DEMO_LABELS)

    print(f"🎯 Demo subset saved:")
    print(f"   - {DEMO_FEATURES} ({demo_X.shape})")
    print(f"   - {DEMO_LABELS} ({demo_y.shape})")


if __name__ == "__main__":
    preprocess_marketlens()
