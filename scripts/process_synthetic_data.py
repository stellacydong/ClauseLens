import os
import numpy as np
import pandas as pd

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
DEMO_DIR = "../data/demo"

N_DEMO = 1000

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# -----------------------------
# 1. Load Raw Data
# -----------------------------
treaties_df = pd.read_csv(os.path.join(RAW_DIR, "treaties_raw.csv"))
reinsurer_df = pd.read_csv(os.path.join(RAW_DIR, "reinsurer_info.csv"))

# -----------------------------
# 2. Clean & Standardize
# -----------------------------
treaties_df["submission_date"] = pd.to_datetime(treaties_df["submission_date"])
treaties_df["attachment_point"].fillna(0, inplace=True)
treaties_df["limit"].fillna(0, inplace=True)
treaties_df["quota_share"].fillna(0, inplace=True)

# Save cleaned synthetic dataset
treaties_clean_path = os.path.join(PROCESSED_DIR, "treaties_synthetic.csv")
treaties_df.to_csv(treaties_clean_path, index=False)
print(f"✅ Saved {treaties_clean_path} ({len(treaties_df)} rows)")

# -----------------------------
# 3. Merge Reinsurer Metadata
# -----------------------------
reinsurer_meta = reinsurer_df[reinsurer_df["entity_type"] == "reinsurer"][
    ["entity_id", "region", "incumbent_flag"]
].rename(columns={"entity_id": "reinsurer_id", "region": "reinsurer_region"})

merged_df = treaties_df.merge(reinsurer_meta, on="reinsurer_id", how="left")

# -----------------------------
# 4. Feature Engineering
# -----------------------------
features = merged_df.copy()

# Ratios
features["premium_to_limit_ratio"] = features.apply(
    lambda row: row["premium"] / row["limit"] if row["limit"] else 0, axis=1
)
features["premium_to_attachment_ratio"] = features.apply(
    lambda row: row["premium"] / row["attachment_point"] if row["attachment_point"] else 0, axis=1
)
features["log_premium"] = features["premium"].apply(lambda x: np.log1p(x))

# One-hot encode categorical columns
features = pd.get_dummies(
    features,
    columns=["treaty_type", "line_of_business", "region"],
    drop_first=True
)

# Select ML features
feature_cols = [
    "premium", "attachment_point", "limit", "quota_share",
    "premium_to_limit_ratio", "premium_to_attachment_ratio", "log_premium",
    "cvar_95", "incumbent_flag"
] + [col for col in features.columns if col.startswith("treaty_type_")
                                  or col.startswith("line_of_business_")
                                  or col.startswith("region_")]

features_ml = features[feature_cols]

# -----------------------------
# 5. Labels
# -----------------------------
labels = features[["accepted", "observed_loss_ratio"]].copy()
labels["deviation_score"] = labels["observed_loss_ratio"] - 0.65

# -----------------------------
# 6. Save Features & Labels
# -----------------------------
features_path = os.path.join(PROCESSED_DIR, "marketlens_features.parquet")
labels_path = os.path.join(PROCESSED_DIR, "marketlens_labels.parquet")

features_ml.to_parquet(features_path, index=False)
labels.to_parquet(labels_path, index=False)

print(f"✅ Saved {features_path} ({features_ml.shape[0]} rows, {features_ml.shape[1]} features)")
print(f"✅ Saved {labels_path} ({labels.shape[0]} rows, {labels.shape[1]} labels)")

# -----------------------------
# 7. Create Demo Subset for Streamlit
# -----------------------------
demo_sample_idx = features_ml.sample(N_DEMO, random_state=42).index

# Demo treaties CSV
demo_treaties = treaties_df.loc[demo_sample_idx]
demo_treaties_path = os.path.join(DEMO_DIR, "sample_treaties.csv")
demo_treaties.to_csv(demo_treaties_path, index=False)

# Demo MarketLens features
demo_features = features_ml.loc[demo_sample_idx]
demo_features_path = os.path.join(DEMO_DIR, "sample_marketlens.parquet")
demo_features.to_parquet(demo_features_path, index=False)

# Demo MarketLens labels
demo_labels = labels.loc[demo_sample_idx]
demo_labels_path = os.path.join(DEMO_DIR, "sample_marketlens_labels.parquet")
demo_labels.to_parquet(demo_labels_path, index=False)

print(f"✅ Demo sample saved: {demo_treaties_path} ({len(demo_treaties)} rows)")
print(f"✅ Demo MarketLens features: {demo_features_path}")
print(f"✅ Demo MarketLens labels: {demo_labels_path}")
