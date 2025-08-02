#!/usr/bin/env python3
"""
02_process_synthetic_data.py

Cleans and feature-engineers the synthetic treaty dataset for MarketLens.
- Loads synthetic treaty data from 01_generate_synthetic_treaties.py
- Performs data cleaning & validation
- Adds engineered features (ratios, risk scores)
- Creates MarketLens-ready feature and label datasets
- Saves processed files to data/processed/
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Optional: Load config
import yaml


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def setup_logger(log_dir: str = "outputs/logs", script_name: str = "02_process_synthetic_data"):
    """Initialize a logger for the script."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console
    logging.info(f"Logger initialized. Output -> {log_file}")


def load_synthetic_data(input_path: str) -> pd.DataFrame:
    """Load synthetic treaties CSV or Parquet file."""
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    logging.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns from {input_path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning and validation."""
    # Drop duplicates
    df = df.drop_duplicates(subset=["TreatyID"])
    
    # Ensure numeric fields have no negatives
    numeric_cols = ["Retention", "Limit", "Premium", "ExpectedLoss"]
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    
    # Validate that Limit >= Retention
    df = df[df["Limit"] >= df["Retention"]]

    logging.info(f"After cleaning: {len(df)} rows remain.")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create MarketLens-ready features."""
    df["Coverage"] = df["Limit"] - df["Retention"]
    df["LossRatio"] = (df["ExpectedLoss"] / df["Premium"]).round(4)
    df["RetentionToLimit"] = (df["Retention"] / df["Limit"]).round(4)
    df["PremiumRate"] = (df["Premium"] / df["Coverage"]).round(4)

    # Simple risk scoring (can be expanded)
    df["RiskScore"] = (
        0.4 * df["LossRatio"] +
        0.3 * df["RetentionToLimit"] +
        0.3 * df["PremiumRate"]
    ).round(4)

    logging.info("Feature engineering complete.")
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate example labels for MarketLens ML models.
    - acceptance_likelihood: binary 0/1 or probability
    - expected_loss_ratio: continuous target
    """
    # Simulate acceptance likelihood (e.g., higher risk → less likely to accept)
    df["AcceptanceLikelihood"] = np.where(df["RiskScore"] < 0.3, 1, 0)

    # Expected loss ratio is already a target
    df["TargetLossRatio"] = df["LossRatio"]

    logging.info("Labels generated for MarketLens models.")
    return df


def save_processed_data(df: pd.DataFrame, output_dir: str = "data/processed"):
    """Save feature and label datasets for MarketLens."""
    os.makedirs(output_dir, exist_ok=True)

    features = df[[
        "Retention", "Limit", "Premium", "ExpectedLoss",
        "Coverage", "LossRatio", "RetentionToLimit", "PremiumRate", "RiskScore"
    ]]
    labels = df[["AcceptanceLikelihood", "TargetLossRatio"]]

    features_path = os.path.join(output_dir, "marketlens_features.parquet")
    labels_path = os.path.join(output_dir, "marketlens_labels.parquet")

    features.to_parquet(features_path, index=False)
    labels.to_parquet(labels_path, index=False)

    logging.info(f"Saved MarketLens features -> {features_path}")
    logging.info(f"Saved MarketLens labels   -> {labels_path}")


# ---------------------------------------------------------------------
# Main CLI Entry
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean and feature-engineer synthetic treaty data for MarketLens.")
    parser.add_argument("--input", type=str, default="data/processed/treaties_synthetic.csv", help="Path to synthetic treaties CSV/Parquet")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed data")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
    args = parser.parse_args()

    # Setup logger
    setup_logger()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        input_path = config.get("synthetic_treaty_file", args.input)
        output_dir = config.get("processed_output_dir", args.output_dir)
        logging.info(f"Config loaded from {args.config}")
    else:
        input_path = args.input
        output_dir = args.output_dir

    # Pipeline steps
    df = load_synthetic_data(input_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = create_labels(df)
    save_processed_data(df, output_dir=output_dir)

    logging.info("Synthetic treaty processing completed successfully.")


if __name__ == "__main__":
    main()
