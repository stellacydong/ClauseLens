#!/usr/bin/env python3
"""
01_generate_synthetic_treaties.py

Generates a synthetic treaty dataset for the Transparent Market Platform.
- Creates treaty structures, cedent info, and basic risk profiles.
- Saves output as CSV and optional Parquet for downstream MarketLens & MARL pipeline.
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

def setup_logger(log_dir: str = "outputs/logs", script_name: str = "01_generate_synthetic_treaties"):
    """Set up a rotating logger for reproducibility."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Also print to console
    logging.info(f"Logger initialized. Output -> {log_file}")


def generate_synthetic_treaties(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic treaty data.
    Columns include: TreatyID, Cedent, Reinsurer, Retention, Limit, Premium, ExpectedLoss, Region.
    """
    np.random.seed(random_state)

    # Example treaty fields
    treaty_ids = [f"T{100000 + i}" for i in range(n_samples)]
    cedents = np.random.choice(["CedentA", "CedentB", "CedentC", "CedentD"], size=n_samples)
    reinsurers = np.random.choice(["ReinsurerX", "ReinsurerY", "ReinsurerZ"], size=n_samples)
    regions = np.random.choice(["NA", "EU", "APAC"], size=n_samples)

    retention = np.random.uniform(1_000_000, 10_000_000, size=n_samples)
    limit = retention + np.random.uniform(5_000_000, 50_000_000, size=n_samples)
    premium = (limit - retention) * np.random.uniform(0.01, 0.05, size=n_samples)
    expected_loss = (limit - retention) * np.random.uniform(0.005, 0.03, size=n_samples)

    treaties_df = pd.DataFrame({
        "TreatyID": treaty_ids,
        "Cedent": cedents,
        "Reinsurer": reinsurers,
        "Region": regions,
        "Retention": retention.round(2),
        "Limit": limit.round(2),
        "Premium": premium.round(2),
        "ExpectedLoss": expected_loss.round(2),
    })

    logging.info(f"Generated synthetic dataset with {len(treaties_df)} treaties.")
    return treaties_df


def save_synthetic_data(df: pd.DataFrame, output_dir: str = "data/processed"):
    """Save the synthetic data to CSV and Parquet formats."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "treaties_synthetic.csv")
    parquet_path = os.path.join(output_dir, "treaties_synthetic.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logging.info(f"Saved synthetic treaties -> {csv_path} and {parquet_path}")


# ---------------------------------------------------------------------
# Main CLI Entry
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic treaty dataset for MARL and MarketLens.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic treaties to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save synthetic data")
    args = parser.parse_args()

    # Setup logger
    setup_logger()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        n_samples = config.get("synthetic_treaty_samples", args.samples)
        seed = config.get("random_seed", args.seed)
        output_dir = config.get("output_dir", args.output_dir)
        logging.info(f"Config loaded from {args.config}")
    else:
        n_samples = args.samples
        seed = args.seed
        output_dir = args.output_dir

    # Generate & Save
    df = generate_synthetic_treaties(n_samples=n_samples, random_state=seed)
    save_synthetic_data(df, output_dir=output_dir)
    logging.info("Synthetic treaty generation completed successfully.")


if __name__ == "__main__":
    main()
