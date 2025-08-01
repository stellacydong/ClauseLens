import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIM_RUNS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "simulation_runs.csv")
OVERRIDE_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "override_log.csv")


def find_high_risk_bids(threshold_compliance: float = 0.6):
    """Return bids with compliance below threshold for manual review."""
    if not os.path.exists(SIM_RUNS_PATH):
        raise FileNotFoundError("❌ simulation_runs.csv not found. Run run_simulation.py first.")
    
    df = pd.read_csv(SIM_RUNS_PATH)
    cols = df.columns.tolist()

    # Normalize column names for easier use
    df.rename(columns={
        "agent_id": "agent",
        "reward": "profit",
        "cvar_95": "cvar"
    }, inplace=True)

    # 1. Generate compliance if missing
    if "compliance" not in df.columns:
        print("⚠️ 'compliance' column not found. Generating proxy compliance...")
        if "cvar" in df.columns:
            df["compliance"] = 1 - (df["cvar"] / (df["cvar"].max() + 1e-9))
        else:
            df["compliance"] = (df["profit"] - df["profit"].min()) / (
                df["profit"].max() - df["profit"].min() + 1e-9
            )

    # 2. Generate synthetic bid_id if missing
    if "bid_id" not in df.columns:
        df["bid_id"] = df["episode"].astype(str) + "_" + df["agent"].astype(str) + "_" + df.index.astype(str)

    risky_bids = df[df["compliance"] < threshold_compliance]
    return risky_bids


def override_policy(bid_ids, reason="Manual override due to risk"):
    """Log a manual override for selected bid IDs."""
    logs = [{"bid_id": bid, "override_reason": reason} for bid in bid_ids]
    
    log_df = pd.DataFrame(logs)
    if os.path.exists(OVERRIDE_LOG_PATH):
        existing = pd.read_csv(OVERRIDE_LOG_PATH)
        log_df = pd.concat([existing, log_df], ignore_index=True)

    log_df.to_csv(OVERRIDE_LOG_PATH, index=False)
    print(f"✅ Overrides logged for {len(bid_ids)} bids at {OVERRIDE_LOG_PATH}")


if __name__ == "__main__":
    risky = find_high_risk_bids()
    print(f"Found {len(risky)} high-risk bids (compliance < 0.6)")
    if not risky.empty:
        override_policy(risky["bid_id"].head(5), reason="Test override for governance demo")
