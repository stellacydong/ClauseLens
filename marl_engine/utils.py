import os
import pandas as pd

# -----------------------------
# Logging & Result Management
# -----------------------------

def compute_episode_summary(results_df: pd.DataFrame):
    """
    Compute summary KPIs for a single MARL simulation episode.
    Expects columns: ['reward', 'cvar_95']
    Returns: dict of summary metrics
    """
    return {
        "avg_profit": results_df["reward"].mean(),
        "avg_cvar": results_df["cvar_95"].mean(),
        "risk_adj_return": results_df["reward"].mean() / (results_df["cvar_95"].mean() + 1e-6),
        "max_cvar": results_df["cvar_95"].max(),
        "min_profit": results_df["reward"].min(),
        "max_profit": results_df["reward"].max(),
        "n_agents": results_df["agent_id"].nunique() if "agent_id" in results_df.columns else None,
        "episode": results_df["episode"].iloc[0] if "episode" in results_df.columns else None,
    }


def save_results(results_df: pd.DataFrame, filepath: str):
    """
    Save simulation or stress test results to CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"✅ Saved results to {filepath} ({len(results_df)} rows)")


def save_episode_summaries(episode_summaries: list, filepath: str):
    """
    Save a list of per-episode summary dicts as a CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    summary_df = pd.DataFrame(episode_summaries)
    summary_df.to_csv(filepath, index=False)
    print(f"✅ Saved episode summaries to {filepath} ({len(summary_df)} episodes)")
    return summary_df


def load_simulation_results(run_path: str, summary_path: str = None):
    """
    Load simulation runs and (optionally) summary CSVs.
    """
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Simulation results not found at {run_path}")
    results_df = pd.read_csv(run_path)

    summary_df = None
    if summary_path and os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)

    return results_df, summary_df


# -----------------------------
# Utility Functions for YC Demo
# -----------------------------

def compute_dashboard_kpis(summary_df: pd.DataFrame):
    """
    Compute dashboard-ready KPIs and rolling trends.
    """
    df = summary_df.copy()
    if "avg_profit" not in df or "avg_cvar" not in df or "avg_compliance" not in df:
        # Ensure we have the expected columns
        if "avg_compliance" not in df:
            df["avg_compliance"] = 0.85  # Default demo value

    df["profit_growth_pct"] = df["avg_profit"].pct_change().fillna(0) * 100
    df["risk_adj_return"] = df["avg_profit"] / (df["avg_cvar"] + 1e-6)
    df["compliance_delta"] = df["avg_compliance"].diff().fillna(0)

    # Rolling trends
    df["rolling_profit"] = df["avg_profit"].rolling(3, min_periods=1).mean()
    df["rolling_cvar"] = df["avg_cvar"].rolling(3, min_periods=1).mean()
    df["rolling_compliance"] = df["avg_compliance"].rolling(3, min_periods=1).mean()

    return df
