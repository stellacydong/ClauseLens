import os
import pandas as pd
import plotly.express as px

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIM_RUNS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "simulation_runs.csv")


def load_policy_traces(n_last: int = 200) -> pd.DataFrame:
    """
    Load recent bids for governance visualization.
    Auto-generates compliance and bid_id if missing.
    """
    if not os.path.exists(SIM_RUNS_PATH):
        raise FileNotFoundError("❌ simulation_runs.csv not found. Run run_simulation.py first.")

    df = pd.read_csv(SIM_RUNS_PATH)
    cols = df.columns.tolist()

    # 1. Normalize column names for consistency
    rename_map = {}
    if "agent_id" in cols:
        rename_map["agent_id"] = "agent"
    if "reward" in cols:
        rename_map["reward"] = "profit"
    if "cvar_95" in cols:
        rename_map["cvar_95"] = "cvar"
    df.rename(columns=rename_map, inplace=True)

    # 2. Generate compliance if missing
    if "compliance" not in df.columns:
        print("⚠️ 'compliance' column not found. Generating proxy compliance...")
        if "cvar" in df.columns:
            # Lower CVaR = safer = higher compliance
            df["compliance"] = 1 - (df["cvar"] / (df["cvar"].max() + 1e-9))
        else:
            # Fallback: normalize profit
            df["compliance"] = (df["profit"] - df["profit"].min()) / (
                df["profit"].max() - df["profit"].min() + 1e-9
            )

    # 3. Generate synthetic bid_id if missing
    if "bid_id" not in df.columns:
        df["bid_id"] = df["episode"].astype(str) + "_" + df["agent"].astype(str) + "_" + df.index.astype(str)

    # 4. Return only the last n_last rows for recent trace visualization
    return df.tail(n_last)


def plot_policy_traces(df: pd.DataFrame):
    """
    Create a scatter plot of Profit vs Compliance for governance visualization.
    """
    if "profit" not in df.columns or "compliance" not in df.columns:
        raise ValueError("Expected columns: profit, compliance")

    fig = px.scatter(
        df,
        x="profit",
        y="compliance",
        color="agent",
        hover_data=["episode", "bid_id"],
        title="Recent Policy Traces: Profit vs Compliance",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    return fig


if __name__ == "__main__":
    df = load_policy_traces(200)
    print(f"✅ Loaded {len(df)} recent bids for policy trace visualization.")
    fig = plot_policy_traces(df)
    fig.show()
