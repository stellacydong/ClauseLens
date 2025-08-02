import numpy as np
import pandas as pd

def catastrophe_shock(profit, severity: float = 0.5):
    """
    Simulate a catastrophe event that reduces profit.
    severity: fraction of profit lost (0.5 = 50% loss)
    """
    return profit * (1 - severity)


def capital_squeeze(cvar_95, multiplier: float = 1.3):
    """
    Simulate tighter capital or regulatory conditions:
    Increase CVaR exposure by multiplier.
    """
    return cvar_95 * multiplier


def market_downturn(profit, volatility: float = 0.3):
    """
    Simulate market downturn with random negative shock.
    volatility: standard deviation for Gaussian drop.
    """
    shock = np.random.normal(loc=-0.1, scale=volatility)
    return profit * (1 + shock)


def run_stress_tests(results_df: pd.DataFrame):
    """
    Apply multiple stress scenarios to simulation results.
    Input: DataFrame with columns [reward, cvar_95]
    Returns: DataFrame with stressed metrics
    """
    stress_df = results_df.copy()

    # Scenario 1: Catastrophe Shock (50% profit loss)
    stress_df["reward_cat"] = stress_df["reward"].apply(lambda x: catastrophe_shock(x, 0.5))

    # Scenario 2: Capital Squeeze (CVaR +30%)
    stress_df["cvar_squeeze"] = stress_df["cvar_95"].apply(lambda x: capital_squeeze(x, 1.3))

    # Scenario 3: Market Downturn
    stress_df["reward_downturn"] = stress_df["reward"].apply(lambda x: market_downturn(x, 0.3))

    # Compute Risk-Adjusted Return under stress
    stress_df["risk_adj_return"] = stress_df["reward_cat"] / (stress_df["cvar_squeeze"] + 1e-6)

    return stress_df


def summarize_stress_results(stress_df: pd.DataFrame):
    """
    Aggregate stress test results for dashboard or reporting.
    """
    summary = {
        "mean_reward_post_cat": stress_df["reward_cat"].mean(),
        "mean_reward_downturn": stress_df["reward_downturn"].mean(),
        "mean_cvar_squeezed": stress_df["cvar_squeeze"].mean(),
        "mean_risk_adj_return": stress_df["risk_adj_return"].mean(),
        "max_cvar_squeezed": stress_df["cvar_squeeze"].max(),
        "episodes": stress_df["episode"].nunique() if "episode" in stress_df.columns else None
    }
    return summary
