"""
visualization.py
----------------
Visualization utilities for ClauseLens:
- Profit vs CVaR scatter plots
- Portfolio KPI bar charts
- Multi-episode results tables for Streamlit dashboards
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional

# ---------------------------
# Scatter Plot: Profit vs CVaR
# ---------------------------
def plot_profit_vs_cvar(
    marl_results: List[Dict],
    baseline_results: List[Dict],
    title: str = "Profit vs Tail-Risk (CVaR)",
    save_path: Optional[str] = None
):
    """
    Create a scatter plot comparing MARL and Baseline profit vs CVaR.
    """
    marl_profits = [res["profit"] for res in marl_results]
    marl_cvar = [res["cvar"] for res in marl_results]
    base_profits = [res["profit"] for res in baseline_results]
    base_cvar = [res["cvar"] for res in baseline_results]

    plt.figure(figsize=(7, 5))
    plt.scatter(base_cvar, base_profits, color="red", label="Baseline")
    plt.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")
    plt.xlabel("CVaR (Tail Risk $)")
    plt.ylabel("Profit ($)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()


# ---------------------------
# Bar Chart: Portfolio KPIs
# ---------------------------
def plot_portfolio_summary(summary_marl: Dict, summary_baseline: Dict, save_path: Optional[str] = None):
    """
    Plot a side-by-side bar chart for Avg Profit, Avg CVaR, and Compliance Rate.
    """
    metrics = ["Avg Profit ($)", "Avg CVaR ($)", "Compliance Rate (%)"]
    marl_values = [
        summary_marl.get("avg_profit", 0),
        summary_marl.get("avg_cvar", 0),
        summary_marl.get("compliance_rate", 0) * 100
    ]
    baseline_values = [
        summary_baseline.get("avg_profit", 0),
        summary_baseline.get("avg_cvar", 0),
        summary_baseline.get("compliance_rate", 0) * 100
    ]

    x = range(len(metrics))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], marl_values, width=0.4, color="green", label="MARL Agents")
    plt.bar([i + 0.2 for i in x], baseline_values, width=0.4, color="red", label="Baseline")
    plt.xticks(x, metrics)
    plt.title("Portfolio KPI Comparison")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()


# ---------------------------
# Episode Table Builder
# ---------------------------
def build_episode_table(marl_results: List[Dict], baseline_results: List[Dict]) -> pd.DataFrame:
    """
    Convert MARL and Baseline episode results into a pandas DataFrame for dashboards.
    """
    data = []
    for i, (marl, base) in enumerate(zip(marl_results, baseline_results), start=1):
        data.append({
            "Episode": i,
            "MARL Profit ($)": f"{marl['profit']:,.0f}",
            "MARL CVaR ($)": f"{marl['cvar']:,.0f}",
            "MARL Compliance": "Pass" if marl['regulatory_flags']['all_ok'] else "Fail",
            "Baseline Profit ($)": f"{base['profit']:,.0f}",
            "Baseline CVaR ($)": f"{base['cvar']:,.0f}",
            "Baseline Compliance": "Pass" if base['regulatory_flags']['all_ok'] else "Fail",
        })

    return pd.DataFrame(data)


# ---------------------------
# Multi-Figure Dashboard Export
# ---------------------------
def generate_dashboard_figures(
    marl_results: List[Dict],
    baseline_results: List[Dict],
    summary_marl: Dict,
    summary_baseline: Dict,
    save_folder: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate all visualizations for dashboards and optionally save to files.
    Returns dict of paths if save_folder is provided.
    """
    paths = {}
    if save_folder:
        import os
        os.makedirs(save_folder, exist_ok=True)
        scatter_path = f"{save_folder}/profit_vs_cvar.png"
        bar_path = f"{save_folder}/portfolio_summary.png"
        plot_profit_vs_cvar(marl_results, baseline_results, save_path=scatter_path)
        plot_portfolio_summary(summary_marl, summary_baseline, save_path=bar_path)
        paths["scatter"] = scatter_path
        paths["bar"] = bar_path
    return paths


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    dummy_marl = [
        {"profit": 120_000, "cvar": 80_000, "regulatory_flags": {"all_ok": True}},
        {"profit": 150_000, "cvar": 90_000, "regulatory_flags": {"all_ok": True}},
    ]
    dummy_baseline = [
        {"profit": 100_000, "cvar": 110_000, "regulatory_flags": {"all_ok": False}},
        {"profit": 95_000, "cvar": 120_000, "regulatory_flags": {"all_ok": False}},
    ]
    table = build_episode_table(dummy_marl, dummy_baseline)
    print(table)
    plot_profit_vs_cvar(dummy_marl, dummy_baseline, save_path="profit_vs_cvar_test.png")
    plot_portfolio_summary(
        {"avg_profit": 135_000, "avg_cvar": 85_000, "compliance_rate": 1.0},
        {"avg_profit": 97_500, "avg_cvar": 115_000, "compliance_rate": 0.0},
        save_path="portfolio_summary_test.png"
    )
    print("✅ Test charts generated.")
