"""
evaluation.py
-------------
Functions to evaluate treaty bids for ClauseLens:
- Profit and CVaR (tail risk)
- Regulatory compliance checks
- Portfolio summary for dashboards and investor reporting
"""

import numpy as np
from typing import List, Dict


# ---------------------------
# Episode-Level Evaluation
# ---------------------------
def evaluate_bids(bids: List[Dict], winner_idx: int, treaty: Dict) -> Dict:
    """
    Evaluate a list of bids for a treaty and return KPIs for the winning bid.

    :param bids: List of bid dicts [{quota_share, premium, expected_loss, tail_risk}]
    :param winner_idx: Index of winning bid
    :param treaty: The treaty dictionary
    :return: KPI dict {profit, cvar, regulatory_flags}
    """
    winning_bid = bids[winner_idx]

    premium = winning_bid["premium"]
    expected_loss = winning_bid["expected_loss"]
    tail_risk = winning_bid.get("tail_risk", expected_loss * 0.3)
    quota_share = winning_bid.get("quota_share", 0.3)

    # Profit: premium minus expected loss
    profit = premium - expected_loss

    # CVaR: simulate tail risk exposure (simplified)
    cvar = tail_risk

    # Regulatory compliance checks
    regulatory_flags = {
        "quota_share_ok": quota_share <= treaty.get("quota_share_cap", 1.0),
        "premium_ok": premium > expected_loss,
        "tail_risk_ok": cvar <= treaty.get("exposure", 1e6) * 0.5,
    }
    regulatory_flags["all_ok"] = all(regulatory_flags.values())

    return {
        "profit": profit,
        "cvar": cvar,
        "regulatory_flags": regulatory_flags,
    }


# ---------------------------
# Portfolio Summary
# ---------------------------
def summarize_portfolio(episode_results: List[Dict]) -> Dict:
    """
    Summarize portfolio KPIs across multiple episodes.

    :param episode_results: List of KPI dicts (output of evaluate_bids)
    :return: Dict with average profit, CVaR, compliance rate
    """
    if not episode_results:
        return {"avg_profit": 0, "avg_cvar": 0, "compliance_rate": 0, "episodes": 0}

    profits = [ep["profit"] for ep in episode_results]
    cvars = [ep["cvar"] for ep in episode_results]
    compliance = [1 if ep["regulatory_flags"]["all_ok"] else 0 for ep in episode_results]

    return {
        "episodes": len(episode_results),
        "avg_profit": float(np.mean(profits)),
        "avg_cvar": float(np.mean(cvars)),
        "compliance_rate": float(np.mean(compliance)),
    }


# ---------------------------
# Portfolio Stress Evaluation
# ---------------------------
def stress_test_portfolio(episode_results: List[Dict], stress_factor: float = 1.5) -> Dict:
    """
    Apply a simple stress factor to CVaR and evaluate impact on compliance.

    :param episode_results: List of KPI dicts
    :param stress_factor: Multiplier for tail risk
    :return: Dict with stressed portfolio summary
    """
    if not episode_results:
        return {"avg_profit": 0, "avg_cvar": 0, "compliance_rate": 0}

    stressed_results = []
    for ep in episode_results:
        stressed_cvar = ep["cvar"] * stress_factor
        flags = ep["regulatory_flags"].copy()
        flags["tail_risk_ok"] = stressed_cvar <= ep["cvar"] * 1.5  # simulate stricter test
        flags["all_ok"] = all(flags.values())
        stressed_results.append({
            "profit": ep["profit"],
            "cvar": stressed_cvar,
            "regulatory_flags": flags,
        })

    return summarize_portfolio(stressed_results)


# ---------------------------
# Episode Table Builder
# ---------------------------
def build_episode_table(marl_results: List[Dict], base_results: List[Dict]) -> List[Dict]:
    """
    Combine MARL and Baseline KPI results into a single table-friendly list.
    """
    table = []
    for i, (marl, base) in enumerate(zip(marl_results, base_results)):
        table.append({
            "Episode": i + 1,
            "MARL Profit ($)": f"{marl['profit']:,.0f}",
            "MARL CVaR ($)": f"{marl['cvar']:,.0f}",
            "MARL Compliance": "Pass" if marl['regulatory_flags']['all_ok'] else "Fail",
            "Baseline Profit ($)": f"{base['profit']:,.0f}",
            "Baseline CVaR ($)": f"{base['cvar']:,.0f}",
            "Baseline Compliance": "Pass" if base['regulatory_flags']['all_ok'] else "Fail",
        })
    return table


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    dummy_treaty = {
        "exposure": 5_000_000,
        "quota_share_cap": 0.5,
    }
    dummy_bids = [
        {"quota_share": 0.3, "premium": 150000, "expected_loss": 100000, "tail_risk": 40000},
        {"quota_share": 0.4, "premium": 200000, "expected_loss": 180000, "tail_risk": 60000},
    ]
    kpi = evaluate_bids(dummy_bids, winner_idx=0, treaty=dummy_treaty)
    print("Episode KPI:", kpi)
    portfolio_summary = summarize_portfolio([kpi])
    print("Portfolio Summary:", portfolio_summary)
