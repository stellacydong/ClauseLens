# src/evaluation.py
import numpy as np

def evaluate_bids(bids, winner_idx, treaty, retrieved_clauses=None):
    """
    Evaluate MARL or baseline bids for profit, CVaR, and regulatory compliance.
    Optionally attach ClauseLens clause IDs for justification.

    Args:
        bids (list[dict]): Bids, each with keys:
            - 'premium', 'expected_loss', 'tail_risk', 'quota_share'
        winner_idx (int): Index of the winning bid
        treaty (dict): Selected treaty details
        retrieved_clauses (list[dict], optional): ClauseLens clauses from ClauseRetriever

    Returns:
        dict: KPIs including profit, CVaR, compliance flags, and clause justification
    """
    if not bids:
        return {
            "winner": None,
            "profit": 0,
            "cvar": 0,
            "regulatory_flags": {"all_ok": False},
            "clauses_used": []
        }

    winning_bid = bids[winner_idx]
    expected_loss = winning_bid.get("expected_loss", 0)
    premium = winning_bid.get("premium", 0)
    tail_risk = winning_bid.get("tail_risk", expected_loss * 0.3)
    quota_share = winning_bid.get("quota_share", 0.0)

    # Compute basic KPIs
    profit = premium - expected_loss
    cvar = expected_loss + tail_risk  # Simple CVaR proxy

    # Compliance rules
    compliance_flags = {
        "quota_share_ok": quota_share <= treaty.get("quota_share_cap", 1.0),
        "premium_ok": premium >= expected_loss * 1.05,  # min 5% margin
        "tail_risk_ok": cvar <= treaty.get("exposure", 0) * 0.5  # simple stress limit
    }
    compliance_flags["all_ok"] = all(compliance_flags.values())

    # Collect clause IDs if provided
    clause_ids = [c["id"] for c in retrieved_clauses] if retrieved_clauses else []

    return {
        "winner": winner_idx,
        "winning_bid": winning_bid,
        "profit": profit,
        "cvar": cvar,
        "regulatory_flags": compliance_flags,
        "clauses_used": clause_ids
    }


def summarize_portfolio(episode_results):
    """
    Summarize portfolio performance over multiple episodes.

    Args:
        episode_results (list[dict]): Results from evaluate_bids()

    Returns:
        dict: Portfolio KPIs including averages and compliance rate
    """
    if not episode_results:
        return {"avg_profit": 0, "avg_cvar": 0, "compliance_rate": 0.0, "episodes": 0}

    profits = [ep["profit"] for ep in episode_results]
    cvars = [ep["cvar"] for ep in episode_results]
    compliance = [ep["regulatory_flags"]["all_ok"] for ep in episode_results]

    return {
        "avg_profit": float(np.mean(profits)),
        "avg_cvar": float(np.mean(cvars)),
        "compliance_rate": float(np.mean(compliance)),
        "episodes": len(episode_results)
    }


# ---------------------------
# Quick Test (Optional)
# ---------------------------
if __name__ == "__main__":
    sample_treaty = {
        "peril": "Hurricane",
        "region": "Florida",
        "line_of_business": "Property Catastrophe",
        "exposure": 500_000_000,
        "limit": 0.3,
        "quota_share_cap": 0.6
    }

    sample_bids = [
        {"premium": 1_200_000, "expected_loss": 950_000, "tail_risk": 300_000, "quota_share": 0.55},
        {"premium": 1_100_000, "expected_loss": 900_000, "tail_risk": 350_000, "quota_share": 0.45},
        {"premium": 1_050_000, "expected_loss": 870_000, "tail_risk": 320_000, "quota_share": 0.50},
    ]

    sample_clauses = [
        {"id": 1, "text": "Solvency II capital must cover 99.5% annual risk."},
        {"id": 3, "text": "IFRS 17 mandates transparent expected loss reporting."}
    ]

    episode_kpi = evaluate_bids(sample_bids, winner_idx=0, treaty=sample_treaty, retrieved_clauses=sample_clauses)
    print("Episode KPIs:", episode_kpi)

    portfolio_summary = summarize_portfolio([episode_kpi])
    print("Portfolio Summary:", portfolio_summary)
