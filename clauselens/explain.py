from typing import List, Dict

class ClauseExplainer:
    """
    Generates natural language explanations for treaty bids using retrieved clauses.
    """

    def __init__(self):
        # Optional: add template library or language model integration here
        self.template = (
            "This quote of ${bid_value:,.0f} for {lob} (region: {region}) "
            "is aligned with expected risk and regulatory thresholds. "
            "Supporting clauses: {clauses}"
        )

    def explain_quote(self, treaty_features: Dict, clauses: List[Dict], bid_value: float) -> str:
        """
        Generates a human-readable explanation for a quote.
        
        Args:
            treaty_features: dict of treaty info (line_of_business, region, layer, etc.)
            clauses: list of top clause dicts (must contain 'clause_text')
            bid_value: proposed quote or bid value
            
        Returns:
            Explanation string
        """
        lob = treaty_features.get("line_of_business", "Unknown")
        region = treaty_features.get("region", "Global")
        clause_texts = "; ".join([c.get("clause_text", "Unknown clause") for c in clauses])

        return self.template.format(
            bid_value=bid_value,
            lob=lob,
            region=region,
            clauses=clause_texts
        )

    def explain_with_risk(self, treaty_features: Dict, clauses: List[Dict], bid_value: float,
                          cvar_95: float, risk_adj_return: float) -> str:
        """
        Generates an explanation including CVaR and risk-adjusted metrics.
        Useful for governance/audit dashboards.
        """
        base_explanation = self.explain_quote(treaty_features, clauses, bid_value)
        risk_info = (
            f" CVaR 95% exposure is ${cvar_95:,.0f} "
            f"with a risk-adjusted return of {risk_adj_return:.2f}."
        )
        return base_explanation + risk_info


# -----------------------------
# Demo Usage
# -----------------------------
if __name__ == "__main__":
    # Example treaty
    treaty = {
        "line_of_business": "Property",
        "region": "EU",
        "layer": "5M xs 5M"
    }

    # Example retrieved clauses
    clauses = [
        {"clause_id": 1, "clause_text": "Solvency II Article 138 on capital requirements"},
        {"clause_id": 2, "clause_text": "NAIC RBC Property Risk Charge"},
    ]

    explainer = ClauseExplainer()

    explanation = explainer.explain_quote(treaty, clauses, bid_value=5_000_000)
    print("Simple Explanation:\n", explanation)

    explanation_risk = explainer.explain_with_risk(treaty, clauses, bid_value=5_000_000,
                                                  cvar_95=3_200_000, risk_adj_return=1.45)
    print("\nExplanation with Risk Metrics:\n", explanation_risk)
