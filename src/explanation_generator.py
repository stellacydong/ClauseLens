# src/explanation_generator.py
import random

class ClauseExplainer:
    """
    Generates ClauseLens explanations for winning bids using retrieved clauses.
    Produces a narrative highlighting compliance with regulatory and risk requirements.
    """

    def __init__(self):
        pass

    def generate_explanation(self, winning_bid, retrieved_clauses):
        """
        Generate a human-readable explanation using the winning bid and retrieved clauses.

        Args:
            winning_bid (dict): MARL agent's winning bid (premium, quota_share, expected_loss, tail_risk)
            retrieved_clauses (list[dict]): Clauses from ClauseRetriever with
                                           fields id, text, category, jurisdiction, score

        Returns:
            explanation (str): Narrative explanation for PDF and Streamlit display
        """

        if not winning_bid:
            return "No winning bid information available."

        if not retrieved_clauses:
            return (
                f"The winning bid offers a quota share of {winning_bid.get('quota_share', 0):.0%} "
                f"with a premium of ${winning_bid.get('premium', 0):,.0f}, "
                "but no relevant clauses were retrieved for this treaty."
            )

        # Pick top 3 clauses for explanation
        top_clauses = retrieved_clauses[:3]

        # Create a narrative
        explanation_parts = [
            f"The winning bid proposes a quota share of {winning_bid.get('quota_share', 0):.0%} "
            f"with a premium of ${winning_bid.get('premium', 0):,.0f}. "
            f"This aligns with regulatory and capital requirements as follows:"
        ]

        for clause in top_clauses:
            explanation_parts.append(
                f"- **{clause['category']} ({clause['jurisdiction']})**: {clause['text']}"
            )

        explanation_parts.append(
            "This structure supports risk transfer and regulatory compliance, "
            "balancing profitability and tail-risk exposure."
        )

        explanation = "\n".join(explanation_parts)
        return explanation


# ---------------------------
# Quick Test (Optional)
# ---------------------------
if __name__ == "__main__":
    # Sample winning bid
    winning_bid = {
        "quota_share": 0.5,
        "premium": 1_200_000,
        "expected_loss": 950_000,
        "tail_risk": 300_000
    }

    # Sample retrieved clauses
    retrieved_clauses = [
        {
            "id": 1,
            "text": "Solvency II Article 101 requires that capital be sufficient to cover 99.5% of annual risk.",
            "category": "Capital Adequacy",
            "jurisdiction": "EU",
            "score": 0.87
        },
        {
            "id": 2,
            "text": "IFRS 17 mandates transparent reporting of expected losses and reinsurance recoverables.",
            "category": "Accounting & Reporting",
            "jurisdiction": "Global",
            "score": 0.83
        },
        {
            "id": 3,
            "text": "Excess-of-loss treaties require reinsurer participation to be well diversified.",
            "category": "Diversification",
            "jurisdiction": "Global",
            "score": 0.79
        }
    ]

    explainer = ClauseExplainer()
    explanation = explainer.generate_explanation(winning_bid, retrieved_clauses)
    print("Generated Explanation:\n")
    print(explanation)
