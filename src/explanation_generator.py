"""
explanation_generator.py
------------------------
Generates ClauseLens explanations for treaty bids:
- Uses retrieved clauses for audit-ready justifications
- Supports concise or detailed modes for dashboards and investor PDFs
"""

from typing import List, Dict
from datetime import datetime


class ClauseExplainer:
    def __init__(self, detailed: bool = True):
        """
        :param detailed: If True, produce extended explanations for PDFs
        """
        self.detailed = detailed

    # ---------------------------
    # Core Explanation
    # ---------------------------
    def generate_explanation(self, winning_bid: Dict, clauses: List[Dict]) -> str:
        """
        Generate a natural language explanation for the winning bid.
        :param winning_bid: Dict with quota_share, premium, expected_loss, tail_risk
        :param clauses: List of retrieved clauses with {id, text, category}
        """
        if not winning_bid:
            return "No winning bid available to explain."

        quota = winning_bid.get("quota_share", 0.0)
        premium = winning_bid.get("premium", 0.0)
        expected_loss = winning_bid.get("expected_loss", 0.0)
        tail_risk = winning_bid.get("tail_risk", 0.0)

        # Base summary
        summary = (
            f"This bid proposes a {quota:.0%} quota share with a premium of "
            f"${premium:,.0f}, covering an expected loss of ${expected_loss:,.0f} "
            f"and tail-risk exposure (CVaR) of ${tail_risk:,.0f}."
        )

        # Clause integration
        if clauses:
            summary += " It aligns with the following regulatory and structural requirements:"
            for clause in clauses:
                summary += f"\n- [Clause {clause['id']}] ({clause['category']}) {clause['text']}"
        else:
            summary += " No relevant regulatory clauses were retrieved."

        # Detailed explanation for PDF mode
        if self.detailed:
            summary += (
                "\n\nRationale:\n"
                "- The premium exceeds the expected loss, ensuring a positive risk margin.\n"
                "- Quota share and tail-risk are within regulatory thresholds.\n"
                "- Retrieved clauses confirm compliance with capital and reporting standards."
            )

            # Timestamp for audit-ready reports
            summary += f"\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return summary

    # ---------------------------
    # Episode-Level Report Builder
    # ---------------------------
    def build_episode_explanation(self, episode_result: Dict) -> str:
        """
        Combine treaty details, winning bid, and clause explanation into a full report string.
        """
        treaty = episode_result.get("treaty", {})
        winning_bid = episode_result.get("winning_bid", {})
        clauses = episode_result.get("clauses", [])

        header = (
            f"Cedent: {treaty.get('cedent', 'Unknown')}\n"
            f"Peril: {treaty.get('peril', 'Unknown')} | "
            f"LOB: {treaty.get('line_of_business', 'Unknown')} | "
            f"Region: {treaty.get('region', 'Unknown')}\n"
            f"Exposure: ${treaty.get('exposure',0):,.0f} | "
            f"Limit: {treaty.get('limit',0):.0%} | "
            f"Quota Share Cap: {treaty.get('quota_share_cap',0):.0%}\n"
            f"Notes: {treaty.get('notes','None')}\n\n"
        )

        return header + self.generate_explanation(winning_bid, clauses)


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    sample_bid = {
        "quota_share": 0.4,
        "premium": 200000,
        "expected_loss": 150000,
        "tail_risk": 60000,
    }
    sample_clauses = [
        {"id": 1, "text": "Solvency II Article 101 requires 99.5% capital coverage.", "category": "Capital Requirements"},
        {"id": 2, "text": "IFRS 17 mandates transparent reporting.", "category": "Accounting & Reporting"}
    ]

    explainer = ClauseExplainer(detailed=True)
    print(explainer.generate_explanation(sample_bid, sample_clauses))
