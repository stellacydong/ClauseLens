import random
import json
import os

class ClauseExplainerComponent:
    """
    Clause Explainer Component for ClauseLens Demo.
    Retrieves sample regulatory clauses and generates human-readable explanations.
    """

    def __init__(self, clause_file="data/clauses_corpus.json", seed=None):
        self.clauses = []
        if os.path.exists(clause_file):
            with open(clause_file) as f:
                self.clauses = json.load(f)
        else:
            # Fallback synthetic clauses
            self.clauses = [
                {"id": 1, "text": "Solvency II Article 101 requires sufficient capital coverage.", "category": "Solvency"},
                {"id": 2, "text": "IFRS 17 mandates transparency in reporting expected losses.", "category": "Reporting"},
                {"id": 3, "text": "NAIC RBC requires tail-risk stress testing for catastrophe treaties.", "category": "Risk"},
            ]
        if seed is not None:
            random.seed(seed)

    def retrieve_clauses(self, treaty, n=3):
        """
        Retrieve relevant clauses for a treaty.
        Currently random for demo purposes.
        """
        if not self.clauses:
            return []
        return random.sample(self.clauses, min(n, len(self.clauses)))

    def generate_explanation(self, treaty, winning_bid, clauses):
        """
        Generate a simple ClauseLens explanation for the winning bid.
        """
        if not clauses:
            return "No clauses retrieved for this treaty."

        # Example logic: tie clauses to treaty features
        explanation_lines = [
            f"This quote for {treaty.get('cedent','Unknown')} ({treaty.get('peril','N/A')}) "
            f"complies with key capital and reporting requirements."
        ]
        for clause in clauses:
            explanation_lines.append(f"- {clause['text']}")

        # Tail-risk mention if applicable
        if winning_bid.get("tail_risk", 0) > 0.25 * winning_bid.get("premium", 1):
            explanation_lines.append("⚠️ High tail-risk component; review Solvency compliance.")

        return "\n".join(explanation_lines)

    def explain_treaty(self, treaty, winning_bid, n=3):
        """
        Convenience function to retrieve clauses and generate explanation in one step.
        Returns (clauses, explanation).
        """
        clauses = self.retrieve_clauses(treaty, n=n)
        explanation = self.generate_explanation(treaty, winning_bid, clauses)
        return clauses, explanation
