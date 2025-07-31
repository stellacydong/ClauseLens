# app/components/clause_explainer.py
import json
import random
import os
import streamlit as st

class ClauseExplainerComponent:
    def __init__(self, clauses_file="data/clauses_corpus.json", max_clauses=3):
        """
        Initialize the Clause Explainer.
        Args:
            clauses_file: path to the JSON file containing clauses
            max_clauses: maximum clauses to retrieve per treaty
        """
        self.max_clauses = max_clauses
        self.clauses = []
        if os.path.exists(clauses_file):
            with open(clauses_file, "r") as f:
                self.clauses = json.load(f)
        else:
            st.warning(f"⚠️ Clause corpus not found at {clauses_file}")
        
    def retrieve_clauses(self, treaty):
        """
        Retrieve relevant clauses for a treaty. 
        Simple heuristic: match by peril, region, or random fallback.
        """
        if not self.clauses:
            return []

        peril = treaty.get("peril", "").lower()
        region = treaty.get("region", "").lower()

        # Heuristic: filter clauses by category or jurisdiction
        candidates = []
        for clause in self.clauses:
            text = clause["text"].lower()
            if peril in text or region in text:
                candidates.append(clause)

        # Fallback to random clauses if none match
        if not candidates:
            candidates = random.sample(self.clauses, min(self.max_clauses, len(self.clauses)))
        else:
            candidates = random.sample(candidates, min(self.max_clauses, len(candidates)))

        return candidates

    def generate_explanation(self, treaty, clauses):
        """
        Generate a clause-grounded explanation for investors.
        """
        cedent = treaty.get("cedent", "This cedent")
        peril = treaty.get("peril", "the peril")
        region = treaty.get("region", "its region")

        explanation = (
            f"{cedent}'s reinsurance program for {peril} in {region} "
            f"is evaluated against key regulatory and risk guidelines. "
            f"Based on retrieved clauses, the proposed treaty aligns with:\n"
        )

        for clause in clauses:
            explanation += f"- {clause['category']}: {clause['text']}\n"

        return explanation

    def display(self, treaty):
        """
        Streamlit visualization: shows clauses and explanation for the given treaty.
        """
        st.markdown("### 📄 ClauseLens Explanation")

        # Retrieve and display clauses
        clauses = self.retrieve_clauses(treaty)
        explanation = self.generate_explanation(treaty, clauses)

        st.info(explanation)
        st.markdown("#### Retrieved Clauses")
        for clause in clauses:
            st.write(f"**[{clause['category']}]** {clause['text']} *(Jurisdiction: {clause['jurisdiction']})*")

        return clauses, explanation
