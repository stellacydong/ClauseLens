# src/clause_retrieval.py
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src import config

class ClauseRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=5):
        """
        Initialize ClauseRetriever with a SentenceTransformer model
        and pre-load the clauses corpus.
        """
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)

        # Load clauses
        with open(config.CLAUSE_CORPUS_PATH, "r") as f:
            self.clauses = json.load(f)

        # Encode all clauses
        self.embeddings = self.model.encode([c["text"] for c in self.clauses], show_progress_bar=False)

    def retrieve(self, treaty, category_filter=None, jurisdiction_filter=None):
        """
        Retrieve top-k clauses most relevant to the treaty description.
        Optionally filter by category or jurisdiction.
        """
        # Build treaty query string
        treaty_desc = (
            f"Peril: {treaty.get('peril','')} | "
            f"Region: {treaty.get('region','')} | "
            f"Line: {treaty.get('line_of_business','')} | "
            f"Exposure: {treaty.get('exposure','')} | "
            f"Limit: {treaty.get('limit','')}"
        )

        # Compute treaty embedding
        treaty_emb = self.model.encode([treaty_desc], show_progress_bar=False)

        # Filter clauses if requested
        filtered_clauses = self.clauses
        filtered_embeddings = self.embeddings
        if category_filter:
            filtered_clauses = [c for c in filtered_clauses if c.get("category") == category_filter]
        if jurisdiction_filter:
            filtered_clauses = [c for c in filtered_clauses if c.get("jurisdiction") == jurisdiction_filter]

        if category_filter or jurisdiction_filter:
            # Rebuild embeddings for the filtered subset
            filtered_embeddings = self.model.encode([c["text"] for c in filtered_clauses], show_progress_bar=False)

        # Compute cosine similarity
        sims = cosine_similarity(treaty_emb, filtered_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][: self.top_k]

        # Return top-k clause dicts with similarity score
        results = []
        for idx in top_indices:
            clause = filtered_clauses[idx]
            clause_copy = {
                "id": clause["id"],
                "text": clause["text"],
                "category": clause.get("category", "General"),
                "jurisdiction": clause.get("jurisdiction", "Global"),
                "score": float(sims[idx])
            }
            results.append(clause_copy)

        return results


# ---------------------------
# Quick Test (Optional)
# ---------------------------
if __name__ == "__main__":
    sample_treaty = {
        "peril": "Hurricane",
        "region": "Florida",
        "line_of_business": "Property Catastrophe",
        "exposure": 500_000_000,
        "limit": 0.3
    }

    retriever = ClauseRetriever(top_k=5)
    results = retriever.retrieve(sample_treaty, category_filter=None)

    print("Retrieved Clauses:")
    for r in results:
        print(f"- [{r['category']}/{r['jurisdiction']}] {r['text']} (score={r['score']:.3f})")
