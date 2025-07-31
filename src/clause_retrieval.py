"""
clause_retrieval.py
-------------------
ClauseLens retrieval system for reinsurance treaty analysis.
- Loads and caches regulatory clauses
- Provides keyword or semantic retrieval
- Returns structured clauses for audit-ready reporting
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from src import config
from src.utils import normalize_text

try:
    from sentence_transformers import SentenceTransformer, util
    USE_SEMANTIC = True
except ImportError:
    USE_SEMANTIC = False


class ClauseRetriever:
    def __init__(self, clause_path: Optional[str] = None, top_k: int = 3):
        """
        Initialize ClauseRetriever
        :param clause_path: Path to clause JSON file
        :param top_k: Number of top clauses to return
        """
        self.clause_path = clause_path or config.CLAUSES_PATH
        self.top_k = top_k
        self.clauses = self._load_clauses()
        self._embeddings = None

        # Load semantic model if available
        if USE_SEMANTIC:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._precompute_embeddings()
        else:
            self.model = None

    # ---------------------------
    # Internal Functions
    # ---------------------------
    def _load_clauses(self) -> List[Dict]:
        """Load clauses from JSON file with caching"""
        if not os.path.exists(self.clause_path):
            raise FileNotFoundError(f"Clause file not found: {self.clause_path}")

        with open(self.clause_path, "r") as f:
            clauses = json.load(f)

        # Normalize and assign categories if missing
        for c in clauses:
            c["text"] = normalize_text(c.get("text", ""))
            if "category" not in c:
                c["category"] = self._infer_category(c["text"])
        return clauses

    def _infer_category(self, text: str) -> str:
        """Simple heuristic to categorize clauses"""
        text_lower = text.lower()
        if "solvency" in text_lower:
            return "Capital Requirements"
        elif "ifrs" in text_lower:
            return "Accounting & Reporting"
        elif "naic" in text_lower or "rbc" in text_lower:
            return "Risk-Based Capital"
        elif "diversified" in text_lower:
            return "Reinsurance Structure"
        return "General Compliance"

    def _precompute_embeddings(self):
        """Precompute embeddings for semantic search"""
        if not USE_SEMANTIC:
            return
        self._embeddings = self.model.encode(
            [c["text"] for c in self.clauses],
            convert_to_tensor=True
        )

    # ---------------------------
    # Public API
    # ---------------------------
    def retrieve(self, treaty: Dict, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve the most relevant clauses for a treaty
        :param treaty: Dictionary containing treaty details
        :param top_k: Override number of top clauses to return
        :return: List of clause dictionaries
        """
        k = top_k or self.top_k

        if USE_SEMANTIC and self.model:
            # Semantic retrieval
            query_text = f"{treaty.get('peril', '')} {treaty.get('line_of_business', '')} {treaty.get('notes', '')}"
            query_emb = self.model.encode(query_text, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, self._embeddings)[0]
            top_indices = scores.argsort(descending=True)[:k]
            return [self.clauses[i] for i in top_indices]

        else:
            # Keyword fallback
            keywords = (treaty.get("peril", "") + " " +
                        treaty.get("line_of_business", "") + " " +
                        treaty.get("notes", "")).lower()
            ranked = sorted(
                self.clauses,
                key=lambda c: sum(kw in c["text"].lower() for kw in keywords.split()),
                reverse=True,
            )
            return ranked[:k]


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    retriever = ClauseRetriever()
    dummy_treaty = {
        "peril": "Hurricane",
        "line_of_business": "Property Cat XL",
        "notes": "High hurricane exposure; cat-excess reinsurance requested"
    }
    results = retriever.retrieve(dummy_treaty)
    print("Retrieved Clauses:")
    for r in results:
        print(f"[{r['id']}] ({r['category']}) {r['text']}")
