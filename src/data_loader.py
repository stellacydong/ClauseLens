"""
data_loader.py
--------------
Utility functions to load and preprocess data for ClauseLens:
- Sample treaties
- Clause corpus
- Optional cache for faster repeated runs
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from src import config
from src.utils import normalize_text


class DataLoader:
    def __init__(self):
        self._treaty_cache = None
        self._clause_cache = None

    # ---------------------------
    # Treaties
    # ---------------------------
    def load_treaties(self, path: Optional[str] = None, refresh: bool = False) -> List[Dict]:
        """
        Load sample treaties from JSON
        :param path: Optional custom file path
        :param refresh: Force reload even if cached
        """
        path = path or config.SAMPLE_TREATIES_PATH
        if self._treaty_cache and not refresh:
            return self._treaty_cache

        if not os.path.exists(path):
            raise FileNotFoundError(f"Sample treaties file not found: {path}")

        with open(path, "r") as f:
            treaties = json.load(f)

        # Ensure required keys
        for t in treaties:
            for key in ["cedent", "peril", "line_of_business", "region", "exposure"]:
                if key not in t:
                    t[key] = "Unknown"
            if "notes" not in t:
                t["notes"] = ""
        self._treaty_cache = treaties
        return treaties

    # ---------------------------
    # Clauses
    # ---------------------------
    def load_clauses(self, path: Optional[str] = None, refresh: bool = False) -> List[Dict]:
        """
        Load clause corpus from JSON
        :param path: Optional custom file path
        :param refresh: Force reload even if cached
        """
        path = path or config.CLAUSES_PATH
        if self._clause_cache and not refresh:
            return self._clause_cache

        if not os.path.exists(path):
            raise FileNotFoundError(f"Clause corpus file not found: {path}")

        with open(path, "r") as f:
            clauses = json.load(f)

        # Normalize text and infer category if missing
        for c in clauses:
            c["text"] = normalize_text(c.get("text", ""))
            if "category" not in c:
                c["category"] = self._infer_category(c["text"])

        self._clause_cache = clauses
        return clauses

    def _infer_category(self, text: str) -> str:
        """Simple heuristic for clause categorization"""
        lower = text.lower()
        if "solvency" in lower:
            return "Capital Requirements"
        elif "ifrs" in lower:
            return "Accounting & Reporting"
        elif "naic" in lower or "rbc" in lower:
            return "Risk-Based Capital"
        elif "diversified" in lower:
            return "Reinsurance Structure"
        return "General Compliance"


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    loader = DataLoader()
    treaties = loader.load_treaties()
    print(f"Loaded {len(treaties)} treaties. Example:", treaties[0])

    clauses = loader.load_clauses()
    print(f"Loaded {len(clauses)} clauses. Example:", clauses[0])
