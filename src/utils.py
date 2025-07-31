"""
utils.py
--------
Shared utilities for ClauseLens:
- Text normalization for PDF exports
- JSON save/load for logs and audit-ready data
- Random treaty selection for simulations
- Matplotlib chart saving for dashboards and PDF reports
"""

import os
import json
import unicodedata
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from src import config


# ---------------------------
# Text Utilities
# ---------------------------
def normalize_text(text: str) -> str:
    """
    Convert Unicode text to ASCII-safe text for PDF export.
    Strips unsupported characters like em dashes, emojis, etc.
    """
    if not text:
        return ""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


# ---------------------------
# JSON Utilities
# ---------------------------
def save_json(obj: Dict, filename: str, folder: Optional[str] = None):
    folder = folder or config.LOG_DIR
    os.makedirs(folder, exist_ok=True)
    file_path = Path(folder) / filename
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=2)
    return str(file_path)


def load_json(filename: str, folder: Optional[str] = None) -> Dict:
    folder = folder or config.LOG_DIR
    file_path = Path(folder) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


# ---------------------------
# Episode Utilities
# ---------------------------
def select_random_treaty(treaties: List[Dict]) -> Dict:
    """Return a random treaty from a list of treaty dicts."""
    if not treaties:
        return {}
    return random.choice(treaties)


def build_episode_summary(episode_idx: int, treaty: Dict, kpi: Dict) -> Dict:
    """
    Build a single episode summary for logging or table display.
    """
    return {
        "Episode": episode_idx + 1,
        "Cedent": treaty.get("cedent", "N/A"),
        "Peril": treaty.get("peril", "N/A"),
        "Profit ($)": f"{kpi.get('profit', 0):,.0f}",
        "CVaR ($)": f"{kpi.get('cvar', 0):,.0f}",
        "Compliance": "Pass" if kpi.get("regulatory_flags", {}).get("all_ok", False) else "Fail",
    }


# ---------------------------
# Visualization Utilities
# ---------------------------
def save_profit_vs_cvar_chart(
    marl_results: List[Dict],
    base_results: List[Dict],
    filename: str = "profit_vs_cvar.png",
    folder: Optional[str] = None
) -> str:
    """
    Save a Profit vs CVaR scatter plot for MARL vs Baseline agents.
    """
    folder = folder or config.REPORTS_DIR
    os.makedirs(folder, exist_ok=True)
    file_path = Path(folder) / filename

    marl_profits = [res["profit"] for res in marl_results]
    marl_cvar = [res["cvar"] for res in marl_results]
    base_profits = [res["profit"] for res in base_results]
    base_cvar = [res["cvar"] for res in base_results]

    plt.figure(figsize=(7, 5))
    plt.scatter(base_cvar, base_profits, color="red", label="Baseline")
    plt.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")
    plt.xlabel("CVaR (Tail Risk $)")
    plt.ylabel("Profit ($)")
    plt.title("Profit vs Tail-Risk (CVaR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return str(file_path)


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    # Test normalization
    print(normalize_text("Episode 1 – Profit ✅"))

    # Test JSON save/load
    test_data = {"profit": 120000, "cvar": 80000, "regulatory_flags": {"all_ok": True}}
    path = save_json(test_data, "test_episode.json")
    print("Saved JSON to:", path)
    loaded = load_json("test_episode.json")
    print("Loaded JSON:", loaded)
