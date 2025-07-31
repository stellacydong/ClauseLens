"""
config.py
---------
Central configuration for ClauseLens demo, training, and reporting.
Manages all file paths, model parameters, and global constants.
"""

import os
from pathlib import Path

# ---------------------------
# Project Root
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------
# Data Paths
# ---------------------------
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_TREATIES_PATH = DATA_DIR / "sample_treaties.json"
CLAUSES_PATH = DATA_DIR / "clauses_corpus.json"

# ---------------------------
# Model & Simulation Settings
# ---------------------------
NUM_AGENTS = 3
DEFAULT_EPISODES = 5
TOP_K_CLAUSES = 3

# Optional: semantic model for clause retrieval
USE_SEMANTIC_RETRIEVAL = True
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# Training / Checkpoints
# ---------------------------
CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Logging & Reporting
# ---------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PDF_NAME = "ClauseLens_Demo_Report.pdf"

# ---------------------------
# Visualization
# ---------------------------
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHART_PATH = FIGURES_DIR / "profit_vs_cvar.png"

# ---------------------------
# Utility Functions
# ---------------------------
def ensure_directories():
    """Ensure that all required directories exist."""
    for path in [DATA_DIR, LOG_DIR, REPORTS_DIR, FIGURES_DIR, CHECKPOINT_DIR]:
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Quick check for paths
    print("Project Root:", PROJECT_ROOT)
    print("Sample Treaties:", SAMPLE_TREATIES_PATH.exists())
    print("Clauses Corpus:", CLAUSES_PATH.exists())
    ensure_directories()
