# src/config.py

"""
Configuration file for Reinsurance Demo
Centralizes hyperparameters, paths, and model settings
"""

import os

# =========================
# Project Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLAUSE_CORPUS_PATH = os.path.join(DATA_DIR, "clauses_corpus.json")
SAMPLE_TREATIES_PATH = os.path.join(DATA_DIR, "sample_treaties.json")

SAMPLE_RESULTS_DIR = os.path.join(DATA_DIR, "sample_results")

# =========================
# Multi-Agent Environment Settings
# =========================
NUM_AGENTS = 3                  # Number of MARL bidding agents in demo
DISCOUNT_FACTOR = 0.99          # RL gamma
EPISODE_LENGTH = 1              # Each demo run = single treaty placement

# =========================
# MARL Training Hyperparameters
# =========================
TRAINING_EPISODES = 10000
BATCH_SIZE = 1024
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
CVaR_ALPHA = 0.95               # Tail risk quantile
RISK_AVERSION = 0.1             # Higher = more conservative bids

# =========================
# Clause Retrieval Settings
# =========================
USE_TRANSFORMERS = True         # Set to False if using TF-IDF fallback
CLAUSE_TOP_K = 3                # Number of clauses to retrieve
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model

# =========================
# Explanation Generator Settings
# =========================
EXPLANATION_MODEL = "google/flan-t5-small"
EXPLANATION_MAX_LEN = 100

# =========================
# Visualization / Demo Settings
# =========================
SHOW_ANIMATION = True
SAVE_RESULTS = True
DASHBOARD_REFRESH_RATE = 1.0    # seconds

# =========================
# Utility Functions
# =========================
def ensure_directories():
    """Ensure that required directories exist."""
    os.makedirs(SAMPLE_RESULTS_DIR, exist_ok=True)

ensure_directories()

