"""
logger.py
---------
Lightweight logging utility for ClauseLens:
- Console logging with timestamps
- Optional file logging for episode results
- Supports INFO, WARNING, ERROR levels
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src import config


class ClauseLensLogger:
    def __init__(self, log_to_file: bool = True, log_file_name: str = "clauselens.log"):
        """
        :param log_to_file: Whether to write logs to file
        :param log_file_name: Log filename (inside config.LOG_DIR)
        """
        self.log_to_file = log_to_file
        self.log_file = Path(config.LOG_DIR) / log_file_name
        if self.log_to_file:
            os.makedirs(config.LOG_DIR, exist_ok=True)

    # ---------------------------
    # Core Logging
    # ---------------------------
    def _log(self, level: str, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"

        # Print to console
        print(log_entry)

        # Append to log file
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str):
        self._log("ERROR", message)

    # ---------------------------
    # Episode-Level Logging
    # ---------------------------
    def log_episode(self, episode_index: int, result: Dict[str, Any]):
        """
        Log an episode summary with key KPIs.
        """
        treaty = result.get("treaty", {})
        kpi = result.get("kpi", {})
        msg = (
            f"Episode {episode_index+1} | Cedent: {treaty.get('cedent','N/A')} | "
            f"Profit: {kpi.get('profit',0):,.0f} | "
            f"CVaR: {kpi.get('cvar',0):,.0f} | "
            f"Compliance: {'Pass' if kpi.get('regulatory_flags',{}).get('all_ok',False) else 'Fail'}"
        )
        self.info(msg)

    # ---------------------------
    # JSON Logging
    # ---------------------------
    def save_episode_json(self, episode_index: int, result: Dict[str, Any], folder: Optional[str] = None):
        """
        Save full episode result to JSON for debugging or reporting.
        """
        folder = folder or config.LOG_DIR
        os.makedirs(folder, exist_ok=True)
        file_path = Path(folder) / f"episode_{episode_index+1}.json"
        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)
        self.info(f"Saved episode {episode_index+1} results to {file_path}")


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    logger = ClauseLensLogger(log_to_file=False)
    logger.info("Starting ClauseLens demo...")
    logger.warning("This is a sample warning.")
    logger.error("This is a sample error.")

    dummy_result = {
        "treaty": {"cedent": "ABC Insurance"},
        "kpi": {"profit": 120_000, "cvar": 80_000, "regulatory_flags": {"all_ok": True}},
    }
    logger.log_episode(0, dummy_result)
