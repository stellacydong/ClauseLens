import streamlit as st
import pandas as pd
import subprocess
import sys
import os


# Add the parent of 'components' (i.e., the app folder) to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils.load_data import load_simulation_results



def render():
    st.header("🤖 Multi-Agent Bidding Simulation (MAPPO Learning)")
    st.write(
        """
        This tab runs the Multi-Agent PPO (MAPPO) simulation in the Treaty Bidding Environment.
        The system learns bidding strategies and optimizes KPIs such as profit, win rate, and CVaR.
        """
    )

    # -----------------------------------------------------------------------------
    # Run Simulation Button
    # -----------------------------------------------------------------------------
    if st.button("▶️ Run MAPPO Simulation"):
        st.info("Running MAPPO simulation... this may take a moment.")
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "03_run_simulation.py")],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            st.success("Simulation completed successfully.")
        else:
            st.error("Simulation failed.")
            st.text(result.stderr)

    # -----------------------------------------------------------------------------
    # Load Simulation Results
    # -----------------------------------------------------------------------------
    df = load_simulation_results()

    if not df.empty:
        st.subheader("📊 MAPPO Learning Curves")

        # Ensure proper columns exist
        required_cols = ["round", "avg_profit", "win_rate", "cvar_95"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns in simulation results: {missing_cols}")
            return

        # Line charts for key KPIs
        st.line_chart(df.set_index("round")[["avg_profit"]], height=200, use_container_width=True)
        st.line_chart(df.set_index("round")[["win_rate"]], height=200, use_container_width=True)
        st.line_chart(df.set_index("round")[["cvar_95"]], height=200, use_container_width=True)

        # Data Table
        st.subheader("Simulation Results Table")
        st.dataframe(df)

    else:
        st.info("No simulation results found. Run the MAPPO simulation to generate KPIs.")
