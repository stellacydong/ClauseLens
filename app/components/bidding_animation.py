# app/components/bidding_animation.py
import time
import numpy as np
import streamlit as st

def simulate_bidding(agents, treaty, steps=50, delay=0.05):
    """
    Simulates live bidding between MARL agents.
    Args:
        agents: list of MARLAgent instances or callables
        treaty: dict of treaty parameters
        steps: number of bid steps to visualize
        delay: delay (seconds) per step for animation
    Returns:
        bids: list of final bid dictionaries from all agents
    """
    bids = []

    # Initialize placeholders for live animation
    bid_table_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(steps):
        status_text.text(f"🟢 Bidding Step {step + 1}/{steps}")

        step_bids = []
        for agent_idx, agent in enumerate(agents):
            # Support both agent objects and callable functions
            if hasattr(agent, "get_bid"):
                bid = agent.get_bid(treaty)
            else:
                bid = agent(treaty)

            bid["agent_id"] = f"Agent_{agent_idx+1}"
            bid["step"] = step + 1
            step_bids.append(bid)

        # Append the most recent bids for final output
        bids = step_bids

        # Display live bidding table
        bid_table_placeholder.table([
            {
                "Agent": bid["agent_id"],
                "Quota Share": f"{bid['quota_share']:.0%}",
                "Premium ($)": f"{bid['premium']:,.0f}",
                "Expected Loss ($)": f"{bid['expected_loss']:,.0f}",
                "Tail Risk ($)": f"{bid['tail_risk']:,.0f}"
            }
            for bid in step_bids
        ])

        # Update progress
        progress_bar.progress((step + 1) / steps)
        time.sleep(delay)

    status_text.text("✅ Bidding Complete")
    return bids


def pick_winning_bid(bids):
    """
    Pick the winning bid based on a simple profit - tail risk tradeoff.
    Args:
        bids: list of bid dictionaries
    Returns:
        dict: winning bid
    """
    if not bids:
        return None

    # Simple scoring: Profit ~ premium - expected loss
    # Risk penalty ~ tail_risk
    def score_bid(bid):
        profit = bid["premium"] - bid["expected_loss"]
        return profit - 0.5 * bid["tail_risk"]

    winning_bid = max(bids, key=score_bid)
    return winning_bid
