import time
import numpy as np
import pandas as pd
import streamlit as st

def simulate_bidding(agents, treaty, steps=50, delay=0.02):
    """
    Simulate and animate a live bidding process in Streamlit.

    Args:
        agents (list): List of MARLAgent instances (or dicts with get_bid()).
        treaty (dict): Current treaty for bidding.
        steps (int): Number of animation steps to simulate.
        delay (float): Delay (in seconds) per step for animation effect.

    Returns:
        list of dict: Final bids for all agents
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    bid_chart = st.empty()

    # Initialize bid history
    history = {f"Agent {i}": [] for i in range(len(agents))}

    for step in range(steps):
        status_text.text(f"Simulating bids... Step {step+1}/{steps}")

        # Each agent generates a bid (simulate incremental bidding)
        current_bids = []
        for idx, agent in enumerate(agents):
            bid = agent.get_bid(treaty) if hasattr(agent, "get_bid") else agent(treaty)
            # Simulate bid increasing slightly over steps
            bid["premium"] *= (0.95 + 0.1 * step / steps)
            current_bids.append(bid)
            history[f"Agent {idx}"].append(bid["premium"])

        # Update line chart of bid premiums
        df = pd.DataFrame(history)
        bid_chart.line_chart(df)

        # Update progress bar
        progress_bar.progress(int((step + 1) / steps * 100))
        time.sleep(delay)

    status_text.text("Bidding Complete ✅")
    return current_bids

def live_bidding_summary(bids):
    """
    Display the final bids from live simulation.
    """
    st.markdown("### Final Bids")
    st.dataframe(pd.DataFrame(bids), use_container_width=True)

    # Highlight winning bid (highest profit)
    profits = [b["premium"] - b["expected_loss"] for b in bids]
    winner_idx = int(np.argmax(profits))
    st.success(f"🏆 Winning Agent: Agent {winner_idx} with Profit ${profits[winner_idx]:,.0f}")
    return winner_idx
