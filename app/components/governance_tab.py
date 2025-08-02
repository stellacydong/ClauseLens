import streamlit as st
import pandas as pd

def render():
    st.header("⚖️ Governance & Human-in-the-Loop Oversight")
    st.write(
        """
        Review agent decision traces, manually override policies, and ensure CVaR and regulatory compliance.
        """
    )

    # Policy trace table (mock)
    trace = pd.DataFrame({
        "step": [1, 2, 3],
        "action": ["Bid 1.2M", "Bid 1.5M", "Hold"],
        "reward": [120_000, 150_000, 90_000],
        "compliant": [True, True, False]
    })
    st.subheader("Policy Trace")
    st.dataframe(trace)

    # Override demo
    st.subheader("Manual Policy Override")
    override_action = st.selectbox("Override next action?", ["No Override", "Force Bid", "Force Hold"])
    if override_action != "No Override":
        st.warning(f"Next simulation step will use override: {override_action}")

    # Governance summary
    st.metric("Compliance Rate", "92%")
    st.metric("Manual Overrides", "1")
