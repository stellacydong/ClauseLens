import os
import pandas as pd
import streamlit as st

# Self-contained PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OVERRIDE_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "override_log.csv")


def render_governance_tab():
    st.subheader("Human-in-the-Loop Governance & Stress Tests")

    if st.button("⚡ Load Governance Tools"):
        from governance.policy_trace import load_policy_traces, plot_policy_traces
        from governance.override_interface import find_high_risk_bids, override_policy

        # Policy Traces
        try:
            df_traces = load_policy_traces(200)
            st.plotly_chart(plot_policy_traces(df_traces), use_container_width=True)
        except Exception as e:
            st.error(f"Policy trace visualization failed: {e}")

        # High-risk bids
        risky_bids = find_high_risk_bids()
        st.markdown(f"### High-Risk Bids (Compliance < 0.6): {len(risky_bids)}")
        st.dataframe(risky_bids.head(10), use_container_width=True)

        # Manual Override
        if st.button("🚨 Trigger Manual Override for 5 Bids"):
            if not risky_bids.empty:
                override_policy(risky_bids["bid_id"].head(5), reason="YC Demo Manual Override")
                st.success("✅ Overrides logged!")
            else:
                st.info("No high-risk bids found to override.")

        # Show Override Log
        if os.path.exists(OVERRIDE_LOG_PATH):
            st.markdown("### Override Log")
            log_df = pd.read_csv(OVERRIDE_LOG_PATH)
            st.dataframe(log_df.tail(20), use_container_width=True)
        else:
            st.info("No overrides logged yet.")
