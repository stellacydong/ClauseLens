import sys, os, time, random, json, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline
import importlib
import src.demo_pipeline as demo_pipeline
importlib.reload(demo_pipeline)
from src.demo_pipeline import run_multi_episode
from src.evaluation import summarize_portfolio

# ---------------------------
# Streamlit Page Settings
# ---------------------------
st.set_page_config(page_title="ClauseLens Investor Demo", layout="wide")
st.title("🧠 ClauseLens + Multi-Agent Treaty Bidding Demo")

st.markdown("""
This demo showcases:

1. **Intelligent Multi-Agent Bidding** with MARL  
2. **ClauseLens Explanations** grounded in regulatory clauses  
3. **Profitability, Tail-Risk (CVaR), and Compliance KPIs**
""")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Simulation Settings")
episodes = st.sidebar.slider("Number of Episodes", min_value=1, max_value=20, value=4, step=1)
show_animation = st.sidebar.checkbox("Show Live Bidding Animation", value=True)
steps_per_episode = st.sidebar.slider("Bidding Steps per Episode", min_value=10, max_value=50, value=30, step=5)
run_button = st.sidebar.button("Run Simulation 🚀")

# ---------------------------
# PDF Export Helper
# ---------------------------
def normalize_text(text: str) -> str:
    """Convert Unicode text to ASCII-safe for FPDF."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        if text else ""
    )

def export_pdf(results, summary, portfolio_results_marl, chart_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ClauseLens + MARL Investor Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, normalize_text(
        f"This report summarizes {len(results)} multi-agent treaty bidding episodes "
        f"powered by ClauseLens. It includes KPIs, winning bids, and full regulatory clause documentation."
    ))
    pdf.ln(10)

    # Portfolio Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Portfolio Summary KPIs", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, normalize_text(
        f"Episodes: {summary['episodes']}\n"
        f"Average Profit: ${summary['avg_profit']:,.0f}\n"
        f"Average CVaR: ${summary['avg_cvar']:,.0f}\n"
        f"Compliance Rate: {summary['compliance_rate']*100:.0f}%"
    ))

    # Episode Details
    for ep_data in results:
        pdf.add_page()
        treaty = ep_data["treaty"]
        winning_bid = ep_data["winning_bid"]

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Episode {ep_data['episode']} - {treaty.get('cedent','Unknown')}", ln=True)

        # Treaty overview
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, normalize_text(
            f"Peril: {treaty['peril']}\n"
            f"Region: {treaty['region']}\n"
            f"Exposure: ${treaty['exposure']:,.0f}\n"
            f"Limit: {treaty['limit']:.0%}\n"
            f"Quota Share Cap: {treaty['quota_share_cap']:.0%}\n"
            f"Notes: {treaty['notes']}"
        ))

        # Winning bid
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Winning Bid (MARL Agent)", ln=True)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, normalize_text(
            f"Quota Share: {winning_bid['quota_share']:.0%}\n"
            f"Premium: ${winning_bid['premium']:,.0f}\n"
            f"Expected Loss: ${winning_bid['expected_loss']:,.0f}\n"
            f"Tail Risk: ${winning_bid['tail_risk']:,.0f}"
        ))

        # ClauseLens Explanation
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "ClauseLens Explanation", ln=True)
        pdf.set_font("Helvetica", size=12)
        explanation = ep_data.get("explanation","No explanation generated.")
        pdf.multi_cell(0, 8, normalize_text(explanation))

        # Clauses
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Retrieved Clauses", ln=True)
        pdf.set_font("Helvetica", size=12)
        for clause in ep_data["clauses"]:
            clause_text = f"[{clause.get('category','N/A')}] {clause['text']} (Jurisdiction: {clause.get('jurisdiction','Global')})"
            pdf.multi_cell(0, 6, normalize_text(clause_text))
            pdf.ln(1)

    # Chart Page
    if chart_path and os.path.exists(chart_path):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Profit vs Tail-Risk (CVaR)", ln=True, align='C')
        pdf.image(chart_path, x=10, y=30, w=180)

    output_file = "ClauseLens_Investor_Report.pdf"
    pdf.output(output_file)
    return output_file

# ---------------------------
# Simulation Execution
# ---------------------------
if run_button:
    st.subheader(f"🔄 Running {episodes} Episode{'s' if episodes>1 else ''}...")
    progress_bar = st.progress(0)

    # Run simulation
    results, portfolio_results_marl, portfolio_results_baseline = run_multi_episode(
        num_episodes=episodes,
        steps_per_episode=steps_per_episode,
        show_animation=show_animation
    )

    # Summarize
    summary_marl = summarize_portfolio(portfolio_results_marl)
    summary_baseline = summarize_portfolio(portfolio_results_baseline)

    # Chart: Profit vs CVaR
    marl_profits = [res["profit"] for res in portfolio_results_marl]
    marl_cvar = [res["cvar"] for res in portfolio_results_marl]
    base_profits = [res["profit"] for res in portfolio_results_baseline]
    base_cvar = [res["cvar"] for res in portfolio_results_baseline]

    fig, ax = plt.subplots()
    ax.scatter(base_cvar, base_profits, color="red", label="Baseline")
    ax.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")
    ax.set_xlabel("CVaR (Tail Risk $)")
    ax.set_ylabel("Profit ($)")
    ax.set_title("Profit vs Tail-Risk Comparison")
    ax.legend()
    chart_path = "profit_vs_cvar.png"
    plt.savefig(chart_path)
    st.pyplot(fig)

    # ---------------------------
    # Compact 3-column Dashboard
    # ---------------------------
    st.markdown("## 📊 Portfolio Summary Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg MARL Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
    col2.metric("Avg MARL CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
    col3.metric("Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")

    # Episode Table
    episodes_df = pd.DataFrame([
        {
            "Episode": r["episode"],
            "Cedent": r["treaty"]["cedent"],
            "Winning Quota Share": f"{r['winning_bid']['quota_share']:.0%}",
            "Premium ($)": f"{r['winning_bid']['premium']:,.0f}",
            "Expected Loss ($)": f"{r['winning_bid']['expected_loss']:,.0f}",
            "Tail Risk ($)": f"{r['winning_bid']['tail_risk']:,.0f}"
        }
        for r in results
    ])
    st.dataframe(episodes_df, use_container_width=True, height=220)

    # Collapsible per-episode details
    for r in results:
        with st.expander(f"Episode {r['episode']} Details"):
            st.json(r["treaty"])
            st.markdown("**Winning Bid:**")
            st.json(r["winning_bid"])
            st.markdown("**ClauseLens Explanation:**")
            st.write(r["explanation"])
            st.markdown("**Retrieved Clauses:**")
            for clause in r["clauses"]:
                st.info(f"[{clause.get('category','N/A')}] {clause['text']} (Jurisdiction: {clause.get('jurisdiction','Global')})")

    # PDF Download
    pdf_file = export_pdf(results, summary_marl, portfolio_results_marl, chart_path)
    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Full Investor Report", f, file_name="ClauseLens_Investor_Report.pdf")

else:
    st.info("Press **Run Simulation 🚀** to start multi-agent treaty bidding and generate the investor dashboard.")
