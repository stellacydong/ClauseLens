import sys, os, json, time, unicodedata
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_bids, summarize_portfolio
from src.marl_agents import MARLAgent
from app.components.bidding_animation import simulate_bidding, live_bidding_summary
from app.components.clause_explainer import ClauseExplainerComponent
from app.components.kpi_dashboard import (
    display_portfolio_kpis,
    display_episode_table,
    display_profit_vs_cvar_chart,
)

# ---------------------------
# Load Sample Treaties
# ---------------------------
DATA_DIR = "data"
with open(os.path.join(DATA_DIR, "sample_treaties.json")) as f:
    SAMPLE_TREATIES = json.load(f)

# ---------------------------
# Streamlit Page Settings
# ---------------------------
st.set_page_config(page_title="ClauseLens Live Demo", layout="wide")
st.title("🧠 ClauseLens + Multi-Agent Treaty Bidding Demo")

st.markdown("""
Welcome to the **ClauseLens Investor Demo**.

This app demonstrates:
1. **Multi-Agent Reinforcement Learning (MARL)** bidding  
2. **Live auction-style bidding animation**  
3. **ClauseLens explanations** grounded in real regulatory clauses  
4. **Portfolio KPIs, Tail-Risk (CVaR), and Compliance Metrics**
""")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Simulation Settings")
episodes = st.sidebar.slider("Number of Episodes", 1, len(SAMPLE_TREATIES), 3)
steps = st.sidebar.slider("Bidding Animation Steps", 10, 100, 50)
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)
run_button = st.sidebar.button("Run Simulation 🚀")

# ---------------------------
# Utilities
# ---------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def export_pdf(results, summary, kpis, chart_path):
    """Generate full audit-ready PDF with clauses and explanations."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ClauseLens Investor Demo Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8,
        "This report summarizes MARL bidding results with ClauseLens explanations "
        "and regulatory clause traceability."
    )

    # Portfolio Summary
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Portfolio Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8,
        f"Episodes: {summary['episodes']}\n"
        f"Average Profit: ${summary['avg_profit']:,.0f}\n"
        f"Average CVaR: ${summary['avg_cvar']:,.0f}\n"
        f"Compliance Rate: {summary['compliance_rate']*100:.0f}%"
    )

    # Episode Results Table
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Episode Results (MARL)", ln=True)
    pdf.set_font("Arial", "B", 9)
    headers = ["Ep", "Cedent", "Profit", "CVaR", "Comp", "Clause IDs", "Categories"]
    widths = [8, 30, 22, 22, 12, 40, 50]
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 8)
    for i, (ep, kpi) in enumerate(zip(results, kpis), 1):
        row = [
            str(i),
            ep["treaty"].get("cedent", "")[:15],
            f"{kpi['profit']:,.0f}",
            f"{kpi['cvar']:,.0f}",
            "P" if kpi['regulatory_flags']['all_ok'] else "F",
            ", ".join(str(c["id"]) for c in ep["clauses"]),
            ", ".join(c.get("category", "N/A") for c in ep["clauses"]),
        ]
        for val, w in zip(row, widths):
            pdf.cell(w, 8, normalize_text(val), border=1)
        pdf.ln()

    # Full episode pages with clauses
    for i, ep in enumerate(results, 1):
        pdf.add_page()
        treaty = ep["treaty"]
        title = f"Episode {i} - {treaty.get('cedent','Unknown')}"
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, normalize_text(title), ln=True)
        pdf.ln(4)

        # Explanation
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "ClauseLens Explanation:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, normalize_text(ep.get("explanation", "No explanation.")))
        pdf.ln(4)

        # Full Clauses
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Retrieved Clauses:", ln=True)
        pdf.set_font("Arial", "", 10)
        if ep["clauses"]:
            for clause in ep["clauses"]:
                text = f"ID {clause['id']} ({clause.get('category','N/A')}): {clause['text']}"
                pdf.multi_cell(0, 6, normalize_text(text))
                pdf.ln(1)
        else:
            pdf.multi_cell(0, 6, "No clauses retrieved.")
        pdf.ln(2)

    # Profit vs CVaR chart
    if chart_path and os.path.exists(chart_path):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Profit vs Tail-Risk (CVaR)", ln=True, align="C")
        pdf.image(chart_path, x=10, y=30, w=180)

    outfile = "ClauseLens_Full_Audit_Report.pdf"
    pdf.output(outfile)
    return outfile


# ---------------------------
# Simulation + Dashboard
# ---------------------------
if run_button:
    np.random.seed(seed)
    explainer = ClauseExplainerComponent(os.path.join(DATA_DIR, "clauses_corpus.json"), seed=seed)

    results, kpis = [], []
    selected_treaties = SAMPLE_TREATIES[:episodes]

    tabs = st.tabs([f"Episode {i+1}" for i in range(episodes)] + ["Portfolio Summary"])

    for ep_idx, treaty in enumerate(selected_treaties):
        with tabs[ep_idx]:
            st.markdown(f"## 🎯 Episode {ep_idx+1}: {treaty['cedent']}")
            agents = [MARLAgent(i) for i in range(3)]

            # Live bidding animation
            st.markdown("#### ⏱️ Live Bidding Animation")
            bids = simulate_bidding(agents, treaty=treaty, steps=steps, delay=0.02)
            winner_idx = live_bidding_summary(bids)

            # ClauseLens retrieval and explanation
            clauses, explanation = explainer.explain_treaty(treaty, bids[winner_idx])

            # Save results
            ep_result = {
                "treaty": treaty,
                "winner": winner_idx,
                "winning_bid": bids[winner_idx],
                "clauses": clauses,
                "explanation": explanation
            }
            results.append(ep_result)
            kpis.append(evaluate_bids([bids[winner_idx]], winner_idx=0, treaty=treaty))

            # Show Explanation and Clauses
            st.markdown("#### 📄 ClauseLens Explanation")
            st.write(explanation)
            st.markdown("#### Retrieved Clauses")
            for clause in clauses:
                st.info(f"ID {clause['id']} ({clause.get('category','N/A')}): {clause['text']}")

    # ---------------------------
    # Portfolio Tab
    # ---------------------------
    with tabs[-1]:
        summary = summarize_portfolio(kpis)
        display_portfolio_kpis(summary)  # Use component
        display_episode_table(kpis)      # Use component

        # Profit vs CVaR Chart
        chart_path = "profit_vs_cvar.png"
        display_profit_vs_cvar_chart(kpis)
        plt.savefig(chart_path)

        # PDF Download
        pdf_file = export_pdf(results, summary, kpis, chart_path)
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download Full Investor Report", f, file_name=pdf_file)
