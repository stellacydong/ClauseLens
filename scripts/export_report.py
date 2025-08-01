import os
import pandas as pd
from datetime import datetime

# Optional PDF export
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# -----------------------------
# Config
# -----------------------------
DEMO_DIR = "../data/demo"
REPORT_DIR = "../data/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# File paths
kpi_file = os.path.join(DEMO_DIR, "dashboard_kpis.csv")
trend_file = os.path.join(DEMO_DIR, "dashboard_trends.csv")
bids_file = os.path.join(DEMO_DIR, "dashboard_bids.csv")
marketlens_file = os.path.join(DEMO_DIR, "sample_marketlens.parquet")

# -----------------------------
# 1. Load Data
# -----------------------------
if not os.path.exists(kpi_file):
    raise FileNotFoundError("❌ dashboard_kpis.csv not found. Run generate_dashboard_data.py first!")

kpis_df = pd.read_csv(kpi_file)
trends_df = pd.read_csv(trend_file)
bids_df = pd.read_csv(bids_file)

# Optional: MarketLens summary
marketlens_summary = None
if os.path.exists(marketlens_file):
    marketlens_df = pd.read_parquet(marketlens_file)
    marketlens_summary = marketlens_df.describe()

print(f"✅ Loaded KPIs: {kpis_df.shape}")
print(f"✅ Loaded Trends: {trends_df.shape}")
print(f"✅ Loaded Bids: {bids_df.shape}")
if marketlens_summary is not None:
    print(f"✅ Loaded MarketLens features: {marketlens_df.shape}")

# -----------------------------
# 2. Export Combined CSV Report
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_report_path = os.path.join(REPORT_DIR, f"ClauseLens_Report_{timestamp}.csv")

# Merge KPI and last 50 bids for summary CSV
kpi_snapshot = kpis_df.tail(1)
latest_bids = bids_df.tail(50)
csv_report = pd.concat([kpi_snapshot, latest_bids], axis=1)
csv_report.to_csv(csv_report_path, index=False)

print(f"✅ Saved CSV report to {csv_report_path}")

# -----------------------------
# 3. Export PDF Report (Optional)
# -----------------------------
if FPDF is None:
    print("⚠️ fpdf not installed. Run `pip install fpdf` to enable PDF reports.")
else:
    pdf_path = os.path.join(REPORT_DIR, f"ClauseLens_Report_{timestamp}.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "ClauseLens Simulation Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    # KPIs Section
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Key Performance Indicators", ln=True)

    pdf.set_font("Arial", size=11)
    last_kpis = kpis_df.tail(1).to_dict(orient="records")[0]
    for key, val in last_kpis.items():
        pdf.cell(200, 8, f"{key}: {val}", ln=True)

    # MarketLens summary section
    if marketlens_summary is not None:
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "MarketLens Feature Summary", ln=True)
        pdf.set_font("Arial", size=10)

        summary_lines = marketlens_summary.to_string().split("\n")
        for line in summary_lines:
            pdf.cell(200, 5, line, ln=True)

    pdf.output(pdf_path)
    print(f"✅ Saved PDF report to {pdf_path}")

print("\n🎯 Reports ready for download.")
