# **Transparent Market Platform for Reinsurance**

🚀 **AI-Powered, Clause-Grounded, Auditable Treaty Bidding**

We are building the **first broker-neutral, transparent reinsurance market platform** that combines:

1. **Open Treaty Bidding** – Multi-Agent Reinforcement Learning (MARL) for dynamic pricing
2. **ClauseLens** – Clause-grounded quote explanations for regulatory transparency
3. **MarketLens** – Market benchmarking & fairness dashboards
4. **Governance Layer** – Human-in-the-loop oversight and audit logging

This platform advances **reinsurance market efficiency, interpretability, and compliance**.

---

## **📂 Repository Structure**

```
TransparentMarketPlatform/
│
├── app/                         # Streamlit / YC demo dashboard
│   ├── demo_app.py               # Main dashboard entry point
│   ├── components/               # Modular UI elements
│   │   ├── bidding_tab.py
│   │   ├── clause_tab.py
│   │   ├── marketlens_tab.py
│   │   └── governance_tab.py
│   ├── assets/                   # Images, logos, icons
│   └── requirements.txt
│
├── marl_engine/                  # Multi-Agent Treaty Bidding (MAPPO + CVaR)
│   ├── simulate_env.py
│   ├── marl_agents.py
│   ├── stress_tests.py
│   ├── utils.py
│   └── __init__.py
│
├── clauselens/                   # Clause-grounded explanation module
│   ├── retrieval.py
│   ├── explain.py
│   ├── legal_corpus/             # (Optional) Clause texts / embeddings
│   └── __init__.py
│
├── marketlens/                   # ML Benchmarking & Fairness
│   ├── preprocess.py
│   ├── train_marketlens.py
│   ├── fairness_audit.py
│   ├── models/                   # XGBoost/LightGBM trained models
│   └── __init__.py
│
├── governance/                   # Human-in-the-loop oversight
│   ├── policy_trace.py
│   ├── override_interface.py
│   └── __init__.py
│
├── data/                         # Synthetic & anonymized treaty data
│   ├── treaties_synthetic.csv
│   ├── treaties_anonymized.csv
│   └── marketlens_features.parquet
│
├── notebooks/                    # Jupyter/Colab experiments
│   ├── marl_training.ipynb
│   ├── marketlens_training.ipynb
│   ├── clauselens_demo.ipynb
│   └── governance_prototype.ipynb
│
├── papers/                       # ICAIF 2025 submissions
│   ├── Multi-AgentTreatyBiddingSystem.pdf
│   ├── MarketLens.pdf
│   ├── ClauseLens.pdf
│   ├── GovernanceInLoop.pdf
│   └── bibtex/
│
├── scripts/                      # Utility scripts
│   ├── run_simulation.py
│   ├── generate_dashboard_data.py
│   └── export_report.py
│
├── tests/                        # Unit tests for reproducibility
│   ├── test_marl.py
│   ├── test_clauselens.py
│   ├── test_marketlens.py
│   └── test_governance.py
│
├── .gitignore
├── README.md                     # Project overview and setup instructions
├── LICENSE
└── setup.py                      # Optional for packaging
```

---

## **🛠️ Installation**

Clone the repository and set up the environment:

```bash
git clone https://github.com/YOUR_USERNAME/TransparentMarketPlatform.git
cd TransparentMarketPlatform/app
pip install -r requirements.txt
```

Recommended: Use **Python 3.10+** and a **virtual environment or conda**.

---


# Data Folder – Transparent Market Platform

This folder contains synthetic and anonymized data for MARL simulation, MarketLens benchmarking, and ClauseLens demo.

---

## Folders

data/
│
├── raw/                          # Unprocessed data (ignored in GitHub if large)
│   ├── treaties_raw.csv           # Synthetic raw treaties (pre-cleaning)
│   └── reinsurer_info.csv         # Cedent/reinsurer metadata
│
├── processed/                     # Cleaned data for model training/demo
│   ├── treaties_synthetic.csv     # Synthetic treaty submissions (100k+)
│   ├── treaties_anonymized.csv    # Optional: anonymized real treaty data
│   ├── marketlens_features.parquet # Feature matrix for ML benchmarking
│   └── marketlens_labels.parquet   # Labels for acceptance/loss predictions
│
├── demo/                          # Small sample for Streamlit demo
│   ├── sample_treaties.csv
│   └── sample_marketlens.parquet
│
└── README.md                      # Documentation of data sources & schema




### 1. `raw/`
- **Purpose**: Original, unprocessed data (ignored in public repo if sensitive)
- **Files**:
  - `treaties_raw.csv` – 100k+ synthetic treaty submissions with full metadata
  - `reinsurer_info.csv` – Reinsurer and cedent metadata (incumbent flags, region)

### 2. `processed/`
- **Purpose**: Cleaned data ready for models and dashboards
- **Files**:
  - `treaties_synthetic.csv` – Primary dataset for simulation and training
  - `treaties_anonymized.csv` – Optional, anonymized real treaty data
  - `marketlens_features.parquet` – ML features for MarketLens model
  - `marketlens_labels.parquet` – Labels: acceptance, loss ratio, deviation

### 3. `demo/`
- **Purpose**: Lightweight sample (~1,000 rows) for Streamlit demo
- **Files**:
  - `sample_treaties.csv`
  - `sample_marketlens.parquet`

---

## Schema

| Column                | Type    | Description                               |
|-----------------------|--------|-------------------------------------------|
| cedent_id             | str    | Unique cedent identifier                   |
| reinsurer_id          | str    | Unique reinsurer identifier                |
| treaty_type           | str    | "XoL" or "Quota Share"                     |
| line_of_business      | str    | Property / Casualty / Specialty            |
| region                | str    | Treaty jurisdiction                        |
| premium               | float  | Quoted premium                             |
| attachment_point      | float  | Attachment threshold for XoL               |
| limit                 | float  | Coverage limit                             |
| quota_share           | float  | % of ceded portfolio (for QS treaties)      |
| accepted              | bool   | 1 if treaty was bound, 0 otherwise          |
| observed_loss_ratio   | float  | Observed loss ratio for the treaty         |
| cvar_95               | float  | Conditional Value at Risk (95%)            |




## **▶️ Run the YC Demo Dashboard**

1. Navigate to the `app` folder:

   ```bash
   cd app
   ```
2. Launch the Streamlit dashboard:

   ```bash
   streamlit run demo_app.py
   ```
3. Open the local URL (usually `http://localhost:8501`) to interact with:

   * **Live Treaty Bidding** (MARL engine simulation)
   * **ClauseLens** (clause-grounded explanations)
   * **MarketLens** (market benchmarking & fairness audit)
   * **Governance Layer** (policy traces & manual override)

---

## **📊 Features in the YC Demo**

* **Multi-Agent Treaty Bidding**

  * Simulated agents compete with PPO/MAPPO under CVaR constraints
  * Live Pareto plot of Profit vs. Tail Risk

* **ClauseLens (Explainable Bids)**

  * Retrieves clauses from Solvency II / IFRS 17 / NAIC
  * Generates natural language quote justifications

* **MarketLens (Benchmarking & Fairness)**

  * Quote acceptance likelihood predictions
  * Loss ratio deviation scoring
  * SHAP-based fairness audit by reinsurer class

* **Governance-in-the-Loop**

  * Policy logging and trace visualization
  * Counterfactual bid explanations
  * Manual override for high-risk bids

---

## **📄 Papers and Research**

This project is supported by 4 ICAIF 2025 companion papers:

1. **Multi-Agent Treaty Bidding System** (Engine)
2. **ClauseLens: Clause-Grounded Quote Explanation**
3. **MarketLens: Benchmarking & Fairness**
4. **Governance-in-the-Loop for Auditable MARL**

See the [`papers/`](papers) folder for preprints.

---

## **🤝 Contributing**

We welcome contributions!

1. Fork the repo
2. Create a feature branch (`feature/new-module`)
3. Submit a pull request with detailed comments

---

## **📜 License**

MIT License – free for research and non-commercial use.

