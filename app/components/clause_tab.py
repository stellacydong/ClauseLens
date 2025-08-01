import os
import pandas as pd
import streamlit as st
from clauselens.retrieval import ClauseRetriever
from clauselens.explain import ClauseExplainer

# Self-contained PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLAUSE_CSV = os.path.join(PROJECT_ROOT, "clauselens", "legal_corpus", "clauses.csv")
EMBED_PATH = os.path.join(PROJECT_ROOT, "clauselens", "legal_corpus", "embeddings.npy")
FAISS_PATH = os.path.join(PROJECT_ROOT, "clauselens", "legal_corpus", "faiss_index.bin")
DATA_DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "demo")


def render_clause_tab():
    st.subheader("Clause-Grounded Quote Explanations")
    st.info("ClauseLens loads lazily for performance.")

    if st.button("⚡ Load ClauseLens Engine"):
        with st.spinner("Loading ClauseLens..."):
            retriever = ClauseRetriever(CLAUSE_CSV, EMBED_PATH, FAISS_PATH)
            explainer = ClauseExplainer()

        treaties_path = os.path.join(DATA_DEMO_DIR, "sample_treaties.csv")
        if os.path.exists(treaties_path):
            sample_treaty = pd.read_csv(treaties_path).sample(1).iloc[0].to_dict()
        else:
            sample_treaty = {"line_of_business": "Property", "region": "EU", "limit": 1_000_000, "premium": 40_000}

        with st.spinner("Generating explanation..."):
            clauses = retriever.semantic_retrieve(sample_treaty, top_k=3)
            explanation = explainer.explain_quote(sample_treaty, clauses, bid_value=5_000_000)

        st.write("### Sample Treaty")
        st.json(sample_treaty)

        st.write("### Retrieved Clauses")
        for c in clauses:
            st.markdown(f"- **{c['clause_text']}** *(Jurisdiction: {c['jurisdiction']})*")

        st.write("### Live Explanation")
        st.info(explanation)
