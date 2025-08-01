import os
import numpy as np
import pandas as pd

# Force single-thread execution for PyTorch + FAISS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import faiss
except ImportError:
    faiss = None

# -----------------------------
# Global cached model
# -----------------------------
_model_cache = None

def get_model(force_reload: bool = False):
    """
    Lazy-load and cache the SentenceTransformer model on CPU.
    If force_reload=True, rebuilds the model to recover from meta tensor state.
    """
    global _model_cache
    if _model_cache is None or force_reload:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("✅ SentenceTransformer model loaded on CPU (single-threaded)")
    return _model_cache


class ClauseRetriever:
    """
    ClauseLens retrieval system supporting:
    1. Keyword-based retrieval
    2. Semantic retrieval (Sentence-BERT)
    3. FAISS-accelerated top-k retrieval
    """

    def __init__(self, clause_csv_path, embedding_path=None, faiss_path=None):
        if not os.path.exists(clause_csv_path):
            raise FileNotFoundError(f"Clause CSV not found: {clause_csv_path}")

        # Load clause dataset
        self.clause_df = pd.read_csv(clause_csv_path)
        self.embeddings = None
        self.index = None

        # Load embeddings if available
        if embedding_path and os.path.exists(embedding_path):
            self.embeddings = np.load(embedding_path).astype('float32')
            print(f"✅ Loaded embeddings: {self.embeddings.shape}")

        # Load FAISS index if available
        if faiss and faiss_path and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            print(f"✅ Loaded FAISS index from {faiss_path}")
        elif faiss_path:
            print("⚠️ FAISS index file not found or faiss not installed. Using fallback methods.")

    # -----------------------------
    # Keyword Retrieval
    # -----------------------------
    def retrieve(self, treaty_features: dict, top_k: int = 3):
        """
        Simple keyword-based retrieval based on line_of_business and jurisdiction.
        """
        lob = treaty_features.get("line_of_business", "").lower()
        region = treaty_features.get("region", "").lower()

        df = self.clause_df.copy()
        df["score"] = 0

        if "line_of_business" in df.columns:
            df["score"] += df["line_of_business"].str.lower().apply(lambda x: 1 if lob in x else 0)

        if "jurisdiction" in df.columns and region:
            df["score"] += df["jurisdiction"].str.lower().apply(lambda x: 1 if region in x else 0)

        df = df.sort_values(by=["score", "clause_id"], ascending=[False, True])
        return df.head(top_k).to_dict(orient="records")

    # -----------------------------
    # Semantic Retrieval
    # -----------------------------
    def semantic_retrieve(self, treaty_features: dict, top_k: int = 3):
        """
        Semantic retrieval using FAISS if available, otherwise cosine similarity over embeddings.
        Falls back to keyword retrieval if no embeddings exist.
        """
        # If FAISS index exists, use it
        if self.index is not None:
            return self._safe_search(self._faiss_search, treaty_features, top_k)

        # If embeddings exist but FAISS is unavailable
        if self.embeddings is not None:
            return self._safe_search(self._embedding_search, treaty_features, top_k)

        # Otherwise, fallback to keyword
        print("⚠️ No embeddings or FAISS index found. Falling back to keyword retrieval.")
        return self.retrieve(treaty_features, top_k)

    # -----------------------------
    # Safe Search Wrapper
    # -----------------------------
    def _safe_search(self, search_func, treaty_features, top_k):
        """
        Wrapper that retries semantic search if a meta tensor error occurs.
        """
        try:
            return search_func(treaty_features, top_k)
        except RuntimeError as e:
            if "meta tensor" in str(e).lower():
                print("⚠️ Detected meta tensor error. Reloading model on CPU and retrying...")
                get_model(force_reload=True)  # Force reload the model
                return search_func(treaty_features, top_k)
            else:
                raise

    # -----------------------------
    # FAISS Search
    # -----------------------------
    def _faiss_search(self, treaty_features: dict, top_k: int):
        model = get_model()

        lob = treaty_features.get("line_of_business", "Unknown")
        region = treaty_features.get("region", "Global")
        query = f"{lob} treaty in {region} with regulatory and solvency compliance"

        # Compute query embedding (single-threaded, no multiprocessing)
        query_vec = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            num_workers=0
        ).astype('float32')
        faiss.normalize_L2(query_vec)

        # Search top-k
        distances, indices = self.index.search(query_vec, top_k)
        return self.clause_df.iloc[indices[0]].to_dict(orient="records")

    # -----------------------------
    # Embedding-only Search
    # -----------------------------
    def _embedding_search(self, treaty_features: dict, top_k: int):
        from sklearn.metrics.pairwise import cosine_similarity

        model = get_model()
        lob = treaty_features.get("line_of_business", "Unknown")
        region = treaty_features.get("region", "Global")
        query = f"{lob} treaty in {region} with regulatory and solvency compliance"

        query_vec = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            num_workers=0
        ).astype('float32').reshape(1, -1)

        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        return self.clause_df.iloc[top_indices].to_dict(orient="records")
