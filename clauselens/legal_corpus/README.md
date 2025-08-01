# ClauseLens Legal Corpus

This folder contains the legal and regulatory clauses used for ClauseLens retrieval.

## Files

- **clauses.csv**: Table of clauses with IDs, text, jurisdiction, and line of business.
- **embeddings.npy**: (Optional) Precomputed vector embeddings of clause_text for semantic retrieval.
- **faiss_index.bin**: (Optional) FAISS index for nearest-neighbor search on embeddings.

## Adding New Clauses

1. Add new rows to `clauses.csv` with unique `clause_id`.
2. (Optional) Recompute embeddings and rebuild FAISS index if using semantic retrieval.
