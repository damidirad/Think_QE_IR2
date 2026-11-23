
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
BASE = Path("/scratch-shared/scur0915/embedding_analysis")
SHARED = Path("/scratch-shared/scur0900/thinkqe/outputs")
FAISS_DIR = Path("/scratch-shared/scur0915/msmarco-sbert")

OUT = BASE / "dl19_all_queries_all_seeds"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "Qwen3.8-27B"
K = 5
SEEDS = [0, 1, 2]
ROUNDS = [0, 1, 2, 3]

# Load DL19 ground-truth judgments
qrels = pd.read_csv(
    BASE / "ground_truth_qrels.csv",
    dtype={"query_id": str, "doc_id": str}
)

qrels = qrels[qrels.dataset == "dl19"].copy()
query_ids = set(qrels.query_id.unique())

print("DL19 judged queries:", len(query_ids))

# Load IDs of previously extracted SBERT embeddings
with np.load(BASE / "ground_truth_embeddings.npz") as data:
    available_ids = set(data["docids"])

print("Available embeddings:", len(available_ids))

# Collect ThinkQE retrieval results
frames = []

for seed in SEEDS:

    run = SHARED / f"dl19_{MODEL}__batched_s{seed}"

    for r in ROUNDS:

        path = run / f"bm25-aug{r}_result_retrieval.trec"

        df = pd.read_csv(
            path,
            sep=r"\s+",
            names=[
                "query_id", "Q0", "doc_id",
                "rank", "score", "tag"
            ],
            dtype={"query_id": str, "doc_id": str}
        )

        # Select judged queries and top-k documents
        df = df[
            df.query_id.isin(query_ids) &
            df["rank"].between(1, K)
        ].copy()

        df = df.sort_values(["query_id", "rank"])

        # Validate retrieval lists
        counts = df.groupby("query_id").size()

        valid = (
            set(counts.index) == query_ids
            and counts.eq(K).all()
            and not df.duplicated(
                ["query_id", "rank"]
            ).any()
            and not df.duplicated(
                ["query_id", "doc_id"]
            ).any()
        )

        if not valid:
            raise ValueError(
                f"Invalid ranking: seed {seed}, round {r}"
            )

        df["seed"] = seed
        df["round"] = r

        frames.append(df)

        print(
            f"Seed {seed} | Round {r} | "
            f"Queries: {df.query_id.nunique()} | "
            f"Documents: {len(df)}"
        )

# Combine retrieval results
retrieval = pd.concat(frames, ignore_index=True)

# Check embedding coverage
retrieved_ids = set(retrieval.doc_id)
existing = retrieved_ids & available_ids
missing = retrieved_ids - available_ids

coverage = 100 * len(existing) / len(retrieved_ids)

# Summary
print("\n========== SUMMARY ==========")
print("Model:", MODEL)
print("Queries:", len(query_ids))
print("Seeds:", len(SEEDS))
print("Retrieval stages:", len(ROUNDS))
print("Top-k:", K)

print(
    "Retrieval lists:",
    retrieval.groupby(
        ["query_id", "seed", "round"]
    ).ngroups
)

print("Total document occurrences:", len(retrieval))
print("Unique retrieved documents:", len(retrieved_ids))
print("Existing embeddings:", len(existing))
print("Missing embeddings:", len(missing))
print(f"Embedding coverage: {coverage:.2f}%")

# Save results
retrieval.to_csv(
    OUT / "retrieved_documents.csv",
    index=False
)

pd.DataFrame({
    "doc_id": sorted(missing)
}).to_csv(
    OUT / "missing_docids.csv",
    index=False
)

pd.DataFrame({
    "doc_id": sorted(retrieved_ids)
}).to_csv(
    OUT / "retrieved_docids.csv",
    index=False
)

print("\nSaved to:", OUT)
print("ANALYSIS COMPLETED")
