# Computes cosine-based semantic similarity, novelty, exact overlap, and gold-document retrieval metrics across DL19 queries, seeds, and ThinkQE rounds.
# Aggregates results per seed, query, and round to analyse retrieval redundancy, semantic diversity, and changes in known relevant-document coverage.

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/scratch-shared/scur0915/embedding_analysis")
OUT = BASE / "dl19_all_queries_all_seeds"

# Load document embeddings
with np.load(OUT / "combined_embeddings.npz") as data:
    ids = data["docids"]
    E = data["embeddings"]

E = E / np.linalg.norm(E, axis=1, keepdims=True)
idx = {d: i for i, d in enumerate(ids)}

# Load retrieval rankings and relevance judgments
retrieval = pd.read_csv(
    OUT / "retrieved_documents.csv",
    dtype={"query_id": str, "doc_id": str}
)

qrels = pd.read_csv(
    BASE / "ground_truth_qrels.csv",
    dtype={"query_id": str, "doc_id": str}
)

gold = (
    qrels[(qrels.dataset == "dl19") & (qrels.relevance >= 2)]
    .groupby("query_id").doc_id
    .apply(lambda x: sorted(set(x))).to_dict()
)

def sim(a, b):
    return E[[idx[d] for d in a]] @ E[[idx[d] for d in b]].T

def distinct_mean(a, b):
    s = sim(a, b)
    mask = np.array(a)[:, None] != np.array(b)[None, :]
    return s[mask].mean() if mask.any() else np.nan

# Analyse all queries, seeds and retrieval rounds
rows = []

for (qid, seed), group in retrieval.groupby(["query_id", "seed"]):
    seen = set()
    prev = None
    g = gold[qid]

    for r, df in group.groupby("round", sort=True):
        docs = df.sort_values("rank").doc_id.tolist()
        assert len(docs) == len(set(docs)) == 5

        s = sim(docs, docs)
        within = s[np.triu_indices(len(docs), 1)].mean()

        new = sorted(set(docs) - seen)
        hits = len(set(docs) & set(g))

        rows.append({
            "query_id": qid,
            "seed": seed,
            "round": r,
            "within_similarity": within,
            "cross_similarity": sim(docs, prev).mean() if prev else np.nan,
            "cross_distinct_ids": distinct_mean(docs, prev) if prev else np.nan,
            "gold_similarity": distinct_mean(docs, g),
            "overlap": len(set(docs) & set(prev)) / 5 if prev else np.nan,
            "semantic_novelty": (
                1 - sim(new, sorted(seen)).max(axis=1)
            ).mean() if new and seen else np.nan,
            "new_documents": len(new),
            "gold_hits": hits,
            "precision_at_5": hits / 5,
            "gold_recall_at_5": hits / len(g)
        })

        seen.update(docs)
        prev = docs

# Aggregate across seeds and queries
results = pd.DataFrame(rows)
assert len(results) == 516

query_means = (
    results.groupby(["query_id", "round"])
    .mean(numeric_only=True)
    .drop(columns="seed")
    .reset_index()
)

summary = query_means.groupby("round").mean(numeric_only=True)
std = query_means.groupby("round").std(numeric_only=True)

# Save reproduced results separately
SAVE = OUT / "reproduced"
SAVE.mkdir(exist_ok=True)

results.to_csv(SAVE / "semantic_similarity_per_seed.csv", index=False)
query_means.to_csv(SAVE / "semantic_similarity_per_query.csv", index=False)
summary.to_csv(SAVE / "semantic_similarity_summary.csv")
std.to_csv(SAVE / "semantic_similarity_std.csv")

print("Queries:", results.query_id.nunique())
print("Seeds:", results.seed.nunique())
print("Observations:", len(results))

print("\nMEAN PER ROUND")
print(summary.round(4).T)

print("\nSTD ACROSS QUERIES")
print(std.round(4).T)

print("\nNO NEW DOCUMENTS (%)")
print(
    results[results["round"] > 0]
    .groupby("round").new_documents
    .apply(lambda x: 100 * x.eq(0).mean()).round(2)
)

print("\nVALID NOVELTY OBSERVATIONS")
print(results.groupby("round").semantic_novelty.count())

print("\nANALYSIS COMPLETED")
