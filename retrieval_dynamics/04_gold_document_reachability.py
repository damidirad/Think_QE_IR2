# Evaluates gold-document discovery, retention, loss, and recovery across ThinkQE retrieval rounds at cutoffs 5, 10, 100, and 1000.
# Computes per-round and cumulative recall using TREC DL19 relevance judgments, then aggregates results across queries and seeds.

import pandas as pd
from pathlib import Path

BASE = Path("/scratch-shared/scur0915/embedding_analysis")
RUNS = Path("/scratch-shared/scur0900/thinkqe/outputs")
OUT = BASE / "dl19_gold_reachability"

OUT.mkdir(parents=True, exist_ok=True)

qrels = pd.read_csv(
    BASE / "ground_truth_qrels.csv",
    dtype={"query_id": str, "doc_id": str}
)

gold = (
    qrels[(qrels.dataset == "dl19") & (qrels.relevance >= 2)]
    .groupby("query_id").doc_id.apply(set).to_dict()
)

assert len(gold) == 43

cutoffs = [5, 10, 100, 1000]
rows = []

for seed in range(3):

    print("Processing seed", seed)

    run = RUNS / f"dl19_Qwen3.8-27B__batched_s{seed}"
    rankings = {}

    for r in range(4):

        df = pd.read_csv(
            run / f"bm25-aug{r}_result_retrieval.trec",
            sep=r"\s+",
            names=["query_id", "Q0", "doc_id", "rank", "score", "tag"],
            dtype={"query_id": str, "doc_id": str}
        )

        df = df[df.query_id.isin(gold)]

        assert df.query_id.nunique() == 43

        rankings[r] = {
            qid: group.sort_values("rank")
            for qid, group in df.groupby("query_id")
        }

    for qid, g in gold.items():

        for k in cutoffs:

            seen = set()
            prev = set()

            for r in range(4):

                df = rankings[r][qid]
                docs = set(df[df["rank"] <= k].doc_id)

                current = docs & g

                new = current - seen
                lost = prev - current
                retained = current & prev
                recovered = (current & seen) - prev

                seen.update(current)

                rows.append({
                    "query_id": qid,
                    "seed": seed,
                    "cutoff": k,
                    "round": r,
                    "total_gold": len(g),
                    "current_gold": len(current),
                    "new_gold": len(new),
                    "retained_gold": len(retained),
                    "lost_gold": len(lost),
                    "recovered_gold": len(recovered),
                    "cumulative_gold": len(seen),
                    "remaining_unseen_gold": len(g - seen),
                    "recall": len(current) / len(g),
                    "cumulative_recall": len(seen) / len(g)
                })

                prev = current

results = pd.DataFrame(rows)

assert len(results) == 43 * 3 * 4 * 4

query_means = (
    results.groupby(["query_id", "cutoff", "round"])
    .mean(numeric_only=True)
    .drop(columns="seed")
    .reset_index()
)

summary = (
    query_means.groupby(["cutoff", "round"])
    .mean(numeric_only=True)
)

results.to_csv(
    OUT / "gold_reachability_per_seed.csv", index=False
)

query_means.to_csv(
    OUT / "gold_reachability_per_query.csv", index=False
)

summary.to_csv(
    OUT / "gold_reachability_summary.csv"
)

for k in cutoffs:

    print(f"\n TOP {k} ")
    print(summary.loc[k].round(4).T.to_string())

print("\nTotal observations:", len(results))
print("Saved to:", OUT)
print("GOLD REACHABILITY ANALYSIS COMPLETED")
