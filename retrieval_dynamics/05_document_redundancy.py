import json
import pandas as pd
from pathlib import Path

BASE = Path("/scratch-shared/scur0915/embedding_analysis")
RUNS = Path("/scratch-shared/scur0900/thinkqe/outputs")
OUT = BASE / "dl19_document_redundancy"

OUT.mkdir(parents=True, exist_ok=True)

K = 5
rows = []

# Reconstruct ThinkQE's original passage filtering logic
for seed in range(3):

    print("Processing seed", seed)

    run = RUNS / f"dl19_Qwen3.8-27B__batched_s{seed}"
    passages = {}

    # Load original candidate passage texts
    for r in range(3):

        path = run / (
            f"bm25-aug{r}_result_retrieval.trectop-psgs.json"
        )

        with open(path) as f:
            passages[r] = json.load(f)

    qids = sorted(passages[0])

    assert len(qids) == 43
    assert all(
        set(passages[r]) == set(qids)
        for r in range(3)
    )

    for qid in qids:

        blacklist = set()

        for r in range(1, 4):

            candidates = passages[r - 1][qid]

            assert isinstance(candidates, list)
            assert all(isinstance(p, str) for p in candidates)

            # Top-K passages from two retrieval rounds earlier
            history = (
                set(passages[r - 2][qid][:K])
                if r >= 2 else set()
            )

            accepted = []
            inspected = 0
            blacklist_rejections = 0
            new_blacklist_additions = 0

            for passage in candidates:

                inspected += 1

                # Reject already blacklisted passages
                if passage in blacklist:
                    blacklist_rejections += 1
                    continue

                # Reject repeated passages and update blacklist
                if passage in history:
                    blacklist.add(passage)
                    new_blacklist_additions += 1
                    continue

                accepted.append(passage)

                if len(accepted) >= K:
                    break

            rejected = (
                blacklist_rejections
                + new_blacklist_additions
            )

            assert inspected == rejected + len(accepted)

            rows.append({
                "query_id": qid,
                "seed": seed,
                "round": r,
                "available_candidates": len(candidates),
                "inspected_candidates": inspected,
                "accepted_passages": len(accepted),
                "blacklist_rejections": blacklist_rejections,
                "new_blacklist_additions": new_blacklist_additions,
                "total_rejected": rejected,
                "rejection_rate": (
                    rejected / inspected if inspected else 0
                ),
                "blacklist_size": len(blacklist),
                "unique_accepted_passages": len(set(accepted)),
                "replacement_candidates": max(0, inspected - K),
                "incomplete_context": int(len(accepted) < K)
            })

# Validate observations
results = pd.DataFrame(rows)

assert len(results) == 43 * 3 * 3

assert results.loc[
    results["round"] == 1, "total_rejected"
].sum() == 0

# Average across seeds for each query
query_means = (
    results.groupby(["query_id", "round"])
    .mean(numeric_only=True)
    .drop(columns="seed")
    .reset_index()
)

# Average across queries
summary = (
    query_means.groupby("round")
    .mean(numeric_only=True)
)

# Total filtering events across queries and seeds
totals = results.groupby("round")[
    [
        "inspected_candidates",
        "accepted_passages",
        "blacklist_rejections",
        "new_blacklist_additions",
        "total_rejected",
        "incomplete_context"
    ]
].sum()

totals["pooled_rejection_rate"] = (
    totals["total_rejected"]
    / totals["inspected_candidates"]
)

affected = (
    results.groupby("round").total_rejected
    .apply(lambda x: 100 * x.gt(0).mean())
)

# Save results
results.to_csv(
    OUT / "redundancy_per_seed.csv", index=False
)

query_means.to_csv(
    OUT / "redundancy_per_query.csv", index=False
)

summary.to_csv(
    OUT / "redundancy_summary.csv"
)

totals.to_csv(
    OUT / "redundancy_totals.csv"
)

# Display results
print("\n===== MEAN PER QUERY =====")
print(summary.round(4).T.to_string())

print("\n===== TOTAL COUNTS ACROSS SEEDS =====")
print(totals.round(4).T.to_string())

print("\n===== FILTER-AFFECTED GENERATIONS (%) =====")
print(affected.round(2))

print("\nObservations:", len(results))
print("Saved to:", OUT)
print("DOCUMENT REDUNDANCY ANALYSIS COMPLETED")
