
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures" / "optimal_round"
FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "query_difficulty_bright.csv"

ROUNDS = [
    "bm25_ndcg10",
    "aug1_ndcg10",
    "aug2_ndcg10",
    "aug3_ndcg10",
]

ROUND_LABELS = ["BM25", "R1", "R2", "R3"]
DIFFICULTY_ORDER = ["Zero", "Low", "Medium", "High"]

EXPECTED_SEEDS = {0, 1, 2}
EPS = 1e-9


# ============================================================
# 2. LOAD AND VALIDATE EXISTING RESULTS
# ============================================================

df = pd.read_csv(INPUT, dtype={"query_id": str})

required = ["domain", "query_id", "seed", *ROUNDS]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[required].isna().any().any():
    raise ValueError("Missing required values.")

if df.duplicated(["domain", "query_id", "seed"]).any():
    raise ValueError("Duplicate query-seed observations.")

coverage = df.groupby(
    ["domain", "query_id"]
)["seed"].agg(set)

if not coverage.apply(
    lambda seeds: seeds == EXPECTED_SEEDS
).all():
    raise ValueError("Incomplete seed coverage.")

if ((df[ROUNDS] < 0) | (df[ROUNDS] > 1)).any().any():
    raise ValueError("Invalid nDCG@10 values.")

baseline_range = df.groupby(
    ["domain", "query_id"]
)["bm25_ndcg10"].agg(
    lambda x: x.max() - x.min()
)

if (baseline_range > EPS).any():
    raise ValueError(
        "BM25 baseline differs across seeds."
    )


# ============================================================
# 3. INITIAL BM25 DIFFICULTY GROUPS
# ============================================================

def difficulty_group(score):
    if score == 0:
        return "Zero"
    if score <= 0.25:
        return "Low"
    if score <= 0.50:
        return "Medium"
    return "High"


df["difficulty"] = df["bm25_ndcg10"].apply(
    difficulty_group
)

df["difficulty"] = pd.Categorical(
    df["difficulty"],
    categories=DIFFICULTY_ORDER,
    ordered=True,
)


# ============================================================
# 4. FIND EARLIEST BEST ROUND
# ============================================================

scores = df[ROUNDS].to_numpy(dtype=float)

# Best observed nDCG@10 for each query-seed.
best_score = scores.max(axis=1)

# Treat near-equal scores as ties.
ties = np.abs(scores - best_score[:, None]) <= EPS

# argmax on a boolean array selects the earliest True.
best_round = ties.argmax(axis=1)

df["best_round"] = best_round
df["best_round_label"] = [
    ROUND_LABELS[i] for i in best_round
]

df["best_ndcg10"] = best_score

# Compare oracle best with the fixed three-round system.
df["oracle_gain_vs_r3"] = (
    df["best_ndcg10"] - df["aug3_ndcg10"]
)

# Compare oracle best with the original baseline.
df["oracle_gain_vs_bm25"] = (
    df["best_ndcg10"] - df["bm25_ndcg10"]
)

# How much does the best round vary across seeds?
df["best_is_r3"] = df["best_round"] == 3


# ============================================================
# 5. BEST ROUND DISTRIBUTION BY DIFFICULTY
# ============================================================

distribution = pd.crosstab(
    df["difficulty"],
    df["best_round_label"],
    normalize="index",
) * 100

distribution = distribution.reindex(
    index=DIFFICULTY_ORDER,
    columns=ROUND_LABELS,
    fill_value=0,
)

distribution.to_csv(
    RESULTS / "optimal_round_distribution.csv"
)


# ============================================================
# 6. FIGURE A: BEST ROUND BY DIFFICULTY
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(distribution))

for round_name in ROUND_LABELS:

    values = distribution[round_name].to_numpy()

    ax.bar(
        distribution.index,
        values,
        bottom=bottom,
        label=round_name,
    )

    bottom += values

ax.set_ylim(0, 100)
ax.set_ylabel("Query-seed observations (%)")
ax.set_xlabel("Initial BM25 difficulty group")

ax.set_title(
    "Earliest best retrieval round by initial difficulty"
)

ax.legend(title="Best round")
fig.tight_layout()

fig.savefig(
    FIGURES / "best_round_by_difficulty.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 7. ORACLE GAIN SUMMARY
# ============================================================

oracle_summary = (
    df.groupby("difficulty", observed=True)
    .agg(
        observations=("query_id", "size"),

        mean_bm25=("bm25_ndcg10", "mean"),
        mean_r3=("aug3_ndcg10", "mean"),
        mean_oracle=("best_ndcg10", "mean"),

        mean_gain_vs_r3=(
            "oracle_gain_vs_r3", "mean"
        ),

        mean_gain_vs_bm25=(
            "oracle_gain_vs_bm25", "mean"
        ),
    )
    .reset_index()
)

oracle_summary.to_csv(
    RESULTS / "optimal_round_oracle_summary.csv",
    index=False,
)


# ============================================================
# 8. FIGURE B: POTENTIAL GAIN OVER FIXED R3
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    oracle_summary["difficulty"].astype(str),
    oracle_summary["mean_gain_vs_r3"],
)

ax.axhline(0, color="gray", linestyle="--")

ax.set_xlabel("Initial BM25 difficulty group")
ax.set_ylabel("Mean oracle gain in nDCG@10")

ax.set_title(
    "Potential gain from selecting the best round "
    "instead of always using R3"
)

fig.tight_layout()

fig.savefig(
    FIGURES / "oracle_gain_vs_r3.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 9. SEED CONSISTENCY OF BEST ROUND
# ============================================================

seed_consistency = (
    df.groupby(["domain", "query_id"])
    .agg(
        difficulty=("difficulty", "first"),
        n_distinct_best_rounds=("best_round", "nunique"),
        earliest_best_round=("best_round", "min"),
        latest_best_round=("best_round", "max"),
    )
    .reset_index()
)

seed_consistency["same_best_round_all_seeds"] = (
    seed_consistency["n_distinct_best_rounds"] == 1
)

consistency_summary = (
    seed_consistency.groupby(
        "difficulty", observed=True
    )
    .agg(
        unique_queries=("query_id", "size"),
        same_best_round_pct=(
            "same_best_round_all_seeds", "mean"
        ),
    )
    .reset_index()
)

consistency_summary["same_best_round_pct"] *= 100

consistency_summary.to_csv(
    RESULTS / "optimal_round_seed_consistency.csv",
    index=False,
)


# ============================================================
# 10. SAVE QUERY-SEED OBSERVATIONS
# ============================================================

df.to_csv(
    RESULTS / "optimal_round_query_seed.csv",
    index=False,
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

n_queries = df.groupby(
    ["domain", "query_id"]
).ngroups

print("\n" + "=" * 65)
print("THINKQE: OPTIMAL EXPANSION DEPTH ANALYSIS")
print("=" * 65)

print(f"Unique queries: {n_queries}")
print(f"Query-seed observations: {len(df)}")

print("\nBEST ROUND DISTRIBUTION (%)")
print(distribution.round(2).to_string())

print("\nORACLE GAIN SUMMARY")
print(oracle_summary.round(4).to_string(index=False))

print("\nSEED CONSISTENCY")
print(consistency_summary.round(2).to_string(index=False))

print("\nFIGURES SAVED TO:")
print(FIGURES)

print("\nANALYSIS COMPLETED")
