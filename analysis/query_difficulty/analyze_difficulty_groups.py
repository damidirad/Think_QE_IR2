
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# 1. CONFIGURATION


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures" / "difficulty_groups"
FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "query_difficulty_bright.csv"

SEEDS = {0, 1, 2}
TOL = 1e-9

GROUPS = ["Zero", "Low", "Medium", "High"]

df = pd.read_csv(INPUT, dtype={"query_id": str})

required = [
    "domain", "seed", "query_id",
    "bm25_ndcg10", "aug3_ndcg10",
    "improvement",
]

if df[required].isna().any().any():
    raise ValueError("Missing values in input data")

if df.duplicated(["domain", "query_id", "seed"]).any():
    raise ValueError("Duplicate query-seed observations")

coverage = df.groupby(
    ["domain", "query_id"]
)["seed"].agg(set)

if not coverage.apply(lambda s: s == SEEDS).all():
    raise ValueError("Incomplete seed coverage")

if not np.allclose(
    df["improvement"],
    df["aug3_ndcg10"] - df["bm25_ndcg10"],
    atol=1e-6,
):
    raise ValueError("Inconsistent improvement values")


# ============================================================
# 2. VERIFY BASELINE CONSISTENCY
# ============================================================

# BM25 should ordinarily be deterministic for the same
# query and retrieval configuration across seeds.

baseline_range = (
    df.groupby(["domain", "query_id"])["bm25_ndcg10"]
    .agg(lambda x: x.max() - x.min())
)

if (baseline_range > 1e-6).any():
    raise ValueError(
        "Baseline BM25 scores differ across seeds. "
        "Check that runs use the same retrieval setup."
    )


# 3. CLASSIFY EACH QUERY-SEED RESULT


df["win"] = df["improvement"] > TOL
df["loss"] = df["improvement"] < -TOL
df["tie"] = ~(df["win"] | df["loss"])

df["rescued"] = (
    (df["bm25_ndcg10"] <= TOL)
    & (df["aug3_ndcg10"] > TOL)
)



# 4. AGGREGATE TO ONE ROW PER UNIQUE QUERY


queries = (
    df.groupby(["domain", "query_id"], as_index=False)
    .agg(
        baseline=("bm25_ndcg10", "mean"),
        final_mean=("aug3_ndcg10", "mean"),
        mean_improvement=("improvement", "mean"),
        improvement_sd=("improvement", "std"),
        winning_seeds=("win", "sum"),
        losing_seeds=("loss", "sum"),
        tied_seeds=("tie", "sum"),
        rescued_seeds=("rescued", "sum"),
    )
)

queries["consistent_win"] = (
    queries["winning_seeds"] == 3
)

queries["consistent_loss"] = (
    queries["losing_seeds"] == 3
)

queries["any_loss"] = (
    queries["losing_seeds"] > 0
)

queries["mixed_outcome"] = (
    (queries["winning_seeds"] > 0)
    & (queries["losing_seeds"] > 0)
)

queries["rescued_any"] = (
    queries["rescued_seeds"] > 0
)

queries["rescued_all"] = (
    queries["rescued_seeds"] == 3
)



# 5. ASSIGN INITIAL DIFFICULTY GROUP


def difficulty_group(score):
    if score <= TOL:
        return "Zero"
    if score <= 0.25:
        return "Low"
    if score <= 0.50:
        return "Medium"
    return "High"


queries["difficulty"] = queries["baseline"].apply(
    difficulty_group
)

queries["difficulty"] = pd.Categorical(
    queries["difficulty"],
    categories=GROUPS,
    ordered=True,
)


# 6. BUILD DIFFICULTY GROUP SUMMARY


summary = (
    queries.groupby(
        ["domain", "difficulty"],
        observed=True,
    )
    .agg(
        n_queries=("query_id", "size"),
        mean_baseline=("baseline", "mean"),
        mean_final=("final_mean", "mean"),
        mean_improvement=("mean_improvement", "mean"),
        median_improvement=("mean_improvement", "median"),
        mean_seed_sd=("improvement_sd", "mean"),
        consistent_wins=("consistent_win", "sum"),
        consistent_losses=("consistent_loss", "sum"),
        any_losses=("any_loss", "sum"),
        mixed_outcomes=("mixed_outcome", "sum"),
        rescued_any=("rescued_any", "sum"),
        rescued_all=("rescued_all", "sum"),
    )
    .reset_index()
)

for column in [
    "consistent_wins",
    "consistent_losses",
    "any_losses",
    "mixed_outcomes",
    "rescued_any",
    "rescued_all",
]:
    summary[column + "_pct"] = (
        100 * summary[column] / summary["n_queries"]
    )


# 7. SAVE DATA


queries.to_csv(
    RESULTS / "query_difficulty_groups.csv",
    index=False,
)

summary.to_csv(
    RESULTS / "difficulty_group_summary.csv",
    index=False,
)



# 8. FIGURE A: IMPROVEMENT BY DIFFICULTY


fig, ax = plt.subplots(figsize=(9, 5))

data = [
    queries.loc[
        queries["difficulty"] == group,
        "mean_improvement",
    ].to_numpy()
    for group in GROUPS
]

ax.boxplot(
    data,
    tick_labels=GROUPS,
    showfliers=True,
)

ax.axhline(0, color="gray", linestyle="--")

ax.set(
    xlabel="Initial BM25 difficulty group",
    ylabel="Mean nDCG@10 improvement across seeds",
    title="ThinkQE improvement by initial difficulty",
)

fig.tight_layout()
fig.savefig(
    FIGURES / "improvement_by_difficulty.png",
    dpi=200,
)
plt.close(fig)



# 9. FIGURE B: CONSISTENT IMPROVEMENT BY DIFFICULTY


rates = (
    queries.groupby("difficulty", observed=True)
    ["consistent_win"]
    .mean()
    .reindex(GROUPS)
    * 100
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(GROUPS, rates)

ax.bar_label(bars, fmt="%.1f%%", padding=3)

ax.set_ylim(0, 100)

ax.set(
    xlabel="Initial BM25 difficulty group",
    ylabel="Queries improved in all 3 seeds (%)",
    title="Consistency of ThinkQE improvement",
)

fig.tight_layout()
fig.savefig(
    FIGURES / "consistent_improvement.png",
    dpi=200,
)
plt.close(fig)


# 10. FIGURE C: RESCUE RATE BY DOMAIN


zero_queries = queries[
    queries["difficulty"] == "Zero"
].copy()

rescue = (
    zero_queries.groupby("domain")
    .agg(
        total=("query_id", "size"),
        rescued_any=("rescued_any", "sum"),
        rescued_all=("rescued_all", "sum"),
    )
)

rescue["any_pct"] = (
    100 * rescue["rescued_any"] / rescue["total"]
)

rescue["all_pct"] = (
    100 * rescue["rescued_all"] / rescue["total"]
)

fig, ax = plt.subplots(figsize=(11, 5))

x = np.arange(len(rescue))
width = 0.36

ax.bar(
    x - width / 2,
    rescue["any_pct"],
    width,
    label="Rescued in at least 1 seed",
)

ax.bar(
    x + width / 2,
    rescue["all_pct"],
    width,
    label="Rescued in all 3 seeds",
)

ax.set_xticks(
    x,
    [d.replace("_", " ").title() for d in rescue.index],
    rotation=30,
    ha="right",
)

ax.set_ylim(0, 100)

ax.set(
    ylabel="Baseline-zero queries rescued (%)",
    title="ThinkQE rescue reliability across domains",
)

ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES / "rescue_by_domain.png",
    dpi=200,
)
plt.close(fig)



# 11. FIGURE D: DEGRADATION BY DIFFICULTY


degradation = (
    queries.groupby("difficulty", observed=True)
    .agg(
        any_loss=("any_loss", "mean"),
        consistent_loss=("consistent_loss", "mean"),
    )
    .reindex(GROUPS)
    * 100
)

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(GROUPS))
width = 0.36

ax.bar(
    x - width / 2,
    degradation["any_loss"],
    width,
    label="Degraded in at least 1 seed",
)

ax.bar(
    x + width / 2,
    degradation["consistent_loss"],
    width,
    label="Degraded in all 3 seeds",
)

ax.set_xticks(x, GROUPS)
ax.set_ylim(0, 100)

ax.set(
    xlabel="Initial BM25 difficulty group",
    ylabel="Queries with degradation (%)",
    title="ThinkQE degradation by initial difficulty",
)

ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES / "degradation_by_difficulty.png",
    dpi=200,
)
plt.close(fig)



# 12. PRINT SUMMARY


print("\n" + "=" * 65)
print("QUERY DIFFICULTY GROUP ANALYSIS")
print("=" * 65)

print(f"Unique queries: {len(queries)}")
print(f"Domains: {queries['domain'].nunique()}")
print(f"Seeds: {sorted(SEEDS)}")

print("\nDIFFICULTY DISTRIBUTION")

print(
    queries["difficulty"]
    .value_counts()
    .reindex(GROUPS)
    .to_string()
)

print("\nQUERY OUTCOMES")

print(
    f"Consistently improved: "
    f"{queries['consistent_win'].sum()}"
)

print(
    f"Consistently degraded: "
    f"{queries['consistent_loss'].sum()}"
)

print(
    f"Degraded in at least one seed: "
    f"{queries['any_loss'].sum()}"
)

print(
    f"Mixed wins and losses across seeds: "
    f"{queries['mixed_outcome'].sum()}"
)

print("\nBASELINE-ZERO RESCUE")

print(
    f"Initially unsuccessful: {len(zero_queries)}"
)

print(
    f"Rescued in at least one seed: "
    f"{zero_queries['rescued_any'].sum()}"
)

print(
    f"Rescued in all three seeds: "
    f"{zero_queries['rescued_all'].sum()}"
)

print("\nSaved:")
print(RESULTS / "query_difficulty_groups.csv")
print(RESULTS / "difficulty_group_summary.csv")
print(FIGURES)
