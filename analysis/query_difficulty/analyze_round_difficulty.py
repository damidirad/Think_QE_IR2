
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures" / "round_difficulty"

FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "query_difficulty_bright.csv"

DOMAINS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
]

SEEDS = {0, 1, 2}

ROUNDS = [
    "bm25_ndcg10",
    "aug1_ndcg10",
    "aug2_ndcg10",
    "aug3_ndcg10",
]

DIFFICULTY_ORDER = ["Zero", "Low", "Medium", "High"]

EPS = 1e-9


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype={"query_id": str},
)

required = [
    "domain",
    "seed",
    "query_id",
    *ROUNDS,
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

if df[required].isna().any().any():
    raise ValueError(
        "Missing values in required columns."
    )

if df.duplicated(
    ["domain", "query_id", "seed"]
).any():
    raise ValueError(
        "Duplicate domain/query/seed observations."
    )

if set(df["domain"]) != set(DOMAINS):
    raise ValueError(
        "Unexpected domain coverage: "
        f"{sorted(df['domain'].unique())}"
    )

if set(df["seed"]) != SEEDS:
    raise ValueError(
        f"Expected seeds {sorted(SEEDS)}."
    )

seed_coverage = (
    df.groupby(["domain", "query_id"])["seed"]
    .agg(set)
)

if not seed_coverage.apply(
    lambda seeds: seeds == SEEDS
).all():
    raise ValueError(
        "Some queries are missing one or more seeds."
    )

if (
    (df[ROUNDS] < 0) |
    (df[ROUNDS] > 1)
).any().any():
    raise ValueError(
        "nDCG@10 must be between 0 and 1."
    )


# ============================================================
# 3. RECONSTRUCT INITIAL BM25 DIFFICULTY
# ============================================================

# Same thresholds as analyze_difficulty_groups.py:
#
# Zero:   BM25 = 0
# Low:    0 < BM25 <= 0.25
# Medium: 0.25 < BM25 <= 0.50
# High:   BM25 > 0.50


def difficulty_group(score):

    if score <= EPS:
        return "Zero"

    if score <= 0.25:
        return "Low"

    if score <= 0.50:
        return "Medium"

    return "High"


# Check that the original BM25 result is identical
# for each query across the three seeds.

baseline_variation = (
    df.groupby(["domain", "query_id"])[
        "bm25_ndcg10"
    ]
    .agg(lambda x: x.max() - x.min())
)

if (baseline_variation > EPS).any():
    raise ValueError(
        "BM25 baseline differs across seeds. "
        "Inspect the input before assigning difficulty."
    )


df["difficulty"] = (
    df["bm25_ndcg10"].apply(difficulty_group)
)

df["difficulty"] = pd.Categorical(
    df["difficulty"],
    categories=DIFFICULTY_ORDER,
    ordered=True,
)


# ============================================================
# 4. COMPUTE ROUND-TO-ROUND CHANGES
# ============================================================

# Marginal effectiveness of each additional expansion.

df["delta_r0_r1"] = (
    df["aug1_ndcg10"] -
    df["bm25_ndcg10"]
)

df["delta_r1_r2"] = (
    df["aug2_ndcg10"] -
    df["aug1_ndcg10"]
)

df["delta_r2_r3"] = (
    df["aug3_ndcg10"] -
    df["aug2_ndcg10"]
)


# Did the second expansion help or hurt?

df["r2_helped"] = (
    df["delta_r1_r2"] > EPS
)

df["r2_hurt"] = (
    df["delta_r1_r2"] < -EPS
)


# Did the third expansion help or hurt?

df["r3_helped"] = (
    df["delta_r2_r3"] > EPS
)

df["r3_hurt"] = (
    df["delta_r2_r3"] < -EPS
)


# Did the final retrieval improve on the original BM25?

df["final_improvement"] = (
    df["aug3_ndcg10"] -
    df["bm25_ndcg10"]
)


# ============================================================
# 5. SUMMARY BY INITIAL DIFFICULTY
# ============================================================

summary = (
    df.groupby("difficulty", observed=True)
    .agg(
        observations=("query_id", "size"),

        mean_r0_r1=("delta_r0_r1", "mean"),
        mean_r1_r2=("delta_r1_r2", "mean"),
        mean_r2_r3=("delta_r2_r3", "mean"),

        r2_help_pct=("r2_helped", "mean"),
        r2_hurt_pct=("r2_hurt", "mean"),

        r3_help_pct=("r3_helped", "mean"),
        r3_hurt_pct=("r3_hurt", "mean"),

        mean_final_improvement=(
            "final_improvement", "mean"
        ),
    )
    .reset_index()
)

percentage_columns = [
    "r2_help_pct",
    "r2_hurt_pct",
    "r3_help_pct",
    "r3_hurt_pct",
]

summary[percentage_columns] *= 100

# Add number of unique queries in each group.

unique_counts = (
    df.drop_duplicates(["domain", "query_id"])
    .groupby("difficulty", observed=True)
    .size()
    .rename("unique_queries")
    .reset_index()
)

summary = summary.merge(
    unique_counts,
    on="difficulty",
    validate="one_to_one",
)

summary.to_csv(
    RESULTS / "round_difficulty_summary.csv",
    index=False,
)


# ============================================================
# 6. FIGURE A: DOES R3 HELP OR HURT?
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(summary))
width = 0.35

ax.bar(
    x - width / 2,
    summary["r3_help_pct"],
    width,
    label="R3 improves over R2",
)

ax.bar(
    x + width / 2,
    summary["r3_hurt_pct"],
    width,
    label="R3 worsens R2",
)

ax.set_xticks(
    x,
    summary["difficulty"].astype(str),
)

ax.set_ylim(0, 100)

ax.set_xlabel(
    "Initial BM25 difficulty group"
)

ax.set_ylabel(
    "Query-seed observations (%)"
)

ax.set_title(
    "Effect of the third expansion round by difficulty"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES / "r3_help_vs_hurt.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 7. FIGURE B: MARGINAL BENEFIT OF ADDITIONAL ROUNDS
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    x - width / 2,
    summary["mean_r1_r2"],
    width,
    label="R1 to R2",
)

ax.bar(
    x + width / 2,
    summary["mean_r2_r3"],
    width,
    label="R2 to R3",
)

ax.axhline(
    0,
    color="gray",
    linestyle="--",
)

ax.set_xticks(
    x,
    summary["difficulty"].astype(str),
)

ax.set_xlabel(
    "Initial BM25 difficulty group"
)

ax.set_ylabel(
    "Mean change in nDCG@10"
)

ax.set_title(
    "Marginal effectiveness of later expansion rounds"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES / "marginal_round_gains.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 8. SUMMARY BY DOMAIN
# ============================================================

domain_summary = (
    df.groupby("domain")
    .agg(
        observations=("query_id", "size"),

        mean_r1_r2=("delta_r1_r2", "mean"),
        mean_r2_r3=("delta_r2_r3", "mean"),

        r3_help_pct=("r3_helped", "mean"),
        r3_hurt_pct=("r3_hurt", "mean"),

        mean_final_improvement=(
            "final_improvement", "mean"
        ),
    )
    .reset_index()
)

domain_summary[
    ["r3_help_pct", "r3_hurt_pct"]
] *= 100

domain_summary["unique_queries"] = (
    domain_summary["observations"] // len(SEEDS)
)

domain_summary.to_csv(
    RESULTS / "round_domain_summary.csv",
    index=False,
)


# ============================================================
# 9. FIGURE C: R3 EFFECTIVENESS BY DOMAIN
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

x_domain = np.arange(
    len(domain_summary)
)

ax.bar(
    x_domain - width / 2,
    domain_summary["r3_help_pct"],
    width,
    label="R3 improves over R2",
)

ax.bar(
    x_domain + width / 2,
    domain_summary["r3_hurt_pct"],
    width,
    label="R3 worsens R2",
)

ax.set_xticks(
    x_domain,
    domain_summary["domain"],
    rotation=30,
    ha="right",
)

ax.set_ylim(0, 100)

ax.set_ylabel(
    "Query-seed observations (%)"
)

ax.set_xlabel(
    "BRIGHT domain"
)

ax.set_title(
    "Effect of the third expansion round across domains"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES / "r3_effect_by_domain.png",
    dpi=200,
)

plt.close(fig)



# 10. SAVE QUERY-SEED RESULTS


df.to_csv(
    RESULTS / "round_difficulty_query_seed.csv",
    index=False,
)



# 11. PRINT RESULTS


unique_queries = (
    df.groupby(["domain", "query_id"]).ngroups
)

print("=" * 65)
print("THINKQE: ROUND DIFFICULTY ANALYSIS")
print("=" * 65)

print(f"Domains: {df['domain'].nunique()}")
print(f"Seeds: {sorted(SEEDS)}")
print(f"Unique queries: {unique_queries}")
print(f"Query-seed observations: {len(df)}")

print("\nDIFFICULTY GROUP COVERAGE")
print(
    unique_counts.to_string(index=False)
)

print("\nDIFFICULTY SUMMARY")
print(
    summary.round(4).to_string(index=False)
)

print("\nDOMAIN SUMMARY")
print(
    domain_summary.round(4).to_string(index=False)
)

print("\nSAVED OUTPUTS")
print(RESULTS / "round_difficulty_query_seed.csv")
print(RESULTS / "round_difficulty_summary.csv")
print(RESULTS / "round_domain_summary.csv")
print(f"Figures: {FIGURES}")

print("\nANALYSIS COMPLETED")
