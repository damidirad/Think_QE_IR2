
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

DATA_FILE = RESULTS / "query_difficulty.csv"


# ============================================================
# 2. LOAD VALIDATED QUERY RESULTS
# ============================================================

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} queries")

required = [
    "query_length",
    "bm25_ndcg10",
    "aug3_ndcg10",
    "improvement",
    "outcome",
]

if df[required].isna().any().any():
    raise ValueError("Missing values in query difficulty results")


# ============================================================
# 3. QUERY DIFFICULTY VS. IMPROVEMENT
# ============================================================

rho, p = spearmanr(
    df["bm25_ndcg10"],
    df["improvement"]
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["bm25_ndcg10"],
    df["improvement"],
    alpha=0.65,
    s=45,
)

ax.axhline(0, linestyle="--", color="gray")

ax.set_xlabel("Baseline BM25 nDCG@10")
ax.set_ylabel("Improvement (round 3 - baseline)")
ax.set_title("Baseline Retrieval Performance vs. ThinkQE Improvement")

ax.text(
    0.03, 0.97,
    f"Spearman rho = {rho:.3f}\np = {p:.4g}",
    transform=ax.transAxes,
    va="top",
)

fig.tight_layout()
fig.savefig(FIGURES / "difficulty_vs_improvement.png", dpi=200)
plt.close(fig)

print(f"Baseline vs. improvement: rho={rho:.3f}, p={p:.4g}")


# ============================================================
# 4. QUERY LENGTH VS. IMPROVEMENT
# ============================================================

rho_len, p_len = spearmanr(
    df["query_length"],
    df["improvement"]
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["query_length"],
    df["improvement"],
    alpha=0.65,
    s=45,
)

ax.axhline(0, linestyle="--", color="gray")

ax.set_xlabel("Original Query Length (words)")
ax.set_ylabel("Improvement (round 3 - baseline)")
ax.set_title("Query Length vs. ThinkQE Improvement")

ax.text(
    0.03, 0.97,
    f"Spearman rho = {rho_len:.3f}\np = {p_len:.4g}",
    transform=ax.transAxes,
    va="top",
)

fig.tight_layout()
fig.savefig(FIGURES / "length_vs_improvement.png", dpi=200)
plt.close(fig)

print(f"Length vs. improvement: rho={rho_len:.3f}, p={p_len:.4g}")


# ============================================================
# 5. WIN / TIE / LOSS DISTRIBUTION
# ============================================================

counts = (
    df["outcome"]
    .value_counts()
    .reindex(["win", "tie", "loss"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    counts.index,
    counts.values,
)

for bar, count in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(count),
        ha="center",
    )

ax.set_xlabel("Outcome")
ax.set_ylabel("Number of Queries")
ax.set_title("ThinkQE Query Outcomes (Round 3 vs. Baseline)")
ax.set_ylim(0, max(counts.values) * 1.15)

fig.tight_layout()
fig.savefig(FIGURES / "query_outcomes.png", dpi=200)
plt.close(fig)


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n--- Summary ---")
print(f"Queries analyzed: {len(df)}")
print(f"Wins: {counts['win']}")
print(f"Ties: {counts['tie']}")
print(f"Losses: {counts['loss']}")
print(f"\nFigures saved to: {FIGURES}")
