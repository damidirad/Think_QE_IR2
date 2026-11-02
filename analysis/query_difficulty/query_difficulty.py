
from pathlib import Path
from collections import Counter
from statistics import mean, median, stdev
import subprocess
import sys
import json
import csv


# 1. CONFIGURATION


BASE = Path("/scratch-shared/scur0900/thinkqe")

RUN = (
    BASE
    / "outputs/biology_DeepSeek-R1-Distill-Qwen-14B__batched_s0"
)

QRELS = (
    BASE
    / "shared_data/bright_data/pyserini_qrels/biology.tsv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "query_difficulty.csv"

ROUNDS = range(4)


# 2. LOAD QRELS


def load_qrels(path):
    """Read relevance judgments for each query."""

    qrels = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            qid, _, docid, relevance = line.split()

            qrels.setdefault(qid, {})[docid] = int(relevance)

    return qrels



# 3. OFFICIAL PER-QUERY nDCG@10


def evaluate_round(round_num):
    """
    Run the official Pyserini evaluator.

    Returns a dictionary:
        query_id -> nDCG@10

    The key 'all' contains the aggregate nDCG@10.
    """

    trec_file = (
        RUN
        / f"bm25-aug{round_num}_result_retrieval.trec"
    )

    command = [
        sys.executable,
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        "-q",
        "-m",
        "ndcg_cut.10",
        str(QRELS),
        str(trec_file),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    scores = {}

    for line in result.stdout.splitlines():
        parts = line.split()

        if (
            len(parts) == 3
            and parts[0] == "ndcg_cut_10"
        ):
            qid = parts[1]
            score = float(parts[2])

            scores[qid] = score

    if "all" not in scores:
        raise RuntimeError(
            f"Missing aggregate evaluation for round {round_num}"
        )

    return scores


# 4. LOAD ORIGINAL QUERIES


def load_queries(path):
    """Load the original BRIGHT query texts."""

    queries = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            qid, query = json.loads(line)
            queries[str(qid)] = query

    return queries


# 5. EVALUATE ALL EXPANSION ROUNDS


print("Evaluating ThinkQE retrieval results...")

qrels = load_qrels(QRELS)

query_file = (
    RUN
    / "bm25-aug0_result_retrieval.trec.topics.jsonl"
)

queries = load_queries(query_file)

official_scores = {}

for round_num in ROUNDS:
    print(f"Evaluating round {round_num}...", flush=True)

    official_scores[round_num] = evaluate_round(round_num)



# 6. PER-QUERY DIFFICULTY ANALYSIS


rows = []

for qid, query in queries.items():

    if qid not in qrels:
        continue

    # Require an official score for every round.
    scores = {}

    for round_num in ROUNDS:
        if qid not in official_scores[round_num]:
            raise ValueError(
                f"Query {qid} missing from round {round_num}"
            )

        scores[round_num] = official_scores[round_num][qid]

    baseline = scores[0]
    final = scores[3]

    improvement = final - baseline

    # Positive = expansion helped.
    # Negative = expansion hurt.
    # Zero = no change.
    if improvement > 1e-9:
        outcome = "win"
    elif improvement < -1e-9:
        outcome = "loss"
    else:
        outcome = "tie"

    rows.append({
        "query_id": qid,
        "query_length": len(query.split()),
        "bm25_ndcg10": baseline,
        "aug1_ndcg10": scores[1],
        "aug2_ndcg10": scores[2],
        "aug3_ndcg10": final,
        "improvement": improvement,
        "outcome": outcome,
    })



# 7. SAVE RESULTS


if not rows:
    raise ValueError(
        "No matching queries found in topics and qrels."
    )

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"\nAnalyzed {len(rows)} queries")
print(f"Saved: {OUTPUT_FILE.resolve()}")


# ============================================================
# 8. OFFICIAL EVALUATION SUMMARY
# ============================================================

print("\n--- Mean nDCG@10 by expansion round ---")

for round_num in ROUNDS:

    official_mean = official_scores[round_num]["all"]

    print(
        f"Round {round_num}: {official_mean * 100:.2f}"
    )


# 9. IMPROVEMENT STATISTICS


improvements = [
    row["improvement"]
    for row in rows
]

print("\n--- Improvement statistics (round 3 vs round 0) ---")

print(f"Mean improvement: {mean(improvements):.4f}")
print(f"Median improvement: {median(improvements):.4f}")

if len(improvements) > 1:
    print(
        f"Standard deviation: {stdev(improvements):.4f}"
    )



# 10. WIN / TIE / LOSS


outcomes = Counter(
    row["outcome"]
    for row in rows
)

print("\n--- Query outcomes ---")

for category in ("win", "tie", "loss"):

    count = outcomes[category]

    percentage = count / len(rows) * 100

    print(
        f"{category}: {count} ({percentage:.1f}%)"
    )



# 11. QUERY COVERAGE


print("\n--- Query coverage ---")

print("Queries in qrels:", len(qrels))
print("Queries in topics:", len(queries))
print("Queries analyzed:", len(rows))

for round_num in ROUNDS:

    evaluated = (
        set(official_scores[round_num]) - {"all"}
    )

    missing = set(qrels) - evaluated

    print(
        f"Queries evaluated in round {round_num}: "
        f"{len(evaluated)}"
    )

    print(
        f"Missing query IDs in round {round_num}: "
        f"{sorted(missing)}"
    )



# 12. COMPARE AGAINST SAVED GROUP METRICS


print("\n--- Validation against group metrics ---")

for round_num in ROUNDS:

    metrics_file = (
        RUN
        / (
            f"bm25-aug{round_num}_result_retrieval"
            ".trec.metrics.json"
        )
    )

    if not metrics_file.exists():
        print(
            f"Round {round_num}: saved metrics not found"
        )
        continue

    with metrics_file.open(encoding="utf-8") as f:
        metrics = json.load(f)

    saved_score = float(metrics["ndcg@10"])
    evaluated_score = official_scores[round_num]["all"] * 100

    difference = abs(saved_score - evaluated_score)

    status = "MATCH" if difference < 0.01 else "CHECK"

    print(
        f"Round {round_num}: "
        f"saved={saved_score:.2f}, "
        f"evaluated={evaluated_score:.2f} "
        f"[{status}]"
    )


print("\nAnalysis completed.")
