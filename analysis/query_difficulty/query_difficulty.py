
from pathlib import Path
from collections import Counter
from statistics import mean, median, stdev
import subprocess
import sys
import json
import csv


# 1. CONFIGURATION

BASE = Path("/scratch-shared/scur0900/thinkqe")
OUTPUTS = BASE / "outputs"

DOMAINS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
]

MODEL = "DeepSeek-R1-Distill-Qwen-14B"
SEEDS = [0, 1, 2]
ROUNDS = range(4)

QRELS_DIR = BASE / "shared_data/bright_data/pyserini_qrels"

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

QUERY_OUTPUT = RESULTS / "query_difficulty_bright.csv"
SUMMARY_OUTPUT = RESULTS / "run_summary.csv"

STRICT = True
TOLERANCE = 0.01


# 2. INPUT HELPERS

def load_qrels(path):
    qrels = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            qid, _, docid, relevance = line.split()
            qrels.setdefault(qid, {})[docid] = int(relevance)

    return qrels


def load_queries(path):
    queries = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            qid, query = json.loads(line)
            queries[str(qid)] = query

    return queries


def get_run_dir(domain, seed):
    return OUTPUTS / (
        f"{domain}_{MODEL}__batched_s{seed}"
    )


def get_trec_file(run_dir, round_num):
    return run_dir / (
        f"bm25-aug{round_num}_result_retrieval.trec"
    )


def evaluate_round(qrels_path, trec_path):
    command = [
        sys.executable,
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        "-q",
        "-m",
        "ndcg_cut.10",
        str(qrels_path),
        str(trec_path),
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

        if len(parts) == 3 and parts[0] == "ndcg_cut_10":
            scores[parts[1]] = float(parts[2])

    if "all" not in scores:
        raise RuntimeError(
            f"Missing aggregate evaluation: {trec_path}"
        )

    return scores


def classify(improvement):
    if improvement > 1e-9:
        return "win"
    if improvement < -1e-9:
        return "loss"
    return "tie"


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# 3. ANALYZE ONE DOMAIN AND SEED

def analyze_run(domain, seed):
    run_dir = get_run_dir(domain, seed)
    qrels_path = QRELS_DIR / f"{domain}.tsv"

    topics_path = (
        run_dir
        / "bm25-aug0_result_retrieval.trec.topics.jsonl"
    )

    required_files = [
        qrels_path,
        topics_path,
        *[
            get_trec_file(run_dir, r)
            for r in ROUNDS
        ],
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    qrels = load_qrels(qrels_path)
    queries = load_queries(topics_path)

    if set(queries) != set(qrels):
        raise ValueError(
            f"Query ID mismatch: {domain}, seed {seed}. "
            f"Topics={len(queries)}, qrels={len(qrels)}"
        )

    official_scores = {}

    for round_num in ROUNDS:
        print(
            f"  Evaluating round {round_num}...",
            flush=True,
        )

        official_scores[round_num] = evaluate_round(
            qrels_path,
            get_trec_file(run_dir, round_num),
        )

    # Verify every query has a score in every round.
    for round_num in ROUNDS:
        evaluated = (
            set(official_scores[round_num]) - {"all"}
        )

        if evaluated != set(qrels):
            raise ValueError(
                f"Coverage mismatch: {domain}, seed {seed}, "
                f"round {round_num}"
            )

    # Validate against ThinkQE's saved aggregate metrics.
    for round_num in ROUNDS:
        metrics_path = (
            run_dir
            / f"bm25-aug{round_num}_result_retrieval.trec.metrics.json"
        )

        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)

        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)

        saved = float(metrics["ndcg@10"])
        evaluated = official_scores[round_num]["all"] * 100

        if abs(saved - evaluated) >= TOLERANCE:
            raise ValueError(
                f"Metric mismatch: {domain}, seed {seed}, "
                f"round {round_num}: "
                f"saved={saved:.4f}, evaluated={evaluated:.4f}"
            )

    rows = []

    for qid, query in queries.items():
        scores = {
            r: official_scores[r][qid]
            for r in ROUNDS
        }

        improvement = scores[3] - scores[0]

        rows.append({
            "domain": domain,
            "seed": seed,
            "query_id": qid,
            "query_length": len(query.split()),
            "bm25_ndcg10": scores[0],
            "aug1_ndcg10": scores[1],
            "aug2_ndcg10": scores[2],
            "aug3_ndcg10": scores[3],
            "improvement": improvement,
            "outcome": classify(improvement),
        })

    improvements = [r["improvement"] for r in rows]
    counts = Counter(r["outcome"] for r in rows)

    summary = {
        "domain": domain,
        "seed": seed,
        "n_queries": len(rows),
        "bm25_mean": official_scores[0]["all"],
        "aug1_mean": official_scores[1]["all"],
        "aug2_mean": official_scores[2]["all"],
        "aug3_mean": official_scores[3]["all"],
        "mean_improvement": mean(improvements),
        "median_improvement": median(improvements),
        "std_improvement": (
            stdev(improvements)
            if len(improvements) > 1 else 0.0
        ),
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
    }

    print(
        f"  Validated {len(rows)} queries | "
        f"BM25={summary['bm25_mean']:.4f} | "
        f"R3={summary['aug3_mean']:.4f} | "
        f"Delta={summary['mean_improvement']:+.4f}"
    )

    return rows, summary


# 4. RUN ALL STANDARD EXPERIMENTS

def main():
    all_rows = []
    all_summaries = []
    failed_runs = []

    for domain in DOMAINS:
        print(f"\n{domain.upper()}")

        for seed in SEEDS:
            print(f"Seed {seed}")

            try:
                rows, summary = analyze_run(domain, seed)
            except (
                FileNotFoundError,
                ValueError,
                RuntimeError,
                subprocess.CalledProcessError,
            ) as exc:
                print(f"  FAILED: {exc}")
                failed_runs.append((domain, seed, str(exc)))

                if STRICT:
                    raise
                continue

            all_rows.extend(rows)
            all_summaries.append(summary)

    expected_runs = len(DOMAINS) * len(SEEDS)

    if len(all_summaries) != expected_runs:
        raise RuntimeError(
            f"Incomplete analysis: {len(all_summaries)} "
            f"of {expected_runs} runs succeeded. "
            "No combined CSV was overwritten."
        )

    query_columns = [
        "domain", "seed", "query_id", "query_length",
        "bm25_ndcg10", "aug1_ndcg10",
        "aug2_ndcg10", "aug3_ndcg10",
        "improvement", "outcome",
    ]

    summary_columns = [
        "domain", "seed", "n_queries",
        "bm25_mean", "aug1_mean",
        "aug2_mean", "aug3_mean",
        "mean_improvement", "median_improvement",
        "std_improvement",
        "wins", "ties", "losses",
    ]

    save_csv(QUERY_OUTPUT, all_rows, query_columns)
    save_csv(SUMMARY_OUTPUT, all_summaries, summary_columns)

    print("\nANALYSIS COMPLETED")
    print(f"Validated runs: {len(all_summaries)}")
    print(f"Total query-run observations: {len(all_rows)}")
    print(f"Per-query results: {QUERY_OUTPUT}")
    print(f"Run summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
