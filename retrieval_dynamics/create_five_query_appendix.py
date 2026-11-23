
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/scratch-shared/scur0915/embedding_analysis")
SHARED = Path("/scratch-shared/scur0900/thinkqe/outputs")

MODEL = "Qwen3.8-27B"
SEED = 0
ROUNDS = [0, 1, 2, 3]

RUN = SHARED / f"dl19_{MODEL}__batched_s{SEED}"
OUT = BASE / "dl19_five_query_case_studies"
OUT.mkdir(parents=True, exist_ok=True)


def get_final_answer(value):
    """Remove explicit thinking blocks and retain the final answer."""

    if isinstance(value, dict):
        for key in ["final", "answer", "response", "text"]:
            if key in value:
                return get_final_answer(value[key])
        return json.dumps(value, ensure_ascii=False)

    text = str(value)

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    return text.strip()


def dcg(grades):
    grades = np.asarray(grades, dtype=float)
    discounts = np.log2(np.arange(2, len(grades) + 2))
    return np.sum((2**grades - 1) / discounts)


def ndcg_at_k(ranked_ids, judgments, k):
    observed = [
        judgments.get(docid, 0)
        for docid in ranked_ids[:k]
    ]

    observed += [0] * (k - len(observed))

    ideal = sorted(judgments.values(), reverse=True)[:k]
    ideal += [0] * (k - len(ideal))

    denominator = dcg(ideal)

    return (
        float(dcg(observed) / denominator)
        if denominator else 0.0
    )


qrels = pd.read_csv(
    BASE / "ground_truth_qrels.csv",
    dtype={"query_id": str, "doc_id": str}
)

qrels = qrels[qrels.dataset == "dl19"].copy()

judgments = {
    str(qid): dict(zip(group.doc_id, group.relevance))
    for qid, group in qrels.groupby("query_id")
}


topics_path = RUN / "bm25-aug0_result_retrieval.trec.topics.jsonl"

topics = {}

with topics_path.open(encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        topics[str(record[0])] = str(record[1])


rankings = {}

for r in ROUNDS:
    path = RUN / f"bm25-aug{r}_result_retrieval.trec"

    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=["qid", "Q0", "docid", "rank", "score", "tag"],
        dtype={"qid": str, "docid": str}
    )

    df = df.sort_values(["qid", "rank"])

    for qid, group in df.groupby("qid"):
        rankings[(qid, r)] = group.docid.tolist()


responses = {}

for r in [1, 2, 3]:
    path = RUN / f"bm25-aug{r}_result_retrieval.trec.responses.json"

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    for qid, values in raw.items():
        if not isinstance(values, list):
            values = [values]

        responses[(str(qid), r)] = [
            get_final_answer(value)
            for value in values
        ]


rows = []

for qid in sorted(judgments):
    if qid not in topics:
        continue

    if not all((qid, r) in rankings for r in ROUNDS):
        continue

    previous_top5 = None

    for r in ROUNDS:
        docs = rankings[(qid, r)]
        top5 = docs[:5]
        grades = judgments[qid]

        relevant = {
            doc for doc in top5
            if grades.get(doc, 0) >= 2
        }

        if previous_top5 is None:
            overlap = np.nan
            new_relevant = np.nan
        else:
            overlap = len(set(top5) & set(previous_top5))
            new_relevant = len(
                relevant - set(previous_top5)
            )

        rows.append({
            "query_id": qid,
            "round": r,
            "relevant_at_5": len(relevant),
            "ndcg_at_5": ndcg_at_k(docs, grades, 5),
            "ndcg_at_10": ndcg_at_k(docs, grades, 10),
            "top5_overlap": overlap,
            "new_relevant_at_5": new_relevant,
        })

        previous_top5 = top5


metrics = pd.DataFrame(rows)

metrics.to_csv(
    OUT / "all_query_metrics.csv",
    index=False
)


pivot = metrics.pivot(
    index="query_id",
    columns="round",
    values="ndcg_at_10"
)

delta = pivot[3] - pivot[0]

selected = ["19335"] if "19335" in delta.index else []

candidates = [
    delta.idxmax(),
    delta.idxmin(),
    (delta - delta.median()).abs().idxmin(),
]

early_gain = pivot[1] - pivot[0]
late_change = pivot[3] - pivot[1]

early_then_decline = early_gain - late_change

candidates.append(early_then_decline.idxmax())

for qid in candidates:
    if qid not in selected:
        selected.append(qid)

for qid in delta.abs().sort_values(ascending=False).index:
    if len(selected) >= 5:
        break

    if qid not in selected:
        selected.append(qid)

selected = selected[:5]


export = {
    "dataset": "TREC DL19",
    "model": MODEL,
    "seed": SEED,
    "relevance_threshold": 2,
    "selected_queries": selected,
    "queries": {}
}

retrieved_rows = []

for qid in selected:
    subset = (
        metrics[metrics.query_id == qid]
        .set_index("round")
    )

    query_data = {
        "original_query": topics[qid],
        "delta_ndcg_at_10": round(float(delta[qid]), 6),
        "rounds": {}
    }

    for r in ROUNDS:
        docs = rankings[(qid, r)]
        grades = judgments[qid]

        top5 = []

        for rank, docid in enumerate(docs[:5], start=1):
            grade = grades.get(docid)

            passage = {
                "rank": rank,
                "doc_id": docid,
                "relevance": int(grade) if grade is not None else None,
                "is_relevant": bool(grade >= 2) if grade is not None else False
            }

            top5.append(passage)

            retrieved_rows.append({
                "query_id": qid,
                "round": r,
                "rank": rank,
                "doc_id": docid,
                "relevance": passage["relevance"],
                "is_relevant": passage["is_relevant"]
            })

        row = subset.loc[r]

        round_data = {
            "stage": "BM25" if r == 0 else f"R{r}",
            "metrics": {
                "relevant_at_5": int(row["relevant_at_5"]),
                "ndcg_at_5": round(float(row["ndcg_at_5"]), 6),
                "ndcg_at_10": round(float(row["ndcg_at_10"]), 6),
                "top5_overlap": (
                    int(row["top5_overlap"])
                    if pd.notna(row["top5_overlap"])
                    else None
                ),
                "new_relevant_at_5": (
                    int(row["new_relevant_at_5"])
                    if pd.notna(row["new_relevant_at_5"])
                    else None
                )
            },
            "top5_passages": top5,
            "generated_responses": (
                responses.get((qid, r), [])
                if r > 0 else []
            )
        }

        query_data["rounds"][str(r)] = round_data

    export["queries"][qid] = query_data


json_path = OUT / "five_query_case_studies.json"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(
        export,
        f,
        indent=2,
        ensure_ascii=False
    )


expansions = {
    qid: {
        "original_query": topics[qid],
        "rounds": {
            str(r): responses.get((qid, r), [])
            for r in [1, 2, 3]
        }
    }
    for qid in selected
}

with (OUT / "selected_expansions.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        expansions,
        f,
        indent=2,
        ensure_ascii=False
    )


metrics[
    metrics.query_id.isin(selected)
].to_csv(
    OUT / "selected_query_metrics.csv",
    index=False
)

pd.DataFrame(retrieved_rows).to_csv(
    OUT / "selected_top5_passages.csv",
    index=False
)


print("\nTHINKQE: FIVE QUERY CASE STUDIES")
print(f"Dataset: TREC DL19 | Model: {MODEL} | Seed: {SEED}")

print("\nSelected queries:")

for i, qid in enumerate(selected, start=1):
    print(
        f"{i}. {qid}: {topics[qid]} "
        f"(Delta nDCG@10: {delta[qid]:+.4f})"
    )


summary = (
    metrics[metrics.query_id.isin(selected)]
    .pivot(
        index="query_id",
        columns="round",
        values="ndcg_at_10"
    )
    .reindex(selected)
)

summary.columns = ["BM25", "R1", "R2", "R3"]
summary["Delta"] = summary["R3"] - summary["BM25"]

print("\nnDCG@10 comparison:")
print(
    summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nOutput files:")
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name != "five_query_appendix.tex":
        print(f"  {path.name}")

print(f"\nSaved to: {OUT}")
print("DONE")
