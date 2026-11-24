"""Coverage evaluation for CoverageBench CRUX (MultiNews, DUC04); data from snellius/prepare_crux.sh.
Metrics: utils/coverage_metrics.py (nDCG@10/20, alpha-nDCG@20, subtopic recall@20, R@100).

Usage (repo root):
  python snellius/coverage_eval.py eval <run> [<run> ...] --dataset crux_multinews [--per_query out.tsv]
  python snellius/coverage_eval.py rounds <thinkqe_output_dir> [...]   # per-round table (dataset from the folder name)
  python snellius/coverage_eval.py bm25-check      # BM25 (pyserini defaults, depth 100) + the paper's runs vs Tables 2-3
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.coverage_metrics import METRICS, evaluate  # noqa: E402

D = "coveragebench/data"
DATASETS = ["crux_multinews", "crux_duc04"]
# CoverageBench paper (arXiv 2603.20034), Tables 2-3: (nDCG@20, alpha-nDCG@20, StRecall@20); None = not reported
PAPER = {
    "crux_multinews": {"initial_bm25": (0.429, 0.469, 0.634), "initial_qwen3": (0.610, 0.648, 0.813),
                       "rank1_bm25": (0.533, None, None), "rerank_bm25_qwen3": (0.494, None, None),
                       "rank1_qwen3": (0.616, None, None), "rerank_qwen3_qwen3": (0.620, None, None)},
    "crux_duc04": {"initial_bm25": (0.448, 0.476, 0.659), "initial_qwen3": (0.701, 0.652, 0.814),
                   "rank1_bm25": (0.610, None, None), "rerank_bm25_qwen3": (0.524, None, None),
                   "rank1_qwen3": (0.754, None, None), "rerank_qwen3_qwen3": (0.713, None, None)},
}
SHORT = {"ndcg@10": "nDCG@10", "ndcg@20": "nDCG@20", "ndcg_graded@20": "nDCG@20-g", "alpha-ndcg@20": "a-nDCG@20",
         "strecall@20": "StRec@20", "recall@100": "R@100"}


def qrels_path(ds):
    return f"{D}/nugget_qrels/{ds}.qrels"


def header(label="run"):
    return f"{label:30s} " + " ".join(f"{SHORT[n]:>9s}" for n in METRICS) + "  topics"


def fmt_row(label, m, paper=None):
    s = f"{label:30s} " + " ".join(f"{m[n]:9.3f}" for n in METRICS) + f"  {m['n_in_run']:3d}/{m['n']}"
    if paper:
        s += "   paper nDCG@20/a-nDCG@20/StRec@20 " + "/".join("  -  " if x is None else f"{x:.3f}" for x in paper)
    return s


def baselines(ds):
    """The paper's runs on this dataset (evaluated here), for reference rows."""
    for name, paper in PAPER[ds].items():
        p = f"{D}/paper_runs/{ds}.{name}.run"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            yield name, evaluate(p, qrels_path(ds)), paper


def bm25_search(dataset, out_path, index=f"{D}/pyserini_indexes/crux", hits=100, threads=16):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(index)
    searcher.set_bm25(0.9, 0.4)  # pyserini defaults (the paper: "BM25 (Pyserini, default parameters)")
    topics = [l.rstrip("\n").split("\t", 1) for l in open(f"{D}/pyserini_queries/{dataset}.tsv") if l.strip()]
    qids, texts = [t[0] for t in topics], [t[1] for t in topics]
    res = searcher.batch_search(texts, qids, k=hits, threads=threads)
    missing = [q for q in qids if q not in res]
    if missing:
        sys.exit(f"batch_search dropped {len(missing)} queries: {missing[:5]}")
    with open(out_path, "w") as f:
        for q in qids:
            for i, h in enumerate(res[q]):
                f.write(f"{q} Q0 {h.docid} {i + 1} {h.score:.6f} bm25\n")
    print(f"wrote {out_path} ({len(qids)} queries, {hits} hits)")


def rounds(out_dir):
    ds = next((d for d in DATASETS if os.path.basename(out_dir.rstrip("/")).startswith(d + "_")), None)
    if ds is None:
        sys.exit(f"cannot tell the dataset from {out_dir} (expected a folder name starting with {DATASETS})")
    runs = sorted(glob.glob(f"{out_dir}/bm25-aug*_result_retrieval.trec"),
                  key=lambda p: int(re.search(r"bm25-aug(\d+)_", p).group(1)))
    lines = [f"Coverage metrics per round ({ds}; nugget qrels {qrels_path(ds)})", header("round")]
    for p in runs:
        r = re.search(r"bm25-aug(\d+)_", p).group(1)
        lines.append(fmt_row(f"round {r}" + (" (BM25)" if r == "0" else ""), evaluate(p, qrels_path(ds))))
    lines.append("reference (the paper's runs, same evaluation):")
    lines += [fmt_row(f"  {n}", m) for n, m, _ in baselines(ds)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="+")
    e.add_argument("--dataset", required=True, choices=DATASETS)
    e.add_argument("--per_query", help="write a per-query TSV here (first run only)")
    r = sub.add_parser("rounds")
    r.add_argument("out_dirs", nargs="+")
    r.add_argument("--write", action="store_true", help="also write <out_dir>/coverage_per_round.txt")
    b = sub.add_parser("bm25-check")
    b.add_argument("--out_dir", default=f"{D}/runs")
    a = ap.parse_args()

    if a.cmd == "eval":
        print(header())
        for i, run in enumerate(a.runs):
            if i == 0 and a.per_query:
                m, rows = evaluate(run, qrels_path(a.dataset), per_query=True)
                with open(a.per_query, "w") as f:
                    f.write("qid\t" + "\t".join(METRICS) + "\n")
                    for q, v in rows.items():
                        f.write(q + "\t" + "\t".join(f"{v[n]:.4f}" for n in METRICS) + "\n")
            else:
                m = evaluate(run, qrels_path(a.dataset))
            print(fmt_row(os.path.basename(run)[:30], m))
    elif a.cmd == "rounds":
        for d in a.out_dirs:
            text = rounds(d)
            print(f"== {d}\n{text}\n")
            if a.write:
                open(os.path.join(d, "coverage_per_round.txt"), "w").write(text + "\n")
    else:
        os.makedirs(a.out_dir, exist_ok=True)
        for ds in DATASETS:
            ours = f"{a.out_dir}/{ds}.bm25_ours.run"
            bm25_search(ds, ours)
            print(f"\n== {ds}")
            print(header())
            print(fmt_row("BM25 (ours, k1=0.9 b=0.4)", evaluate(ours, qrels_path(ds)), PAPER[ds]["initial_bm25"]))
            for name, m, paper in baselines(ds):
                print(fmt_row(f"paper run: {name}", m, paper))


if __name__ == "__main__":
    main()
