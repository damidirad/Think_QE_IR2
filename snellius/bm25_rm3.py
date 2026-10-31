"""BM25 and BM25+RM3 baselines (no LLM) on DL19, DL20 and the 7 BRIGHT domains.

Plain BM25 is run too, as a sanity check: it must match ThinkQE's round 0
(DL19 30.1/50.6/75.0, DL20 28.6/48.0/78.6, BRIGHT 18.2/27.9/16.4/13.4/10.9/16.3/16.1 nDCG@10).

Usage (from the repo root, inside the thinkqe_310 env):
    python snellius/bm25_rm3.py --out_dir $WORK/outputs/baselines [--datasets dl19 biology ...]
Writes <out_dir>/<method>/<dataset>.trec and prints a results table.
RM3 uses Anserini's standard settings (fb_terms=10, fb_docs=10, original_query_weight=0.5);
BM25 uses Pyserini's defaults (k1=0.9, b=0.4), as ThinkQE does with --disable_bm25_param.
"""
import argparse
import os
import subprocess
import sys

from pyserini.search import get_topics
from pyserini.search.lucene import LuceneSearcher

BRIGHT = ["biology", "earth_science", "economics", "psychology",
          "robotics", "stackoverflow", "sustainable_living"]
DL = {"dl19": ("dl19-passage", "dl19-passage"), "dl20": ("dl20", "dl20-passage")}


def load_tsv_topics(path):
    topics = {}
    with open(path) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            topics[qid] = text
    return topics


def dataset_config(ds):
    if ds in DL:
        topics_name, qrels = DL[ds]
        topics = {str(k): v["title"] for k, v in get_topics(topics_name).items()}
        return LuceneSearcher.from_prebuilt_index("msmarco-v1-passage"), topics, qrels
    base = "./bright/data"
    return (LuceneSearcher(f"{base}/pyserini_indexes/{ds}"),
            load_tsv_topics(f"{base}/pyserini_queries/{ds}.tsv"),
            f"{base}/pyserini_qrels/{ds}.tsv")


def run(searcher, topics, out_path, tag, k=1000, threads=16):
    qids = list(topics)
    hits = searcher.batch_search([topics[q] for q in qids], qids, k=k, threads=threads)
    with open(out_path, "w") as f:
        for q in qids:
            for rank, h in enumerate(hits[q], 1):
                f.write(f"{q} Q0 {h.docid} {rank} {h.score:.6f} {tag}\n")


def ndcg10(qrels, run_path):
    out = subprocess.run([sys.executable, "-m", "pyserini.eval.trec_eval", "-c", "-m", "ndcg_cut.10",
                          qrels, run_path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("ndcg_cut_10") and "all" in line:
            return 100 * float(line.split()[-1])
    raise RuntimeError(f"trec_eval failed for {run_path}:\n{out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DL) + BRIGHT)
    ap.add_argument("--fb_terms", type=int, default=10)
    ap.add_argument("--fb_docs", type=int, default=10)
    ap.add_argument("--original_query_weight", type=float, default=0.5)
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        searcher, topics, qrels = dataset_config(ds)
        for method in ("bm25", "rm3"):
            os.makedirs(f"{args.out_dir}/{method}", exist_ok=True)
            if method == "rm3":
                searcher.set_rm3(args.fb_terms, args.fb_docs, args.original_query_weight)
            else:
                searcher.unset_rm3()
            path = f"{args.out_dir}/{method}/{ds}.trec"
            run(searcher, topics, path, method)
            results[(method, ds)] = ndcg10(qrels, path)
            print(f"{ds:20s} {method:5s} nDCG@10 = {results[(method, ds)]:.1f}", flush=True)

    print("\n" + "dataset".ljust(20) + "  BM25   +RM3")
    for ds in args.datasets:
        print(f"{ds:20s} {results[('bm25', ds)]:5.1f}  {results[('rm3', ds)]:5.1f}")
    b = [ds for ds in args.datasets if ds in BRIGHT]
    if len(b) == len(BRIGHT):
        avg = lambda m: sum(results[(m, d)] for d in b) / len(b)
        print(f"{'BRIGHT average':20s} {avg('bm25'):5.1f}  {avg('rm3'):5.1f}")


if __name__ == "__main__":
    main()
