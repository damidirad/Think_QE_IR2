"""How relevant are the documents ThinkQE feeds to the LLM in each round, with and without the
redundancy filter?  Tests the hypothesis that the filter pushes later rounds onto weaker documents
(and that accumulation compensates for it).

The run folders do not store which 5 passages the model read, so this script replays thinkqe.py's
selection logic (lines ~520-546 and ~606-615 of the original code) on the saved rankings:
  round 1 reads the top-K of ranking aug0;
  round t>1 reads ranking aug(t-1), skipping documents that are in a blacklist or in the raw top-K of
  ranking aug(t-2) (the latter are added to the blacklist when encountered), until K are collected.
Without the filter, round t simply reads the top-K of ranking aug(t-1).
(The original compares passage texts; we compare docids, which is the same unless two documents
have identical text.)

Usage (repo root, thinkqe_310 env):  python snellius/feedback_relevance.py [--out_dir ...]
"""
import argparse
import os
import random
from collections import defaultdict

DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow",
           "sustainable_living"]
CONFIGS = {  # name -> (run-folder variant, filter on?)
    "full (filter on, accum on)": ("", True),
    "no filter (accum on)": ("nofilter_", False),
    "no accumulation (filter on)": ("noaccum_", True),
    "neither": ("noboth_", False),
}
K = 5


def read_run(path):
    ranked = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, *_ = line.split()
            ranked[qid].append((int(rank), docid))
    return {q: [d for _, d in sorted(v)] for q, v in ranked.items()}


def read_qrels(path):
    rel = defaultdict(set)
    with open(path) as f:
        for line in f:
            qid, _, docid, r = line.split()
            if int(r) > 0:
                rel[qid].add(docid)
    return rel


def shown_docs(rankings, use_filter, rounds=3):
    """rankings: list of {qid: [docids]} for aug0..aug(rounds-1). Returns {round: {qid: [docids]}}."""
    out = {}
    qids = rankings[0].keys()
    seen = {q: set() for q in qids}          # blacklist
    last_top = {q: set(rankings[0][q][:K]) for q in qids}
    last_last = {q: set() for q in qids}
    for t in range(1, rounds + 1):
        prev = rankings[t - 1]
        out[t] = {}
        for q in qids:
            if not use_filter:
                out[t][q] = prev[q][:K]
                continue
            picked = []
            for d in prev[q]:
                if d in seen[q]:
                    continue
                if d in last_last[q]:
                    seen[q].add(d)
                    continue
                picked.append(d)
                if len(picked) >= K:
                    break
            out[t][q] = picked
        if t < rounds:  # history update after retrieval of round t (ranking aug t)
            for q in qids:
                last_last[q] = last_top[q]
                last_top[q] = set(rankings[t][q][:K])
    return out


def sign_flip_p(diffs, n_perm=20000, seed=0):
    rnd = random.Random(seed)
    obs = abs(sum(diffs))
    hits = 0
    for _ in range(n_perm):
        s = sum(d if rnd.random() < 0.5 else -d for d in diffs)
        hits += abs(s) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=f"/scratch-shared/{os.environ.get('USER')}/thinkqe/outputs")
    ap.add_argument("--qrels_dir", default="bright/data/pyserini_qrels")
    ap.add_argument("--model", default="DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # per (config, round): per-query value averaged over seeds; key = (domain, qid)
    prec = defaultdict(lambda: defaultdict(list))   # share of shown docs that are relevant
    anyrel = defaultdict(lambda: defaultdict(list)) # at least one relevant doc shown
    nshown = defaultdict(lambda: defaultdict(list))
    same_r1 = defaultdict(list)                     # is the aug1 ranking identical to the full run's?
    for dom in DOMAINS:
        qrels = read_qrels(os.path.join(args.qrels_dir, f"{dom}.tsv"))
        for s in seeds:
            full_aug1 = None
            for cname, (variant, use_filter) in CONFIGS.items():
                folder = os.path.join(args.out_dir, f"{dom}_{args.model}__{variant}batched_s{s}")
                paths = [os.path.join(folder, f"bm25-aug{r}_result_retrieval.trec") for r in range(3)]
                if not all(os.path.exists(p) for p in paths):
                    print(f"  missing: {folder}")
                    continue
                rankings = [read_run(p) for p in paths]
                if variant == "":
                    full_aug1 = rankings[1]
                elif full_aug1 is not None:
                    same_r1[cname].append(sum(rankings[1][q][:10] == full_aug1[q][:10] for q in rankings[1])
                                          / len(rankings[1]))
                shown = shown_docs(rankings, use_filter)
                for t, per_q in shown.items():
                    for q, docs in per_q.items():
                        r = sum(d in qrels[q] for d in docs)
                        prec[cname, t][(dom, q)].append(r / K)
                        anyrel[cname, t][(dom, q)].append(float(r > 0))
                        nshown[cname, t][(dom, q)].append(len(docs))

    def mean_over(qdict):
        vals = [sum(v) / len(v) for v in qdict.values()]
        return 100 * sum(vals) / len(vals), len(vals)

    print("\nRelevant documents among the K=5 the LLM reads, BRIGHT pooled, mean over seeds per query")
    print(f"{'configuration':32s} {'round':>5s} {'P(shown)':>9s} {'>=1 rel':>8s} {'docs':>5s} {'n':>5s}")
    for cname in CONFIGS:
        for t in (1, 2, 3):
            if (cname, t) not in prec:
                continue
            p, n = mean_over(prec[cname, t])
            a, _ = mean_over(anyrel[cname, t])
            k, _ = mean_over(nshown[cname, t])
            print(f"{cname:32s} {t:5d} {p:8.1f}% {a:7.1f}% {k/100:5.2f} {n:5d}")
        print()

    print("Filter on vs off, paired per query (sign-flip test):")
    for on, off in (("full (filter on, accum on)", "no filter (accum on)"),
                    ("no accumulation (filter on)", "neither")):
        for t in (2, 3):
            a, b = prec[on, t], prec[off, t]
            keys = sorted(set(a) & set(b))
            diffs = [sum(a[k]) / len(a[k]) - sum(b[k]) / len(b[k]) for k in keys]
            if not diffs:
                continue
            print(f"  round {t}: {on}  minus  {off}: {100 * sum(diffs) / len(diffs):+.1f} pp "
                  f"(n={len(diffs)}, p={sign_flip_p(diffs):.4f})")

    for cname, v in same_r1.items():
        print(f"\nShare of queries whose round-1 ranking (top 10) equals the full run's: {cname}: "
              f"{100 * sum(v) / len(v):.1f}%")


if __name__ == "__main__":
    main()
