"""Coverage metrics for CoverageBench (arXiv 2603.20034); checked to reproduce the paper's Tables 2-3 exactly
from its released runs (snellius/coverage_eval.py bm25-check).

Input: a TREC run and nugget qrels "qid nugget_id docid rel" (ndeval / diversity format).
Macro-average over the qrels' topics; a topic missing from the run scores 0. Ranking order follows trec_eval
(score descending, ties by docid descending).
  ndcg@k          document-level, binary gain (relevant = supports >= 1 nugget), as in the paper's nDCG@20
  ndcg_graded@20  gain = number of nuggets the document supports
  alpha-ndcg@20   Clarke et al. 2008, alpha=0.5, greedy ideal ranking from the judged docs (as ndeval)
  strecall@20     subtopic recall: fraction of the topic's nuggets supported by >= 1 of the top 20
  recall@100      document-level recall
"""
import math
from collections import defaultdict

ALPHA = 0.5


def load_nugget_qrels(path):
    """qid -> docid -> set(nugget ids), positive judgments only."""
    q = defaultdict(lambda: defaultdict(set))
    for line in open(path):
        p = line.split()
        if len(p) == 4 and int(p[3]) > 0:
            q[p[0]][p[2]].add(p[1])
    return q


def load_run(path):
    """qid -> [docid] in trec_eval order."""
    r = defaultdict(list)
    for line in open(path):
        p = line.split()
        if len(p) >= 6:
            r[p[0]].append((float(p[4]), p[2]))
    return {q: [d for _, d in sorted(v, reverse=True)] for q, v in r.items()}


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(ranked, rel, k, graded=False):
    gain = (lambda d: len(rel.get(d, ()))) if graded else (lambda d: 1.0 if d in rel else 0.0)
    ideal = sorted((len(v) if graded else 1.0 for v in rel.values()), reverse=True)[:k]
    return dcg([gain(d) for d in ranked[:k]]) / dcg(ideal) if ideal else 0.0


def alpha_ndcg(ranked, rel, k):
    def gains(order):
        seen, out = defaultdict(int), []
        for d in order:
            g = 0.0
            for n in rel.get(d, ()):
                g += (1 - ALPHA) ** seen[n]
                seen[n] += 1
            out.append(g)
        return out

    # greedy ideal: repeatedly take the judged document with the largest marginal alpha-gain
    seen, pool, ideal = defaultdict(int), dict(rel), []
    for _ in range(min(k, len(pool))):
        best, bg = None, -1.0
        for d, ns in pool.items():
            g = sum((1 - ALPHA) ** seen[n] for n in ns)
            if g > bg or (g == bg and d > best):
                best, bg = d, g
        ideal.append(bg)
        for n in pool.pop(best):
            seen[n] += 1
    den = dcg(ideal)
    return dcg(gains(ranked[:k])) / den if den > 0 else 0.0


def st_recall(ranked, rel, k):
    allnug = set().union(*rel.values())
    got = set().union(*(rel.get(d, set()) for d in ranked[:k]))
    return len(got) / len(allnug) if allnug else 0.0


def recall(ranked, rel, k):
    return len(set(ranked[:k]) & set(rel)) / len(rel) if rel else 0.0


METRICS = {
    "ndcg@10": lambda r, q: ndcg(r, q, 10),
    "ndcg@20": lambda r, q: ndcg(r, q, 20),
    "ndcg_graded@20": lambda r, q: ndcg(r, q, 20, graded=True),
    "alpha-ndcg@20": lambda r, q: alpha_ndcg(r, q, 20),
    "strecall@20": lambda r, q: st_recall(r, q, 20),
    "recall@100": lambda r, q: recall(r, q, 100),
}


def evaluate(run_path, nugget_qrels_path, per_query=False):
    """Mean of each metric (0-1); with per_query also {qid: {metric: value}}."""
    qrels = load_nugget_qrels(nugget_qrels_path)
    run = load_run(run_path)
    rows = {qid: {name: f(run.get(qid, []), rel) for name, f in METRICS.items()} for qid, rel in qrels.items()}
    out = {name: sum(r[name] for r in rows.values()) / len(rows) for name in METRICS}
    out["n"] = len(qrels)
    out["n_in_run"] = sum(q in run for q in qrels)
    return (out, rows) if per_query else out
