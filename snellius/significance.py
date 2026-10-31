"""Paired significance tests per query for the ThinkQE reproduction.

Run from the repo root with the conda env's Python (needs numpy + scipy):
    ~/.conda/envs/thinkqe_310/bin/python snellius/significance.py --preset main
    ~/.conda/envs/thinkqe_310/bin/python snellius/significance.py --preset phi
    ~/.conda/envs/thinkqe_310/bin/python snellius/significance.py --preset ablations
    ~/.conda/envs/thinkqe_310/bin/python snellius/significance.py \
        --compare "Qwen3-14B:batched" "DeepSeek-R1-Distill-Qwen-14B:batched" --datasets bright

A system is "MODEL[:VARIANT][@ROUND]":
  MODEL    model folder name, e.g. DeepSeek-R1-Distill-Qwen-14B, Qwen3-14B, Phi-4-reasoning
  VARIANT  run variant without the seed suffix (default "batched"), e.g. thinksplit_batched, nothink1r_batched
  ROUND    0 = BM25, 1..3 = after that round; default = the run's final round
Per-query nDCG@10 is averaged over all seeds of a variant before testing.

Tests: paired randomisation (sign-flip) test, 20,000 permutations, and paired t-test;
Holm-Bonferroni correction over all comparisons printed in one invocation.
nDCG@10 is computed here (trec_eval conventions: gain = relevance label, log2 discount,
ties broken by docid descending, queries missing from a run count as 0) and checked against
the trec_eval score stored for each run; a mismatch is reported.
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np
from scipy import stats

OUT = f"/scratch-shared/{os.environ['USER']}/thinkqe/outputs"
BRIGHT = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]
DL = ["dl19", "dl20"]
QRELS = {"dl19": os.path.expanduser("~/.cache/anserini/topics-and-qrels/qrels.dl19-passage.txt"),
         "dl20": os.path.expanduser("~/.cache/anserini/topics-and-qrels/qrels.dl20-passage.txt")}
QRELS.update({d: f"bright/data/pyserini_qrels/{d}.tsv" for d in BRIGHT})

# ---------- evaluation ----------
_qrels_cache = {}
def load_qrels(ds):
    if ds not in _qrels_cache:
        q = defaultdict(dict)
        for line in open(QRELS[ds]):
            p = line.split()
            if len(p) >= 4:
                q[p[0]][p[2]] = int(p[3])
        _qrels_cache[ds] = q
    return _qrels_cache[ds]

def ndcg10_per_query(run_file, qrels):
    run = defaultdict(list)
    for line in open(run_file):
        p = line.split()
        if len(p) >= 6:
            run[p[0]].append((float(p[4]), p[2]))
    out = {}
    for qid, rels in qrels.items():
        ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:10]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        if idcg == 0:
            continue                      # trec_eval skips queries without relevant docs
        docs = sorted(run.get(qid, []), reverse=True)[:10]   # score desc, then docid desc (trec_eval)
        dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, (_, d) in enumerate(docs) if rels.get(d, 0) > 0)
        out[qid] = dcg / idcg
    return out

# ---------- locating runs ----------
def runs_for(model, variant, ds):
    """All seed folders for model+variant on dataset ds."""
    pat = re.compile(rf"^{re.escape(ds)}_{re.escape(model)}(?:__(.+))?$")
    found = []
    for d in glob.glob(f"{OUT}/{ds}_{model}*"):
        m = pat.match(os.path.basename(d))
        if not m: continue
        v = m.group(1) or "unbatched_s0"
        if re.sub(r"_s\d+$", "", v) == variant:
            found.append(d)
    return sorted(found)

def final_round(run_dir):
    rounds = 3
    ea = os.path.join(run_dir, "extra_args.txt")
    if os.path.exists(ea):
        m = re.search(r"--num_interaction\s+(\d+)", open(ea).read())
        if m: rounds = int(m.group(1))
    return rounds

_warned = set()
def system_scores(spec, ds):
    """Per-query nDCG@10 for a system spec, averaged over its seeds. None if unavailable."""
    m = re.fullmatch(r"([^:@]+)(?::([^@]+))?(?:@(\d+))?", spec)
    if not m: sys.exit(f"Bad system spec: {spec}")
    model, variant, rnd = m.group(1), m.group(2) or "batched", m.group(3)
    qrels = load_qrels(ds)
    per_seed = []
    for rd in runs_for(model, variant, ds):
        r = int(rnd) if rnd is not None else final_round(rd)
        f = f"{rd}/bm25-aug{r}_result_retrieval.trec"
        mf = f + ".metrics.json"
        if not (os.path.exists(f) and os.path.exists(mf)):
            continue                      # round not finished
        pq = ndcg10_per_query(f, qrels)
        stored = json.load(open(mf)).get("ndcg@10")
        mine = 100 * sum(pq.values()) / len(pq)
        if stored is not None and abs(mine - stored) > 0.05 and f not in _warned:
            _warned.add(f)
            print(f"  WARNING: {os.path.basename(rd)} round {r}: recomputed nDCG@10 {mine:.2f} != trec_eval {stored:.2f}")
        per_seed.append(pq)
    if not per_seed:
        return None, 0
    qids = set.intersection(*(set(p) for p in per_seed))
    return {q: float(np.mean([p[q] for p in per_seed])) for q in qids}, len(per_seed)

# ---------- tests ----------
def perm_test(diff, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(diff)))
    return float((np.abs((signs * diff).mean(axis=1)) >= obs - 1e-12).mean())

def compare(a, b, ds_list, label=None):
    rows = []
    pooled_a, pooled_b = [], []
    for ds in ds_list:
        sa, na = system_scores(a, ds)
        sb, nb = system_scores(b, ds)
        if sa is None or sb is None:
            rows.append((ds, None)); continue
        q = sorted(set(sa) & set(sb))
        xa = np.array([sa[k] for k in q]); xb = np.array([sb[k] for k in q])
        if ds in BRIGHT:
            pooled_a += list(xa); pooled_b += list(xb)
        rows.append((ds, (xa, xb, na, nb)))
    if len([r for r in rows if r[0] in BRIGHT and r[1] is not None]) == 7:
        rows.append(("BRIGHT pooled", (np.array(pooled_a), np.array(pooled_b), None, None)))
    results = []
    for ds, data in rows:
        if data is None:
            results.append(dict(label=label or f"{a} vs {b}", ds=ds, missing=True)); continue
        xa, xb, na, nb = data
        d = xa - xb
        p_t = float(stats.ttest_rel(xa, xb).pvalue) if np.any(d != 0) else 1.0
        p_p = perm_test(d) if np.any(d != 0) else 1.0
        results.append(dict(label=label or f"{a} vs {b}", ds=ds, n=len(d), ma=100 * xa.mean(), mb=100 * xb.mean(),
                            diff=100 * d.mean(), win=int((d > 1e-9).sum()), tie=int((np.abs(d) <= 1e-9).sum()),
                            loss=int((d < -1e-9).sum()), p_perm=p_p, p_t=p_t, seeds=(na, nb)))
    return results

def holm(results):
    tested = [r for r in results if not r.get("missing")]
    order = sorted(range(len(tested)), key=lambda i: tested[i]["p_perm"])
    m, running = len(tested), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tested[i]["p_perm"]))
        tested[i]["p_holm"] = running

def print_results(results):
    holm(results)
    cur = None
    for r in results:
        if r["label"] != cur:
            cur = r["label"]; print(f"\n== {cur}")
            print(f"  {'dataset':22s} {'n':>4s} {'A':>6s} {'B':>6s} {'A-B':>6s} {'win/tie/loss':>13s} {'p_perm':>8s} {'p_t':>8s} {'p_holm':>8s}")
        if r.get("missing"):
            print(f"  {r['ds']:22s}  (runs missing or unfinished)"); continue
        star = " *" if r["p_holm"] < 0.05 else ""
        s = r["seeds"]; sd = f"  seeds {s[0]}/{s[1]}" if s[0] else ""
        print(f"  {r['ds']:22s} {r['n']:4d} {r['ma']:6.1f} {r['mb']:6.1f} {r['diff']:+6.1f} "
              f"{r['win']:4d}/{r['tie']:3d}/{r['loss']:<4d} {r['p_perm']:8.4f} {r['p_t']:8.4f} {r['p_holm']:8.4f}{star}{sd}")
    print("\nnDCG@10 x100, per-query scores averaged over seeds. * = significant after Holm correction (p < 0.05)"
          " over all comparisons above.")

# ---------- presets ----------
MODELS = {"DeepSeek-R1-Distill-Qwen-14B": "batched", "Qwen3-8B": "batched", "Qwen3-14B": "batched",
          "OpenThinker2-7B": "batched", "Phi-4-reasoning": "thinksplit_batched"}
R1 = "DeepSeek-R1-Distill-Qwen-14B"

def preset(name):
    """Yields (A, B, datasets, label)."""
    if name == "main":                       # ThinkQE vs BM25 (round 0 of the same runs)
        for m, v in MODELS.items():
            yield f"{m}:{v}", f"{m}:{v}@0", DL + BRIGHT, f"{m} ThinkQE vs BM25"
    elif name == "phi":
        yield "Phi-4-reasoning:thinksplit_batched", "Phi-4-reasoning:batched", DL + BRIGHT, "Phi-4: correct parsing vs as-is"
    elif name == "models":                   # every model vs R1-14B
        for m, v in MODELS.items():
            if m != R1:
                yield f"{m}:{v}", f"{R1}:batched", DL + BRIGHT, f"{m} vs R1-14B"
    elif name == "ablations":                # R1-14B on BRIGHT
        yield f"{R1}:batched@1", f"{R1}:nothink1r_batched", BRIGHT, "Thinking vs no thinking (1 round, 2 samples)"
        yield f"{R1}:batched@1", f"{R1}:par1_batched", BRIGHT, "2 samples vs 1 sample (1 round)"
        yield f"{R1}:batched@2", f"{R1}:par4_batched", BRIGHT, "4 calls: 2 rounds vs 1 round x 4 samples"
        yield f"{R1}:batched@3", f"{R1}:par6_batched", BRIGHT, "6 calls: 3 rounds vs 1 round x 6 samples"
        yield f"{R1}:batched", f"{R1}:noaccum_batched", BRIGHT, "Full vs accumulation off"
        yield f"{R1}:batched", f"{R1}:nofilter_batched", BRIGHT, "Full vs filter off"
        yield f"{R1}:batched", f"{R1}:noboth_batched", BRIGHT, "Full vs both off"
    else:
        sys.exit(f"Unknown preset {name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["main", "phi", "models", "ablations"])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--datasets", default="all", help="all | dl | bright | comma-separated names")
    args = ap.parse_args()
    if not os.path.isdir("bright/data"):
        sys.exit("Run from the repo root (bright/data must exist).")
    sel = {"all": DL + BRIGHT, "dl": DL, "bright": BRIGHT}.get(args.datasets, args.datasets.split(","))
    jobs = list(preset(args.preset)) if args.preset else []
    if args.compare:
        jobs.append((args.compare[0], args.compare[1], sel, None))
    if not jobs:
        ap.error("give --preset or --compare")
    results = []
    for a, b, dss, label in jobs:
        results += compare(a, b, [d for d in dss if d in sel] if args.preset else dss, label)
    print_results(results)
