"""Compare your ThinkQE runs with the paper (Tables 3 and 4). From the repo root:
    python snellius/collect_results.py            # all models
    python snellius/collect_results.py 14B        # only runs whose model name contains "14B"
Reads /scratch-shared/$USER/thinkqe/outputs/<dataset>_<model>[__<variant>]/.
Per run: final-round score vs the paper, plus generations without "</think>\\n" (what
extract_answer() splits on; in those the whole reasoning trace becomes the expansion).
Then, per model, mean ± std over all seeds of the batched runs."""
import json, os, glob, re, sys, statistics as st

OUT = f"/scratch-shared/{os.environ['USER']}/thinkqe/outputs"
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
BRIGHT = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]
ORDER = ["dl19", "dl20"] + BRIGHT
PAPER = {  # DL: (mAP, nDCG@10, R@1k); BRIGHT: nDCG@10 per domain
  "DeepSeek-R1-Distill-Qwen-14B": {"dl19": (45.9, 68.8, 89.3), "dl20": (43.9, 64.7, 87.8), "bright": [47.3, 52.5, 29.2, 40.0, 19.3, 28.0, 27.9]},
  "Qwen3-8B":                     {"dl19": (44.5, 65.0, 87.9), "dl20": (41.9, 62.8, 88.0), "bright": [49.8, 55.3, 27.6, 36.7, 19.9, 29.0, 28.3]},
  "Qwen3-14B":                    {"dl19": (45.2, 64.9, 88.4), "dl20": (42.4, 63.5, 88.4), "bright": [51.5, 53.2, 27.8, 37.2, 22.0, 16.1, 27.5]},
  "OpenThinker2-7B":              {"dl19": (44.8, 65.3, 87.3), "dl20": (43.2, 63.5, 87.9), "bright": [50.5, 54.1, 25.8, 36.7, 18.1, 28.2, 28.9]},
  "Phi-4-reasoning":              {"dl19": (44.0, 65.0, 87.1), "dl20": (43.0, 63.9, 87.2), "bright": [51.8, 53.5, 29.7, 38.5, 21.8, 29.3, 27.7]},
}
KEYS = ("map", "ndcg@10", "recall@1000")
# Paper's R1-14B ablations on BRIGHT (nDCG@10 per domain), keyed by the ETAG used for that run.
# Tables 10-13. Suggested ETAGs: nothink1r, par1..par6 (1 round, k samples), noaccum, nofilter, noboth, lam1..lam6
PAPER_ABL = {
  "nothink1r": [39.1, 45.6, 25.0, 30.0, 18.0, 26.5, 24.4],   # Table 10, w/o thinking (1 round)
  "par1": [42.6, 47.3, 25.1, 30.3, 18.1, 24.8, 25.2],        # Table 11, parallel scaling, 1 LLM call
  "par2": [42.6, 50.6, 26.2, 35.8, 18.8, 28.4, 25.1],        # = Table 10 w. thinking = round 1 of main runs
  "par3": [44.2, 50.4, 26.6, 33.6, 18.0, 26.5, 26.5],
  "par4": [42.4, 49.8, 27.7, 35.5, 17.8, 28.0, 27.4],
  "par5": [41.7, 50.7, 26.7, 35.2, 19.3, 27.5, 27.4],
  "par6": [45.3, 50.3, 26.4, 34.5, 19.0, 28.2, 28.0],
  "noaccum": [47.5, 50.7, 27.9, 34.8, 17.7, 26.5, 28.4],     # Table 12, accumulation off, filter on
  "nofilter": [46.4, 51.5, 27.8, 39.5, 17.9, 28.2, 28.0],    # Table 12, accumulation on, filter off
  "lam1": [38.2, 47.5, 25.0, 32.1, 18.4, 25.5, 24.0],        # Table 13
  "lam2": [44.9, 50.1, 28.2, 37.0, 19.3, 27.0, 26.8],
  "lam4": [48.9, 52.3, 28.2, 38.4, 20.2, 28.1, 29.4],
  "lam5": [49.7, 52.5, 29.5, 37.7, 19.6, 28.9, 27.6],
  "lam6": [49.1, 51.8, 29.5, 39.1, 19.4, 29.0, 28.8],
}
CORPUS_CURVE = {1: 32.5, 2: 34.4, 3: 34.9}   # Table 11 corpus-interaction scaling (2, 4, 6 calls), BRIGHT avg

def etag_of(variant):
    return variant.split("_batched")[0].split("_s")[0] if not variant.startswith(("batched", "unbatched")) else ""

def paper(model, ds, variant=""):
    e = etag_of(variant)
    if e:
        if model == "DeepSeek-R1-Distill-Qwen-14B" and e in PAPER_ABL and ds in BRIGHT:
            return PAPER_ABL[e][BRIGHT.index(ds)]
        if e not in ("thinksplit",):
            return None          # ablation without a paper number
    p = PAPER.get(model)
    if not p: return None
    return p[ds] if ds in ("dl19", "dl20") else p["bright"][BRIGHT.index(ds)]

def fmt(v): return "  -  " if v is None else f"{v:5.1f}"
def fmt3(t): return "/".join(fmt(x) for x in t) if t else "-"

runs = {}  # (model, variant) -> {ds: record}
for run in glob.glob(f"{OUT}/*"):
    name = os.path.basename(run)
    ds = next((d for d in ORDER if name.startswith(d + "_")), None)
    if ds is None: continue
    model, _, variant = name[len(ds) + 1:].partition("__")
    variant = variant or "unbatched_s0"
    if FILTER and not re.search(FILTER, model): continue
    files = sorted(glob.glob(f"{run}/bm25-aug*_result_retrieval.trec.metrics.json"),
                   key=lambda f: int(f.split("bm25-aug")[1].split("_")[0]))
    rec = {"round": None, "rounds": 3}
    ea = os.path.join(run, "extra_args.txt")        # written by run_thinkqe.job when a run finishes
    if os.path.exists(ea):
        m_ = re.search(r"--num_interaction\s+(\d+)", open(ea).read())
        if m_: rec["rounds"] = int(m_.group(1))
    if files:
        try:
            m = json.load(open(files[-1]))
        except (json.JSONDecodeError, OSError):   # file being written by a running job
            files = files[:-1]
            if not files:
                runs.setdefault((model, variant), {})[ds] = rec; continue
            m = json.load(open(files[-1]))
        rec["round"] = int(files[-1].split("bm25-aug")[1].split("_")[0])
        rec["vals"] = tuple(m.get(k) for k in KEYS) if ds in ("dl19", "dl20") else (m.get("ndcg@10"),)
        rec["by_round"] = {}
        for f in files:
            try: rec["by_round"][int(f.split("bm25-aug")[1].split("_")[0])] = json.load(open(f)).get("ndcg@10")
            except (json.JSONDecodeError, OSError): pass
        n = noclose = nonl = 0
        for f in glob.glob(f"{run}/*.responses.json"):
            try: data = json.load(open(f))
            except (json.JSONDecodeError, OSError): continue   # being written
            for resps in data.values():
                for r in resps:
                    n += 1
                    if "</think>" not in r: noclose += 1      # never finished thinking
                    elif "</think>\n" not in r: nonl += 1    # closed, no newline: the default split misses it
        rec["noclose"], rec["nonl"], rec["n"] = noclose, nonl, n
    runs.setdefault((model, variant), {})[ds] = rec

if not runs: print(f"No runs found in {OUT}"); sys.exit()
print("DL19/DL20: mAP/nDCG@10/R@1k.  BRIGHT: nDCG@10.  Only finished runs (last round reached) are final.")
for (model, variant) in sorted(runs):
    print(f"\n== {model} [{variant}]")
    recs = runs[(model, variant)]
    for ds in ORDER:
        if ds not in recs: continue
        r = recs[ds]
        if r["round"] is None: print(f"  {ds:19s} no results yet"); continue
        p = paper(model, ds, variant)
        ours = fmt3(r["vals"]) if ds in ("dl19", "dl20") else fmt(r["vals"][0])
        theirs = fmt3(p) if ds in ("dl19", "dl20") else fmt(p)
        note = f"round {r['round']}/{r['rounds']}" + (f", {r['noclose']}/{r['n']} no </think>" if r["noclose"] else "") \
               + (f", {r['nonl']}/{r['n']} </think> without newline" if r["nonl"] else "")
        print(f"  {ds:19s} ours {ours} | paper {theirs} | {note}")
    done = [recs[d]["vals"][0] for d in BRIGHT if d in recs and recs[d]["round"] == recs[d]["rounds"]]
    if len(done) == 7:
        pv = [paper(model, d, variant) for d in BRIGHT]
        pb = st.mean(pv) if all(x is not None for x in pv) else None
        print(f"  {'BRIGHT average':19s} ours {fmt(st.mean(done))} | paper {fmt(pb)}")

# Mean ± std over seeds: runs of the same model and variant that differ only in seed (…_s0, _s1, …)
groups = {}
for (m, v) in runs:
    if re.search(r"_s\d+$", v) and v != "unbatched_s0":
        groups.setdefault((m, re.sub(r"_s\d+$", "", v)), []).append(v)
for (model, base), seeds in sorted(groups.items()):
    if len(seeds) < 2: continue
    print(f"\n== {model} [{base}]: mean ± std over seeds ({', '.join(sorted(seeds))})")
    for ds in ORDER:
        vals = [runs[(model, s)][ds]["vals"] for s in seeds
                if ds in runs[(model, s)] and runs[(model, s)][ds]["round"] == runs[(model, s)][ds]["rounds"]]
        if len(vals) < 2: continue
        cols = list(zip(*vals))
        ms = " / ".join(f"{st.mean(c):4.1f}±{st.stdev(c):4.2f}" for c in cols)
        p = paper(model, ds, base)
        print(f"  {ds:19s} n={len(vals)}  {ms} | paper {fmt3(p) if ds in ('dl19','dl20') else fmt(p)}")
    per_seed = []
    for s in seeds:
        v = [runs[(model, s)][d]["vals"][0] for d in BRIGHT if d in runs[(model, s)] and runs[(model, s)][d]["round"] == runs[(model, s)][d]["rounds"]]
        if len(v) == 7: per_seed.append(st.mean(v))
    if len(per_seed) >= 2:
        pv = [paper(model, d, base) for d in BRIGHT]
        pb = st.mean(pv) if all(x is not None for x in pv) else None
        print(f"  {'BRIGHT average':19s} n={len(per_seed)}  {st.mean(per_seed):5.2f}±{st.stdev(per_seed):4.2f} (per seed: {', '.join(f'{x:.2f}' for x in per_seed)}) | paper {fmt(pb)}")

# Corpus-interaction curve (Figure 1 / Table 11): BRIGHT average after rounds 1, 2, 3 of the main runs
for model in sorted({m for m, _ in runs}):
    seeds = [v for (m, v) in runs if m == model and re.fullmatch(r"batched_s\d+", v)]
    curves = {}
    for s_ in seeds:
        recs = runs[(model, s_)]
        if all(d in recs and recs[d].get("round") == 3 for d in BRIGHT):
            for k in (1, 2, 3):
                curves.setdefault(k, []).append(st.mean(recs[d]["by_round"][k] for d in BRIGHT))
    if len(curves.get(1, [])) >= 1:
        print(f"\n== {model}: BRIGHT average by round (corpus-interaction scaling; 2, 4, 6 LLM calls)")
        for k in (1, 2, 3):
            v = curves[k]; sd = f"±{st.stdev(v):4.2f}" if len(v) > 1 else ""
            pc = f" | paper {CORPUS_CURVE[k]:.1f}" if model == "DeepSeek-R1-Distill-Qwen-14B" else ""
            print(f"  round {k} ({2*k} calls)    n={len(v)}  {st.mean(v):5.2f}{sd}{pc}")
