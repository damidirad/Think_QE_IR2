"""Validate the evaluation logs of a ThinkQE run folder (written with --log_trajectory true).

Rebuilds every logged step independently from the run's own outputs and the index:
  - BM25 query per round  == <round>.topics.jsonl, and == recomputed from the logged expansions
  - top_docids/top_scores == the .trec ranking
  - responses             == <round>.responses.json; expansions / think_split_found recomputed
  - llm_input_docids and skipped_by_filter == a replay of thinkqe.py's redundancy filter (on passage texts)
  - prompt                contains the user prompt rebuilt from those documents (same truncation)
  - finish_reasons / token counts present and consistent; top-psgs.json consistent with --top_passages_*
  - run_config.json complete; optionally, a Slurm log written with --quiet_log true contains no per-query dumps

Usage (repo root; `source snellius/common.sh` first: pyserini needs the Java module):
    python snellius/check_trajectory.py <run_dir> [--slurm_log thinkqe_....out] [--max_queries N]
Exit code 0 = all checks passed.
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_prompt

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--slurm_log", default=None)
ap.add_argument("--max_queries", type=int, default=0, help="check only the first N queries (0 = all)")
a = ap.parse_args()
D = a.run_dir
errors = defaultdict(list)
def err(kind, msg):
    errors[kind].append(msg)

# ---------- run_config ----------
cfg = json.load(open(f"{D}/run_config.json"))
args = cfg["args"]
for k in ("git_commit", "slurm_job_id"):
    if not cfg.get(k): err("run_config", f"{k} missing")
for k, v in cfg["versions"].items():
    if not v and k in ("vllm", "torch", "transformers", "pyserini"): err("run_config", f"version of {k} missing")
R = args["num_interaction"] + 1
K = args["keep_passage_num"]
TOPK = args["trajectory_top_k"]
THINK = args["think_split"]

# ---------- trajectory ----------
traj = [json.loads(l) for l in open(f"{D}/trajectory.jsonl")]
if a.max_queries: traj = traj[:a.max_queries]
print(f"{len(traj)} queries, {R} rounds expected; model={args['generation_model']} batch={args['batch_generation']} "
      f"filter={args['use_passage_filter']} accumulate={args['accumulate']}")

def load_trec(p):
    run = defaultdict(list)
    for l in open(p):
        q, _, d, _, s, _ = l.split()
        run[q].append((d, float(s)))
    return run
trec = [load_trec(f"{D}/bm25-aug{r}_result_retrieval.trec") for r in range(R)]
topics = [dict(tuple(json.loads(l)) for l in open(f"{D}/bm25-aug{r}_result_retrieval.trec.topics.jsonl")) for r in range(R)]
resp = [None] + [json.load(open(f"{D}/bm25-aug{r}_result_retrieval.trec.responses.json")) for r in range(1, R)]

# ---------- passage texts (as thinkqe.py builds them) and truncation ----------
from pyserini.search.lucene import LuceneSearcher
_tidx = args.get("text_index") or args["index"]  # --text_index: texts come from there
searcher = LuceneSearcher(_tidx) if os.path.exists(_tidx) else LuceneSearcher.from_prebuilt_index(_tidx)
_text_cache = {}
def text_of(docid):
    if docid not in _text_cache:
        raw = json.loads(searcher.doc(docid).raw())
        _text_cache[docid] = "\t".join(raw[k] for k in args["answer_key"].split("|") if raw[k])
    return _text_cache[docid]
import spacy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
def truncate(t, n):
    # the original method; over spaCy's 1M-character limit (BrowseComp-Plus) the tokenizer on the full text (the
    # pipeline does not change tokens), so --fast_truncate's prefix cut is still checked independently
    if len(t) <= nlp.max_length:
        return nlp(t)[:n].text[:(n * 10)]
    return nlp.tokenizer(t)[:n].text[:(n * 10)]

stats = defaultdict(int)
for t in traj:
    q, query, rounds = t["query_id"], t["query"], t["rounds"]
    qs = str(q)   # .trec / responses.json keys are strings; query_id may be an int (DL19/DL20)
    if len(rounds) != R: err("rounds", f"{q}: {len(rounds)} rounds"); continue
    seen, last_k, last_last_k = set(), set(), set()   # filter state, replayed on texts
    acc = []
    for r, rec in enumerate(rounds):
        if rec.get("round") != r: err("rounds", f"{q} r{r}: round index {rec.get('round')}")
        # retrieval
        if rec["bm25_query"] != topics[r][q]: err("bm25_query", f"{q} r{r} != topics.jsonl")
        if qs not in trec[r]: err("top_docids", f"{q} r{r}: query missing from .trec"); continue
        if rec["top_docids"] != [d for d, _ in trec[r][qs][:TOPK]]: err("top_docids", f"{q} r{r} != .trec")
        if any(abs(x - y) > 1e-3 for x, (_, y) in zip(rec["top_scores"], trec[r][qs])): err("top_scores", f"{q} r{r}")
        if r > 0:
            # replay the filter on the previous round's ranking
            prev = rounds[r - 1]["top_docids"]
            exp_in, exp_skip = [], []
            if args["use_passage_filter"]:
                for i, d in enumerate(prev):
                    p = text_of(d)
                    if p in seen:
                        exp_skip.append((d, "blacklisted")); continue
                    if p in last_last_k:
                        seen.add(p); exp_skip.append((d, "top_k_two_rounds_ago (added to blacklist)")); continue
                    exp_in.append(d)
                    if len(exp_in) >= K: break
                if len(exp_in) < K and len(prev) == TOPK:
                    err("depth", f"{q} r{r}: filter needed more than trajectory_top_k={TOPK} docs to replay")
            else:
                exp_in = prev[:K]
            if rec["llm_input_docids"] != exp_in: err("llm_input_docids", f"{q} r{r}: {rec['llm_input_docids']} != replay {exp_in}")
            got_skip = [(s["docid"], s["reason"]) for s in rec["skipped_by_filter"]]
            if got_skip != exp_skip: err("skipped_by_filter", f"{q} r{r}: {got_skip} != replay {exp_skip}")
            stats["skipped"] += len(got_skip); stats["max_rank_read"] = max(stats["max_rank_read"],
                max([prev.index(d) + 1 for d in rec["llm_input_docids"]] or [0]))
            # prompt
            psgs = [text_of(d) for d in rec["llm_input_docids"]]
            if args["max_demo_len"]: psgs = [truncate(p, args["max_demo_len"]) for p in psgs]
            user = get_prompt(args["expansion_method"], query, "\n".join(f"{i+1}. {p}" for i, p in enumerate(psgs)))
            if user not in rec["prompt"]: err("prompt", f"{q} r{r}: rebuilt user prompt not found in logged prompt")
            # generations
            if rec["responses"] != resp[r][qs]: err("responses", f"{q} r{r} != responses.json")
            n = args["gen_num"]
            for k in ("responses", "finish_reasons", "completion_tokens", "think_split_found", "expansions"):
                if len(rec[k]) != n: err("generation", f"{q} r{r}: {k} has {len(rec[k])} entries, expected {n}")
            if rec["expansions"] != [x.split(THINK)[-1] for x in rec["responses"]]: err("expansions", f"{q} r{r}")
            if rec["think_split_found"] != [THINK in x for x in rec["responses"]]: err("think_split_found", f"{q} r{r}")
            if not isinstance(rec["prompt_tokens"], int) or rec["prompt_tokens"] <= 0: err("prompt_tokens", f"{q} r{r}")
            for fr, nt in zip(rec["finish_reasons"], rec["completion_tokens"]):
                if fr not in ("stop", "length"): err("finish_reasons", f"{q} r{r}: {fr}")
                if not isinstance(nt, int) or nt <= 0: err("completion_tokens", f"{q} r{r}: {nt}")
                if rec["prompt_tokens"] + nt > 32768: err("completion_tokens", f"{q} r{r}: prompt+completion > 32768")
                stats[f"finish_{fr}"] += 1; stats["completion_tokens"] += nt; stats["generations"] += 1
            stats["no_think_split"] += sum(not x for x in rec["think_split_found"])
            # BM25 query recomputed from the logged expansions (finalize_query_rewrite)
            acc = acc + rec["expansions"] if args["accumulate"] else rec["expansions"]
            rep = max(int(len("\n".join(acc).split()) / (len(query.split()) * args["reqeat_weight"])), 1)
            if rec["bm25_query"] != "\n".join([query] * rep + acc).lower(): err("bm25_query", f"{q} r{r}: not rebuilt from expansions")
        elif set(rec) != {"round", "bm25_query", "top_docids", "top_scores"}:
            err("rounds", f"{q} r0 keys {sorted(rec)}")
        # advance filter state exactly like thinkqe.py does after each search
        if r > 0: last_last_k = last_k
        last_k = {text_of(d) for d in rec["top_docids"][:K]}

# ---------- top-psgs.json ----------
if args["write_top_passages"]:
    for r in range(R):
        tp = json.load(open(f"{D}/bm25-aug{r}_result_retrieval.trectop-psgs.json"))
        kk = args["top_passages_k"] or None
        for t in traj:
            q = str(t["query_id"]); exp_ids = [d for d, _ in trec[r][q]][:kk]
            if q not in tp or not exp_ids: err("top-psgs", f"r{r} {q}: missing"); continue
            if args["top_passages_format"] == "docids":
                if tp[q] != exp_ids: err("top-psgs", f"r{r} {q}: docids != .trec")
            elif len(tp[q]) != len(exp_ids) or (tp[q] and tp[q][0] != text_of(exp_ids[0])):
                err("top-psgs", f"r{r} {q}: texts do not match .trec")

# ---------- Slurm log ----------
if a.slurm_log:
    log = open(a.slurm_log, errors="replace").read()
    if args["quiet_log"]:
        for s in ("Input message:", "INPUT TO the model", "R1 trace", "Already in blacklist", "Adding to blacklist", "|||"):
            if s in log: err("slurm_log", f"quiet_log but log contains {s!r}")
        for r in range(1, R):
            if f"Round {r}: " not in log: err("slurm_log", f"round summary {r} missing")
    print(f"Slurm log: {os.path.getsize(a.slurm_log) / 1e6:.2f} MB")

g = max(stats["generations"], 1)
print(f"generations {stats['generations']}: finish stop={stats['finish_stop']} length={stats['finish_length']}, "
      f"without think split={stats['no_think_split']}, mean completion tokens={stats['completion_tokens'] / g:.0f}; "
      f"filter skipped {stats['skipped']} docs, deepest rank read {stats['max_rank_read']}")
print(f"trajectory.jsonl: {os.path.getsize(f'{D}/trajectory.jsonl') / 1e6:.1f} MB")
if errors:
    for k, v in errors.items():
        print(f"FAIL {k}: {len(v)}  e.g. {v[:3]}")
    sys.exit(1)
print("ALL CHECKS PASSED")
