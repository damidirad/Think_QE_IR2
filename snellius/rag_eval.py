"""RAG answer-quality evaluation of a ThinkQE run on BRIGHT (StackExchange domains).

Original query + top-k documents of the run's final round -> answer by an LLM on the LiteLLM proxy (Qwen3.6)
-> scored against BRIGHT's gold_answer (the accepted StackExchange answer) with Exact Match, token F1,
BERTScore and an LLM judge (gpt-oss-120b).

    python snellius/rag_eval.py run <run_dir> [--k 5] [--round -1] [--limit N] [--stages build,answer,judge,score]
    python snellius/rag_eval.py summary <run_dir> [<run_dir> ...]      # table over domains

Prompts (adapted from published setups):
  short  FlashRAG default generation prompt (RUC-NLPIR/FlashRAG, flashrag/prompt/base_prompt.py): answer only.
         Documents are shown as "Doc i: text" without a title (BRIGHT has none; docid folders name the source
         question, so they would leak relevance).
  long   BRIGHT's own QA prompt (xlang-ai/BRIGHT, generate_answer.py): documents and post in delimited blocks,
         "Let's think step by step to address the query and give a detailed answer."
  judge  BRIGHT's QA judge (xlang-ai/BRIGHT, eval_claude.py): 0-100 in steps of 10, "addresses about N% of the
         reference content", output REASON / SCORE.
EM and F1 follow FlashRAG/SQuAD (normalize_answer: lowercase, no punctuation, no articles, single spaces).
BERTScore F1: roberta-large, layer 17; reported raw and rescaled with the package's baseline (rescaled scores
are negative when texts are less similar than the baseline's random pairs, common for long answers);
inputs longer than 512 tokens are truncated by the model, so long answers/golds are compared on their start.

Baselines: --round 0 answers from plain BM25 (no expansion); --k 0 is closed book (no documents; short = FlashRAG's
no-retrieval prompt "Answer the question based on your own knowledge...", long = BRIGHT's prompt without documents).

Outputs in <run_dir>/rag_eval_<final|rN>_k<k>[_limN]/ (closed book: rag_eval_closedbook/):
  prompts_<style>.jsonl, answers_<style>.jsonl, judge_<style>.jsonl, per_query.tsv, metrics.json
Every API response is cached (see vllm_server/litellm_completion.py), so re-running a stage costs nothing.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import string
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.common import load_jsonl, save_json, save_jsonl  # noqa: E402

BRIGHT = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]
STYLES = ["short", "long"]
MAX_DOC_WORDS = 1000   # BRIGHT chunks are short (median 40-60 words); the cap only guards rare long ones


# ---------------------------------------------------------------- data
def detect_domain(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    for d in BRIGHT:
        if name.startswith(d + "_"):
            return d
    raise SystemExit(f"Not a BRIGHT StackExchange run (no gold answers): {name}")


def clean_gold(text):
    """StackExchange export artefacts out, whitespace normalised; content unchanged."""
    text = text.replace("$\\begingroup$", "").replace("$\\endgroup$", "")
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def load_examples(domain):
    """BRIGHT `examples` split (query + gold_answer) from the local HF cache, no network."""
    import pyarrow as pa
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    paths = glob.glob(f"{hf}/datasets/xlangai___bright/examples/*/*/bright-{domain}.arrow")
    if not paths:
        raise SystemExit(f"BRIGHT examples for {domain} not in {hf} (run snellius/01_download.sh)")
    with pa.memory_map(sorted(paths)[-1]) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    return {r["id"]: {"query": r["query"], "gold": clean_gold(r["gold_answer"])} for r in table.to_pylist()}


def final_round(run_dir):
    rounds = [int(re.search(r"bm25-aug(\d+)_", p).group(1)) for p in glob.glob(f"{run_dir}/bm25-aug*_result_retrieval.trec")]
    if not rounds:
        raise SystemExit(f"No bm25-aug*_result_retrieval.trec in {run_dir}")
    return max(rounds)


def load_ranking(run_dir, rnd, k):
    path = f"{run_dir}/bm25-aug{rnd}_result_retrieval.trec"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise SystemExit(f"Missing or empty ranking: {path}")
    per_q = {}
    with open(path) as fp:
        for line in fp:
            qid, _, docid, rank, _score, _ = line.split()
            per_q.setdefault(qid, []).append((int(rank), docid))
    return {q: [d for _, d in sorted(v)[:k]] for q, v in per_q.items()}


def load_docs(domain, docids):
    docs = {}
    with open(f"{REPO}/bright/data/pyserini_corpus/{domain}/{domain}.jsonl", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            if rec["id"] in docids:
                docs[rec["id"]] = rec["contents"]
    missing = docids - docs.keys()
    if missing:
        raise SystemExit(f"{len(missing)} retrieved docids not in the corpus, e.g. {sorted(missing)[:3]}")
    return docs


def truncate_words(text, n=MAX_DOC_WORDS):
    words = text.split()
    return text.strip() if len(words) <= n else " ".join(words[:n])


# ---------------------------------------------------------------- prompts
def short_messages(query, docs):
    if not docs:   # closed book: FlashRAG's no-retrieval prompt (ConditionalPipeline / AdaptivePipeline)
        system = "Answer the question based on your own knowledge. Only give me the answer and do not output any other words."
        return [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {query}"}]
    reference = "".join(f"Doc {i + 1}: {d}\n" for i, d in enumerate(docs))
    system = ("Answer the question based on the given document. Only give me the answer and do not output any "
              "other words.\nThe following are given documents.\n\n" + reference)
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {query}"}]


def long_messages(query, docs):
    documentation = "\n".join(f"---------- Document {i + 1} start ----------\n{d}\n---------- Document {i + 1} end ----------"
                              for i, d in enumerate(docs))
    user = (f"{documentation}\n---------- post start ----------\n{query.strip()}\n---------- post end ----------\n\n"
            "Let's think step by step to address the query and give a detailed answer.")
    return [{"role": "user", "content": user}]


JUDGE_SYSTEM = "You are a teacher to judge student's answer"
JUDGE_CRITERIA = "\n".join(["0 - The student's answer is completely irrelevant or blank."] +
                           [f"{p} - The student's answer addresses about {p}% of the reference content."
                            for p in range(10, 101, 10)])


def judge_messages(query, answer, gold):
    user = (f"---------- PROBLEM START ----------\n{query.strip()}\n---------- PROBLEM END ----------\n\n"
            f"---------- STUDENT ANSWER START ----------\n{answer.strip()}\n---------- STUDENT ANSWER END ----------\n\n"
            f"---------- REFERENCE ANSWER START ----------\n{gold}\n---------- REFERENCE ANSWER END ----------\n\n"
            f"Criteria:\n{JUDGE_CRITERIA}\n\n"
            "Use the following format to give a score:\n\nREASON:\nDescribe why you give a specific score\n"
            "SCORE:\nThe score you give, e.g., 60\n\nDo not say anything after the score.")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_score(text):
    hits = re.findall(r"SCORE\s*[:：]?\s*\**\s*(\d{1,3}(?:\.\d+)?)", text or "")
    if not hits:
        return None
    return max(0.0, min(100.0, float(hits[-1])))


# ---------------------------------------------------------------- metrics (FlashRAG / SQuAD)
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def bertscore(cands, refs):
    """BERTScore F1 per pair: (raw, rescaled). Rescaled = (raw - b) / (1 - b) with the package's roberta-large
    layer-17 baseline b, as rescale_with_baseline=True does; computed once from the raw scores."""
    import torch
    from bert_score import score
    from huggingface_hub import snapshot_download
    import bert_score
    model = snapshot_download("FacebookAI/roberta-large", local_files_only=True)
    _, _, F = score(cands, refs, model_type=model, num_layers=17, lang="en", rescale_with_baseline=False,
                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(os.path.dirname(bert_score.__file__), "rescale_baseline/en/roberta-large.tsv")) as fp:
        header = fp.readline().strip().split(",")
        row = next(r for r in (line.strip().split(",") for line in fp) if int(r[0]) == 17)
    b = float(row[header.index("F")])
    raw = F.tolist()
    return raw, [(x - b) / (1 - b) for x in raw]


# ---------------------------------------------------------------- stages
def stage_build(args, out, domain):
    examples = load_examples(domain)
    if args.k == 0:   # closed book: no retrieval, the run dir only names the domain
        rnd, ranking = None, {q: [] for q in examples}
    else:
        rnd = final_round(args.run_dir) if args.round < 0 else args.round
        ranking = load_ranking(args.run_dir, rnd, args.k)
    qids = [q for q in examples if q in ranking]
    if len(qids) != len(examples):
        print(f"WARNING: {len(examples) - len(qids)} queries have no ranking in round {rnd}")
    if args.limit:
        qids = qids[:args.limit]
    docs = load_docs(domain, {d for q in qids for d in ranking[q]})
    for style, build in (("short", short_messages), ("long", long_messages)):
        rows = []
        for q in qids:
            texts = [truncate_words(docs[d]) for d in ranking[q]]
            rows.append({"query_id": q, "round": rnd, "docids": ranking[q], "query": examples[q]["query"],
                         "gold": examples[q]["gold"], "messages": build(examples[q]["query"], texts)})
        save_jsonl(rows, f"{out}/prompts_{style}.jsonl")
    print(f"build: {domain}, " + ("closed book (no documents)" if args.k == 0 else f"round {rnd}, top-{args.k}") +
          f", {len(qids)} queries -> prompts_{{short,long}}.jsonl")


def make_api(model, args, seed):
    from vllm_server.litellm_completion import LiteLLMCompletion
    api = LiteLLMCompletion(model_name=f"litellm/{model}", api_base=args.api_base, seed=seed)
    api.verbose = False
    return api


def stage_answer(args, out):
    api = make_api(args.answer_model, args, args.seed)
    for style in args.styles:
        prompts = load_jsonl(f"{out}/prompts_{style}.jsonl")
        api.completion_chat_batch([p["messages"] for p in prompts], n=1, temperature=args.temperature,
                                  top_p=args.top_p, max_tokens=args.max_tokens)
        rows = []
        for p, meta in zip(prompts, api.last_meta):
            answer = meta["content"][0].strip()
            rows.append({"query_id": p["query_id"], "answer": answer, "reasoning": meta["reasoning"][0],
                         "finish_reason": meta["finish_reasons"][0], "prompt_tokens": meta["prompt_tokens"],
                         "completion_tokens": meta["completion_tokens"][0],
                         "truncated": meta["finish_reasons"][0] == "length", "empty": not answer})
        save_jsonl(rows, f"{out}/answers_{style}.jsonl")
        print(f"answer[{style}]: {len(rows)} answers, {sum(r['empty'] for r in rows)} empty, "
              f"{sum(r['truncated'] for r in rows)} cut off at max_tokens")


def stage_judge(args, out):
    api = make_api(args.judge_model, args, 0)
    for style in args.styles:
        prompts = {p["query_id"]: p for p in load_jsonl(f"{out}/prompts_{style}.jsonl")}
        answers = load_jsonl(f"{out}/answers_{style}.jsonl")
        todo = [a for a in answers if not a["empty"]]   # blank answers score 0 by the rubric, no call needed
        api.completion_chat_batch([judge_messages(prompts[a["query_id"]]["query"], a["answer"],
                                                  prompts[a["query_id"]]["gold"]) for a in todo],
                                  n=1, temperature=0, top_p=1, max_tokens=args.max_tokens)
        judged = {a["query_id"]: meta for a, meta in zip(todo, api.last_meta)}
        rows = []
        for a in answers:
            meta = judged.get(a["query_id"])
            text = meta["content"][0] if meta else ""
            rows.append({"query_id": a["query_id"], "score": parse_score(text) if meta else 0.0,
                         "judge_text": text, "finish_reason": meta["finish_reasons"][0] if meta else "blank-answer"})
        save_jsonl(rows, f"{out}/judge_{style}.jsonl")
        print(f"judge[{style}]: {len(rows)} judged, {sum(r['score'] is None for r in rows)} unparsable")


def stage_score(args, out, domain):
    metrics = {"dataset": domain, "run_dir": os.path.abspath(args.run_dir), "k": args.k,
               "answer_model": args.answer_model, "judge_model": args.judge_model, "styles": {}}
    tsv = ["query_id\tstyle\tem\tf1\tbertscore_f1\tbertscore_f1_raw\tjudge\tanswer_words\tempty\ttruncated"]
    for style in args.styles:
        prompts = {p["query_id"]: p for p in load_jsonl(f"{out}/prompts_{style}.jsonl")}
        answers = load_jsonl(f"{out}/answers_{style}.jsonl")
        judges = {j["query_id"]: j["score"] for j in load_jsonl(f"{out}/judge_{style}.jsonl")} \
            if os.path.exists(f"{out}/judge_{style}.jsonl") else {}
        preds = [a["answer"] for a in answers]
        golds = [prompts[a["query_id"]]["gold"] for a in answers]
        em = [exact_match(p, g) for p, g in zip(preds, golds)]
        f1 = [token_f1(p, g) for p, g in zip(preds, golds)]
        braw, bf = bertscore(preds, golds)
        js = [judges.get(a["query_id"]) for a in answers]
        valid_js = [j for j in js if j is not None]
        metrics["round"] = next(iter(prompts.values()))["round"]
        metrics["styles"][style] = {
            "n": len(answers), "exact_match": 100 * st.mean(em), "f1": 100 * st.mean(f1),
            "bertscore_f1": 100 * st.mean(bf), "bertscore_f1_raw": 100 * st.mean(braw),
            "judge": st.mean(valid_js) if valid_js else None, "judge_unparsable": len(js) - len(valid_js),
            "empty_answers": sum(a["empty"] for a in answers), "truncated_answers": sum(a["truncated"] for a in answers),
            "mean_answer_words": st.mean(len(p.split()) for p in preds),
            "mean_gold_words": st.mean(len(g.split()) for g in golds)}
        for a, e, f, b, br, j in zip(answers, em, f1, bf, braw, js):
            tsv.append(f"{a['query_id']}\t{style}\t{e:.0f}\t{f:.4f}\t{b:.4f}\t{br:.4f}\t{'' if j is None else j:}\t"
                       f"{len(a['answer'].split())}\t{int(a['empty'])}\t{int(a['truncated'])}")
    save_json(metrics, f"{out}/metrics.json")
    with open(f"{out}/per_query.tsv", "w") as fp:
        fp.write("\n".join(tsv) + "\n")
    print(json.dumps(metrics["styles"], indent=1))


def summary(run_dirs, sub):
    rows = {}
    for rd in run_dirs:
        for path in sorted(glob.glob(f"{rd}/{sub}/metrics.json")):
            m = json.load(open(path))
            rows[m["dataset"]] = m
    if not rows:
        raise SystemExit("No metrics.json found")
    keys = [("exact_match", "EM"), ("f1", "F1"), ("bertscore_f1", "BERTSc"), ("bertscore_f1_raw", "BERTSc-raw"),
            ("judge", "Judge")]
    for style in STYLES:
        print(f"\n== {style} answers (EM/F1/BERTScore x100, judge 0-100)")
        print(f"{'domain':20s} {'n':>4s} " + " ".join(f"{h:>9s}" for _, h in keys))
        for d in BRIGHT:
            if d in rows and style in rows[d]["styles"]:
                s = rows[d]["styles"][style]
                print(f"{d:20s} {s['n']:4d} " + " ".join(f"{s[k]:9.1f}" if s[k] is not None else f"{'-':>9s}" for k, _ in keys))
        vals = [rows[d]["styles"][style] for d in BRIGHT if d in rows and style in rows[d]["styles"]]
        if vals:
            print(f"{'macro-average':20s} {sum(v['n'] for v in vals):4d} " +
                  " ".join(f"{st.mean(v[k] for v in vals if v[k] is not None):9.1f}" for k, _ in keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("run_dir")
    r.add_argument("--stages", default="build,answer,judge,score")
    r.add_argument("--styles", default="short,long")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--round", type=int, default=-1, help="ranking round to answer from (-1 = final round)")
    r.add_argument("--limit", type=int, default=0, help="first N queries only (pilot)")
    r.add_argument("--answer_model", default="Qwen3.6")
    r.add_argument("--judge_model", default="gpt-oss-120b")
    r.add_argument("--api_base", default="https://llmproxy.uva.nl/v1")
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--top_p", type=float, default=0.95)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max_tokens", type=int, default=32768)
    s = sp.add_parser("summary")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--sub", default="rag_eval_final_k5", help="rag_eval output folder name inside each run dir")
    args = ap.parse_args()

    if args.cmd == "summary":
        return summary(args.run_dirs, args.sub)
    args.styles = [x for x in args.styles.split(",") if x]
    domain = detect_domain(args.run_dir)
    tag = "closedbook" if args.k == 0 else ("final" if args.round < 0 else f"r{args.round}") + f"_k{args.k}"
    out = os.path.join(args.run_dir, f"rag_eval_{tag}" + (f"_lim{args.limit}" if args.limit else ""))
    os.makedirs(out, exist_ok=True)
    stages = args.stages.split(",")
    if "build" in stages:
        stage_build(args, out, domain)
    if "answer" in stages:
        stage_answer(args, out)
    if "judge" in stages:
        stage_judge(args, out)
    if "score" in stages:
        stage_score(args, out, domain)
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
