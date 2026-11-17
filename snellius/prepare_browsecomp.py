"""Convert BrowseComp-Plus (Tevatron/browsecomp-plus + Tevatron/browsecomp-plus-corpus) to the Pyserini
format thinkqe.py reads (same layout as bright/data/). Run on a login node, thinkqe_310 env:
    python snellius/prepare_browsecomp.py
Inputs (downloaded to scratch):
    /scratch-shared/$USER/browsecomp-plus/decrypted.jsonl      (queries, decrypted with the dataset card's script)
    /scratch-shared/$USER/browsecomp-plus-corpus/data/*.parquet
Outputs in /scratch-shared/$USER/browsecomp-plus/pyserini/ :
    corpus/corpus.jsonl     {"id", "contents"} per document (100,195)
    queries.tsv             qid \\t query  (whitespace, incl. newlines, collapsed to single spaces)
    qrels_evidence.tsv      qid 0 docid 1 for every evidence document (primary qrels in BrowseComp-Plus)
    qrels_gold.tsv          qid 0 docid 1 for every gold document (evidence docs that contain the answer)
    answers.jsonl           {"query_id", "answer"} for the later answer evaluation

PLAIN TEXT of a benchmark meant to stay off the web: keep these files on Snellius scratch,
never commit or upload them (the repo's .gitignore does not cover scratch paths, it does not need to).
"""
import glob, json, os
import pyarrow.parquet as pq

BASE = f"/scratch-shared/{os.environ['USER']}"
OUT = f"{BASE}/browsecomp-plus/pyserini"
os.makedirs(f"{OUT}/corpus", exist_ok=True)

n_docs, ids = 0, set()
with open(f"{OUT}/corpus/corpus.jsonl", "w", encoding="utf-8") as f:
    for p in sorted(glob.glob(f"{BASE}/browsecomp-plus-corpus/data/*.parquet")):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1000, columns=["docid", "text"]):
            for d, t in zip(batch.column("docid").to_pylist(), batch.column("text").to_pylist()):
                assert d not in ids, f"duplicate docid {d}"
                ids.add(d); n_docs += 1
                f.write(json.dumps({"id": d, "contents": t}, ensure_ascii=False) + "\n")

n_q = n_ev = n_gold = 0
missing = 0
with open(f"{BASE}/browsecomp-plus/decrypted.jsonl", encoding="utf-8") as fin, \
     open(f"{OUT}/queries.tsv", "w", encoding="utf-8") as fq, \
     open(f"{OUT}/qrels_evidence.tsv", "w") as fe, open(f"{OUT}/qrels_gold.tsv", "w") as fg, \
     open(f"{OUT}/answers.jsonl", "w", encoding="utf-8") as fa:
    for line in fin:
        r = json.loads(line)
        q = r["query_id"]
        fq.write(f"{q}\t{' '.join(r['query'].split())}\n")
        fa.write(json.dumps({"query_id": q, "answer": r["answer"]}, ensure_ascii=False) + "\n")
        for d in sorted({x["docid"] for x in r["evidence_docs"]}):
            fe.write(f"{q}\t0\t{d}\t1\n"); n_ev += 1; missing += d not in ids
        for d in sorted({x["docid"] for x in r["gold_docs"]}):
            fg.write(f"{q}\t0\t{d}\t1\n"); n_gold += 1; missing += d not in ids
        n_q += 1
for p in (f"{OUT}/queries.tsv", f"{OUT}/qrels_evidence.tsv", f"{OUT}/qrels_gold.tsv", f"{OUT}/answers.jsonl"):
    os.chmod(p, 0o600)
print(f"{n_docs} docs, {n_q} queries, {n_ev} evidence and {n_gold} gold judgements, {missing} judged docids missing from corpus")
print(f"written to {OUT}")
