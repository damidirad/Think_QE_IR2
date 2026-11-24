#!/bin/bash
# CoverageBench CRUX (MultiNews + DUC04): download corpus, topics, qrels, nuggets and the paper's runs into
# coveragebench/data/ (BRIGHT-like layout). Login node, from the repo root:  bash snellius/prepare_crux.sh
# Then build the index:  sbatch snellius/index_crux.job
#   corpus  DylanJHJ/crux-mds-corpus (565,015 passages: MultiNews test + train docs, DUC04 docs; already {id, contents})
#   labels  hltcoe/coveragebench (Apache-2.0; arXiv 2603.20034)
set -euo pipefail
D=coveragebench/data
CORPUS=https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus/resolve/main/collections
CB=https://huggingface.co/datasets/hltcoe/coveragebench/resolve/main
mkdir -p $D/pyserini_corpus/crux $D/pyserini_queries $D/pyserini_qrels $D/nugget_qrels $D/nuggets $D/paper_runs

for f in duc04-test-docs.jsonl multi_news-test-docs.jsonl multi_news-train-docs.jsonl; do
  [ -s $D/pyserini_corpus/crux/$f ] || curl -sfL -o $D/pyserini_corpus/crux/$f $CORPUS/$f
done

# topics: "qid<TAB>query" (same as BRIGHT's pyserini_queries)
curl -sfL -o $D/pyserini_queries/crux_multinews.tsv $CB/topics/crux_multinews_topics.tsv
curl -sfL -o $D/pyserini_queries/crux_duc04.tsv     $CB/topics/duc04_topics.tsv
# nugget qrels "qid nugget_id docid 1" (ndeval / diversity format) -> alpha-nDCG, subtopic recall
curl -sfL -o $D/nugget_qrels/crux_multinews.qrels $CB/qrels/crux_multinews.qrels
curl -sfL -o $D/nugget_qrels/crux_duc04.qrels     $CB/qrels/crux_duc04.qrels
curl -sfL -o $D/nuggets/crux_multinews.jsonl "$CB/nuggets/multinews.jsonl."    # sic: trailing dot upstream
curl -sfL -o $D/nuggets/crux_duc04.jsonl     $CB/nuggets/duc04.jsonl
# document-level qrels "qid 0 docid 1" (relevant = supports at least one nugget) -> nDCG, recall (trec_eval)
for ds in crux_multinews crux_duc04; do
  awk '$4>0 {print $1"\t0\t"$3"\t1"}' $D/nugget_qrels/$ds.qrels | sort -u > $D/pyserini_qrels/$ds.tsv
done
# the paper's runs (BM25 / Qwen3-8B depth 100; rerankers on top 50), for checking our evaluation
for pair in CRUX-MultiNews:crux_multinews CRUX-DUC04:crux_duc04; do
  src=${pair%%:*}; ds=${pair##*:}
  for r in Initial-Retrieval/initial_bm25 Initial-Retrieval/initial_qwen3 Rank1-Reranking/rank1_bm25 \
           Rank1-Reranking/rank1_qwen3 Qwen3-Reranker/rerank_bm25_qwen3 Qwen3-Reranker/rerank_qwen3_qwen3; do
    curl -sfL -o $D/paper_runs/${ds}.$(basename $r).run $CB/ranking/$src/$r.run
  done
done
# upstream has Rank1-Reranking/rank1_qwen3.run swapped between CRUX-MultiNews and CRUX-DUC04 (checked: swapped back,
# they give the paper's nDCG@20 0.616 / 0.754)
if head -1 $D/paper_runs/crux_multinews.rank1_qwen3.run | grep -q '^duc04'; then
  mv $D/paper_runs/crux_multinews.rank1_qwen3.run $D/paper_runs/tmp.run
  mv $D/paper_runs/crux_duc04.rank1_qwen3.run $D/paper_runs/crux_multinews.rank1_qwen3.run
  mv $D/paper_runs/tmp.run $D/paper_runs/crux_duc04.rank1_qwen3.run
fi
wc -l $D/pyserini_corpus/crux/*.jsonl $D/pyserini_queries/crux_*.tsv $D/pyserini_qrels/crux_*.tsv
