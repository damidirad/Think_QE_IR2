# ThinkQE results: Qwen3.6 (UvA LiteLLM API) on TREC DL, BRIGHT and CoverageBench CRUX, plus BrowseComp-Plus

Status 2026-11-23: all runs finished. Every run, including BrowseComp-Plus and CRUX, passed `snellius/check_trajectory.py` (`check_trajectory_report.txt` in each run folder).

- **ThinkQE retrieval runs:** 9 of 9 done (TREC DL19, DL20 and the 7 BRIGHT StackExchange domains).
- **CoverageBench CRUX (information coverage):** done for CRUX-MultiNews and CRUX-DUC04; see section 2d.
- **BrowseComp-Plus run:** done with DeepSeek-R1-0528-Qwen3-8B on a local GPU; see section 2b.
- **RAG answer evaluation:** done for all 7 BRIGHT domains, with closed-book and BM25 baselines (2026-11-24). TREC DL has no reference answers, so it has no RAG evaluation.
- **Retrieval-dynamics analysis of the local-model runs (no API):** section 2c, taken from the paper draft.

All numbers are from one run with seed 0. The API seed is best-effort, so reruns are not bit-identical.

## 1. Retrieval results

### TREC DL (MAP / nDCG@10 / R@1k, final round)
| | Qwen3.6 (ours) | Paper R1-14B | Qwen3-8B (scur0900, 3 seeds) | Qwen3.8-27B (scur0900, 3 seeds) |
|---|---|---|---|---|
| DL19 | **46.0 / 67.4 / 89.0** | 45.9 / 68.8 / 89.3 | 44.3 / 64.7 / 87.8 | 45.7 / 66.6 / 88.3 |
| DL20 | **46.1 / 67.7 / 88.7** | 43.9 / 64.7 / 87.8 | 41.8 / 63.2 / 88.4 | 46.3 / 66.9 / 87.9 |

Per round, Qwen3.6:

| Round | DL19 nDCG@10 | DL19 MAP | DL19 R@1k | DL20 nDCG@10 | DL20 MAP | DL20 R@1k |
|---|---|---|---|---|---|---|
| 0 (BM25) | 50.6 | 30.1 | 75.0 | 48.0 | 28.6 | 78.6 |
| 1 | 65.9 | 44.5 | 88.7 | 66.8 | 45.2 | 88.2 |
| 2 | 66.9 | 45.7 | 89.1 | 66.5 | 45.2 | 88.5 |
| 3 | 67.4 | 46.0 | 89.0 | 67.7 | 46.1 | 88.7 |

Recall at the top of the ranking, per round (relevance ≥ 2, from `*.metrics.json`):

| Round | DL19 R@5 | DL19 R@10 | DL20 R@5 | DL20 R@10 |
|---|---|---|---|---|
| 0 (BM25) | 11.4 | 17.5 | 15.8 | 24.7 |
| 1 | 15.5 | 23.6 | 23.7 | 35.6 |
| 2 | 15.5 | 24.6 | 24.3 | 34.5 |
| 3 | 16.2 | 25.0 | 25.2 | 35.0 |

DL queries have many relevant passages (often dozens), so R@5 and R@10 are low by construction. They measure the share of all relevant passages in the top 5 or top 10.

### BRIGHT (nDCG@10, final round)
| Domain | Qwen3.6 (ours) | R1-14B (scur0900, 3 seeds) | Qwen3-8B (scur0900, 3 seeds) | Paper R1-14B |
|---|---|---|---|---|
| biology | **56.7** | 46.4 | 48.7 | 47.3 |
| earth_science | **58.1** | 51.7 | 54.2 | 52.5 |
| economics | 29.2 | 28.5 | 27.0 | 29.2 |
| psychology | 38.4 | 38.6 | 36.3 | 40.0 |
| robotics | **23.6** | 20.8 | 20.3 | 19.3 |
| stackoverflow | **30.4** | 28.5 | 28.8 | 28.0 |
| sustainable_living | 28.4 | 28.1 | 28.9 | 27.9 |
| **average** | **37.8** | 34.7 | 34.9 | 34.9 |

Per round, Qwen3.6, as nDCG@10 / R@100:

| Round | biology | earth_sci | economics | psychology | robotics | stackoverflow | sust_living | average |
|---|---|---|---|---|---|---|---|---|
| 0 (BM25) | 18.4 / 42.1 | 28.4 / 60.0 | 16.4 / 40.8 | 13.4 / 37.7 | 10.9 / 37.1 | 16.6 / 41.0 | 16.1 / 42.0 | 17.2 / 42.9 |
| 1 | 56.1 / 83.0 | 57.9 / 80.1 | 28.9 / 58.5 | 38.1 / 69.6 | 21.1 / 52.8 | 29.2 / 63.9 | 27.4 / 65.2 | 37.0 / 67.6 |
| 2 | 57.2 / 82.9 | 57.8 / 79.2 | 28.7 / 57.1 | 39.4 / 69.4 | 22.8 / 53.4 | 30.1 / 63.7 | 28.6 / 64.8 | 37.8 / 67.2 |
| 3 | 56.7 / 81.9 | 58.1 / 79.5 | 29.2 / 58.2 | 38.4 / 71.5 | 23.6 / 56.5 | 30.4 / 62.3 | 28.4 / 65.0 | 37.8 / 67.8 |

Per round, Qwen3.6, as R@5 / R@10. These come from trec_eval with the same qrels, computed afterwards from the saved `bm25-aug<r>_result_retrieval.trec` files, and are also appended to each run's `metrics_per_round.txt`:

| Round | biology | earth_sci | economics | psychology | robotics | stackoverflow | sust_living | average |
|---|---|---|---|---|---|---|---|---|
| 0 (BM25) | 16.4 / 22.4 | 24.7 / 32.9 | 10.8 / 17.7 | 11.1 / 15.8 | 10.3 / 14.5 | 14.3 / 19.8 | 12.7 / 20.7 | 14.3 / 20.6 |
| 1 | 47.6 / 59.2 | 49.1 / 59.7 | 23.4 / 31.5 | 37.1 / 46.6 | 20.5 / 24.1 | 23.2 / 33.9 | 26.7 / 34.8 | 32.5 / 41.4 |
| 2 | 49.1 / 59.8 | 48.8 / 59.1 | 22.2 / 33.2 | 36.2 / 46.5 | 22.3 / 25.4 | 24.1 / 33.8 | 27.8 / 35.4 | 32.9 / 41.9 |
| 3 | 48.8 / 60.0 | 48.5 / 59.3 | 23.7 / 32.3 | 35.6 / 45.7 | 21.1 / 27.5 | 24.3 / 34.0 | 27.3 / 35.5 | 32.8 / 42.0 |

R@5 is the share of a query's gold documents that are in the top 5 passed to the answer model in section 2. It more than doubles from BM25 (14.3) to ThinkQE's final round (32.8). This matches the short-answer refusals halving, from 41% to 21%.

Observations:
- **Gains by round:** almost all of the gain comes in round 1. Rounds 2 and 3 move nDCG@10 by about ±1.
- **Comparison with the local models:** Qwen3.6 averages 37.8 on BRIGHT, against 34.7–34.9 for the local models and the paper.
- **Biology and earth science:** the lead there is 4–10 points. Qwen3.6 is a recent model and may have seen BRIGHT or StackExchange data in training, so treat these two gaps with caution.
- **Truncated generations:** 4 of 4,494 BRIGHT generations hit the token limit and have no `</think>` (2 in earth_science, 1 in biology, 1 in psychology).
- **Mean completion length:** 2.8k–3.7k tokens per generation on BRIGHT and about 1.7k on DL.

## 2. RAG answer evaluation (BRIGHT, top 5 documents of ThinkQE's final round)

EM, F1 and BERTScore are ×100. The judge score is on a 0–100 scale.

### Short answers (FlashRAG prompt)
| Domain | n | EM | F1 | BERTScore (rescaled) | BERTScore (raw) | Judge | Refusals |
|---|---|---|---|---|---|---|---|
| biology | 103 | 0.0 | 13.2 | -14.7 | 80.6 | 34.8 | 22 |
| earth_science | 116 | 0.0 | 13.9 | -13.2 | 80.9 | 38.3 | 22 |
| economics | 103 | 0.0 | 12.6 | -16.7 | 80.3 | 27.6 | 20 |
| psychology | 101 | 0.0 | 9.5 | -20.6 | 79.6 | 27.8 | 19 |
| robotics | 101 | 0.0 | 11.2 | -14.1 | 80.7 | 28.4 | 23 |
| stackoverflow | 117 | 0.0 | 13.3 | -12.5 | 81.0 | 47.5 | 14 |
| sustainable_living | 108 | 0.0 | 10.0 | -13.8 | 80.8 | 18.3 | 39 |
| **macro-average** | 749 | **0.0** | **12.0** | **-15.1** | **80.6** | **31.8** | 159 (21%) |

### Long answers (BRIGHT QA prompt)
| Domain | n | EM | F1 | BERTScore (rescaled) | BERTScore (raw) | Judge |
|---|---|---|---|---|---|---|
| biology | 103 | 0.0 | 23.4 | -14.9 | 80.6 | 75.6 |
| earth_science | 116 | 0.0 | 20.3 | -15.8 | 80.5 | 79.7 |
| economics | 103 | 0.0 | 19.5 | -18.3 | 80.0 | 62.1 |
| psychology | 101 | 0.0 | 18.8 | -18.8 | 79.9 | 62.3 |
| robotics | 101 | 0.0 | 13.0 | -22.5 | 79.3 | 56.8 |
| stackoverflow | 117 | 0.0 | 19.0 | -6.0 | 82.1 | 65.7 |
| sustainable_living | 108 | 0.0 | 21.3 | -15.8 | 80.5 | 57.8 |
| **macro-average** | 749 | **0.0** | **19.4** | **-16.0** | **80.4** | **65.7** |

Data quality:
- Across all 2 × 749 answers there are no empty answers and no unparsable judge outputs.
- One long answer (robotics) hit the token limit.
- Short answers average 30–46 words. Long answers average 520–800 words. The gold answers average 120–300 words.

How to read it:
- **EM is 0 by construction.** The gold answers are full StackExchange answers, so no answer can match one exactly.
- **BERTScore barely discriminates.** The raw score is about 80 for every style and domain. The rescaled score is negative because the generated answers and the gold answers are long and differ in length. Gold answers longer than 512 tokens are truncated for BERTScore.
- **The judge is the most informative metric.** Long answers score twice as high as short ones (65.7 vs 31.8), and the domains are clearly separated. F1 agrees, more weakly.
- **Refusals pull down the short-answer scores.** The FlashRAG prompt asks for an answer based on the given documents. In 21% of short answers the model replied that the documents do not contain the answer. This count comes from a regex, so it is approximate. Short-answer quality therefore mainly measures whether the top 5 documents contain the answer.

### Judge vs. refusals, and short vs. long answers
- **The judge score is the quality metric.** gpt-oss-120b sees the question, the answer and the reference, and rates how much of the reference content the answer covers (0–100). Every answer is judged.
- **The refusal count is a diagnostic, not a score.** It counts answers that decline, such as "The provided documents do not contain information…", found with a regex. Refusals are still judged and almost always get 0. In biology, 19 of 22 scored 0; partial refusals got 10–30. A low short-answer judge score therefore has two causes: retrieval failure (a refusal) or an incomplete answer. The refusal rate separates them.
- **Short and long answers use the same question and documents.** Only the instruction differs:
  - **short** (FlashRAG): "answer based on the given document, only the answer". Answers are 30–46 words, and the model refuses when the documents lack the answer, so this style mainly measures whether retrieval surfaced the answer. It is also penalised for brevity against references of 120–300 words.
  - **long** (BRIGHT): "think step by step… give a detailed answer". Answers are 520–800 words, refusals are ≤3%, and the model adds its own knowledge, so this style measures the full system (retrieval plus model).
- **Example:** biology query 0 (insects and LED light), with the same ThinkQE documents. The short answer was a refusal and got judge 0. The long answer, 604 words, got 80.
- The rubric asks for steps of 10, but the judge sometimes gives values in between (e.g. 95). They are kept as given.

### Baselines: closed book and BM25 (all 7 domains)
Each baseline uses the same answer model, sampling settings, judge, metrics and queries as the ThinkQE condition. Only the documents in the prompt change:
- **Closed book:** Qwen3.6 alone, with no documents (`--k 0`).
  - Short prompt: FlashRAG's no-retrieval prompt, "Answer the question based on your own knowledge. Only give me the answer and do not output any other words."
  - Long prompt: BRIGHT's QA prompt with the document blocks left out.
- **BM25:** the top 5 of round 0, i.e. plain BM25 on the original query with no expansion (`--round 0`).
- **ThinkQE:** the top 5 of round 3, as in the tables above.

Macro-averages over the 7 domains (749 queries):

| Condition | Short: F1 | Short: BERTScore raw | Short: Judge | Short: refusals | Long: F1 | Long: BERTScore raw | Long: Judge |
|---|---|---|---|---|---|---|---|
| Closed book | **12.8** | 78.9 | **36.6** | 3 | 18.4 | 80.1 | 64.3 |
| BM25 top 5 | 9.2 | 80.2 | 21.6 | 309 (41%) | 19.0 | 80.5 | 63.6 |
| ThinkQE top 5 | 12.0 | 80.6 | 31.8 | 159 (21%) | **19.4** | 80.4 | **65.7** |

EM is 0.0 in every condition.

Judge score per domain (short / long):

| Domain | Closed book | BM25 top 5 | ThinkQE top 5 |
|---|---|---|---|
| biology | 46.1 / 74.3 | 14.5 / 72.2 | 34.8 / 75.6 |
| earth_science | 40.2 / 74.2 | 24.5 / 74.8 | 38.3 / 79.7 |
| economics | 33.2 / 63.0 | 18.0 / 60.1 | 27.6 / 62.1 |
| psychology | 32.6 / 59.8 | 15.8 / 57.3 | 27.8 / 62.3 |
| robotics | 29.8 / 52.9 | 21.9 / 55.3 | 28.4 / 56.8 |
| stackoverflow | 48.1 / 67.2 | 45.8 / 67.4 | 47.5 / 65.7 |
| sustainable_living | 26.5 / 58.7 | 10.6 / 57.8 | 18.3 / 57.8 |
| **macro-average** | **36.6** / 64.3 | 21.6 / 63.6 | 31.8 / **65.7** |

Paired differences (same queries). Each row gives the macro-average difference with a 95% bootstrap CI (2,000 resamples of queries within each domain), the Wilcoxon signed-rank p over all 749 queries, and the number of domains where the first condition is ahead:

| Comparison | Short: Judge | Short: F1 | Long: Judge | Long: F1 |
|---|---|---|---|---|
| ThinkQE − BM25 | **+10.2** [+8.4, +12.1], p<1e-20, 7/7 | **+2.8** [+2.3, +3.3], p<1e-20, 6/7 | **+2.2** [+0.5, +3.9], p=0.04, 5/7 | **+0.4** [+0.2, +0.6], p=2e-4, 6/7 |
| ThinkQE − closed book | −4.9 [−7.1, −2.5], p=3e-6, 0/7 | −0.8 [−1.5, −0.1], p=0.10, 2/7 | +1.4 [−0.4, +3.3], p=0.77, 4/7 | **+1.0** [+0.8, +1.2], p<1e-18, 7/7 |
| Closed book − BM25 | +15.1 [+12.8, +17.3], p<1e-30, 7/7 | +3.6 [+2.9, +4.3], p<1e-20, 7/7 | +0.7 [−1.1, +2.5], p=0.11, 4/7 | −0.6 [−0.9, −0.4], p=4e-8, 0/7 |

What the baselines show:
- **ThinkQE's documents help more than BM25's in both styles.** Short answers gain +10.2 judge points and win in all 7 domains. Long answers gain +2.2, which is a small but significant difference.
- **Qwen3.6 alone is a strong baseline.** On long answers, closed book (64.3) sits between BM25 (63.6) and ThinkQE (65.7); neither difference to closed book is significant on the judge. The model already knows much of BRIGHT's StackExchange content (closed-book judge 74 on biology and earth science), which fits the contamination caveat in section 1. The retrieved documents improve word overlap with the reference (long-answer F1 +1.0 for ThinkQE over closed book, 7/7 domains) more than the judged content.
- **Short answers are best without documents.** The "answer based on the given document" prompt makes the model refuse when the top 5 lack the answer: 41% of the time with BM25, 21% with ThinkQE, and almost never in closed book, where it answers from its own knowledge. For this style, the gap between the RAG conditions reflects how often retrieval surfaces the answer. ThinkQE halves the refusals relative to BM25.
- **Closed-book side effect:** in 15 short answers and 2 long ones the model kept reasoning until it hit the 32k-token limit and gave no answer. Those answers score 0, which explains closed-book psychology's low BERTScore (raw 73.9).
- **Noise:** each condition has one answer sample per query at temperature 0.6. The CIs cover query variance, not answer-sampling variance, so treat differences of 1–2 judge points as tentative.

## 2b. BrowseComp-Plus retrieval (DeepSeek-R1-0528-Qwen3-8B, local vLLM)

Qwen3.6 was **not** run on BrowseComp-Plus. Its data must not be sent to an external API, and the LiteLLM backend refuses it. This run uses DeepSeek-R1-0528-Qwen3-8B on a local H100 with vLLM.

- **Run:** Slurm job 29375184 on `gpu_h100`, 2 h 43 min, seed 0. It covers all 830 queries, with 1,660 generations per round and none missing the `</think>` split.
- **ThinkQE settings:** same as above, with `--max_demo_len 512`.
- **Long-document options:** `--index index_lean --text_index index --passage_text_depth 5 --fast_truncate true`. These leave the ranking unchanged; see `snellius/README_SNELLIUS.md`, "BrowseComp-Plus".
- **Evaluation:** trec_eval with `-c` over all 830 queries.
  - **evidence qrels:** the documents needed to answer the question.
  - **gold qrels:** the documents that contain the final answer.

| Round | Evidence: nDCG@10 | Evidence: R@5 | Evidence: R@10 | Evidence: R@100 | Evidence: R@1000 | Gold: nDCG@10 | Gold: R@5 | Gold: R@10 | Gold: R@100 | Gold: R@1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (BM25) | 1.63 | 1.18 | 1.73 | 4.66 | 13.65 | 1.66 | 1.39 | 2.06 | 6.10 | 17.28 |
| 1 | 1.80 | 1.31 | 1.68 | 4.56 | 13.09 | 1.96 | 2.10 | 2.58 | 6.68 | 17.74 |
| 2 | 1.91 | 1.38 | 1.78 | 4.62 | 12.28 | 2.23 | 2.14 | 2.82 | 6.78 | 16.05 |
| 3 | **2.05** | **1.50** | **1.93** | **4.99** | 12.17 | **2.48** | **2.53** | **3.13** | **6.99** | 16.05 |

What the table shows:
- **Top of the ranking:** BrowseComp-Plus queries are long, multi-constraint puzzles, and single-shot BM25 retrieval is close to the floor. ThinkQE's top-10 metrics improve steadily over the rounds: gold nDCG@10 goes from 1.66 to 2.48, and R@5 from 1.39 to 2.53.
- **Deep recall:** R@1000 falls as the accumulated expansions drift. Gold R@1000 drops from 17.7 after round 1 to 16.1, and evidence R@1000 from 13.7 to 12.2.
- **Smoke run:** the 20-query run `browsecomp_DeepSeek-R1-0528-Qwen3-8B__smoke20_batched_s0` (jobs 29331147 and 29339037) was only a pipeline check. Its metrics are averaged over all 830 queries, so they are not meaningful.

## 2c. Retrieval-dynamics analysis of the local-model runs (no API; from the paper draft)

These results come from the paper draft's "Extension and Further Analysis" section and appendix, not from this
document's Qwen3.6 runs. They analyse scur0900's local-GPU runs with 3 seeds. The analysis scripts are not in this repo.
The numbers were transcribed from the draft and cross-checked for internal consistency (group sizes, weighted means,
round-to-round gains, percentages and the per-query top-5 counts all add up). They were not recomputed from the run files.

### Query difficulty and expansion rounds (BRIGHT, 7 domains)
- 749 queries × 3 seeds = 2,247 query-seed observations, pooled over queries and seeds.
- The draft section does not name the model.
- Queries are grouped by their initial BM25 nDCG@10 x: Zero (x = 0), Low (0 < x ≤ 0.25), Medium (0.25 < x ≤ 0.50), High (x > 0.50).

Mean nDCG@10 per round:

| Difficulty | N | BM25 | R1 | R2 | R3 |
|---|---|---|---|---|---|
| Zero | 454 | 0.0000 | 0.1462 | 0.1824 | 0.1882 |
| Low | 99 | 0.1627 | 0.3532 | 0.3782 | 0.3811 |
| Medium | 94 | 0.3668 | 0.5852 | 0.5953 | 0.5915 |
| High | 102 | 0.7701 | 0.8227 | 0.8106 | 0.8044 |
| **Overall** | 749 | 0.1724 | 0.3208 | 0.3457 | 0.3482 |

These are pooled per-query means. The BRIGHT tables in section 1 are macro-averages over domains.

Mean marginal gain Δt = nDCG@10(t) − nDCG@10(t−1), and oracle gain = max over t ∈ {0..3} of nDCG@10(t), minus nDCG@10(3):

| Difficulty | R0→R1 | R1→R2 | R2→R3 | Oracle |
|---|---|---|---|---|
| Zero | +0.1462 | +0.0362 | +0.0057 | +0.0165 |
| Low | +0.1905 | +0.0250 | +0.0028 | +0.0440 |
| Medium | +0.2185 | +0.0101 | −0.0038 | +0.0622 |
| High | +0.0526 | −0.0121 | −0.0062 | +0.0762 |
| **Overall** | +0.1484 | +0.0249 | +0.0025 | +0.0340 |

Findings:
- **Zero-score queries:** 454 queries (60.6%) start at nDCG@10 = 0. Among those query-seed observations, nDCG@10 first becomes positive at R1 in 35.39%, at R2 in 4.26% and at R3 in 1.84%. The other 58.52% never recover.
- **Diminishing returns:** gains are +0.148 at R1, +0.025 at R2 and +0.003 at R3. Medium queries gain the most in R1. High-baseline queries lose in both R2 and R3.
- **Earlier rounds can be better:** stopping at the best round would add +0.034 on average and +0.076 for high-baseline queries. Choosing that round needs relevance judgments, so this is an upper bound, not a usable stopping rule. It motivates adaptive stopping based on signals available at retrieval time.

### Document-level dynamics (TREC DL19)
- Qwen3.8-27B, all 43 judged queries, 3 seeds, 3 rounds × 2 expansions.
- Including BM25, that is 516 top-5 lists with 2,580 document occurrences.

**Gold-document reachability:** gold means relevance ≥ 2. Recall is cumulative, counting the union of documents retrieved up to that round, so R1–R3 are higher than the per-round R@1k in section 1.

| Depth | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| 5 | 0.1137 | 0.1844 | 0.2017 | 0.2080 |
| 10 | 0.1751 | 0.2734 | 0.2870 | 0.2916 |
| 100 | 0.4910 | 0.6356 | 0.6499 | 0.6523 |
| 1000 | 0.7501 | 0.8881 | 0.8984 | 0.9023 |

At depth 1000, newly found gold documents per query-seed are 9.155 at R1, 0.419 at R2 and 0.093 at R3. 94.71% of all new discoveries happen in R1.

**Passage filter (redundancy):** reconstructed from the saved candidates with the original filter logic. Values are means per query-seed generation.

| | R1 | R2 | R3 |
|---|---|---|---|
| Inspected | 5.00 | 7.71 | 10.40 |
| Rejected | 0.00 | 2.71 | 5.40 |
| Accepted | 5.00 | 5.00 | 5.00 |
| Rejection rate (%) | 0.00 | 31.64 | 51.63 |
| Blacklist size | 0.00 | 2.71 | 5.47 |
| Generations affected (%) | 0.00 | 87.60 | 100.00 |

The rejection rate is a mean of per-generation ratios, so it is not Rejected / Inspected of the means. The filter always ends up with 5 distinct passages, but it has to read about twice as deep by R3.

**Semantic similarity:**
- Method: precomputed SBERT embeddings, calibrated on 20,349 unique judged documents from DL19 and DL20.
- Cross-round cosine similarity between distinct documents rises from 0.5530 at R1 to 0.6449 at R3. The draft says within-round similarity stays roughly stable but gives no numbers.
- Semantic novelty, measured over lists that introduce previously unseen documents, falls from 0.3297 to 0.2153.
- Conclusion: the retrieved documents converge, which motivates diversity-aware passage selection and adaptive stopping.

### Five DL19 case studies (Qwen3.8-27B, seed 0; appendix)
The draft chose these queries to show contrasting outcomes, so they are not a representative sample.

| Query | Text | BM25 | R1 | R2 | R3 | Δ (R3 − BM25) | Rel@5 (BM25→R1→R2→R3) | Pattern |
|---|---|---|---|---|---|---|---|---|
| 19335 | anthropological definition of environment | 0.605 | 0.747 | 0.817 | 0.838 | +0.233 | 2→4→5→5 | gradual improvement |
| 962179 | when was the salvation army founded | 0.066 | 1.000 | 1.000 | 0.940 | +0.874 | 0→5→5→5 | strong R1 gain |
| 489204 | right pelvic pain causes | 0.314 | 0.136 | 0.138 | 0.135 | −0.179 | 2→0→0→0 | decline |
| 182539 | example of monotonic function | 0.509 | 0.615 | 0.611 | 0.615 | +0.106 | 1→2→2→3 | moderate improvement |
| 183378 | exons definition biology | 0.377 | 1.000 | 0.943 | 0.943 | +0.566 | 2→5→5→5 | strong R1 gain |

nDCG@10 is shown per round. Rel@5 is the number of top-5 passages with grade ≥ 2.

Observations:
- **Top-5 overlap with the previous round:** 4–5 of 5 passages from R2 on in every case. The top of the ranking stops moving after R1.
- **Query 489204:** the expansions are fluent lists of causes, yet they replace both relevant BM25 passages with unjudged or grade-1 ones. A plausible expansion does not guarantee better retrieval.

## 2d. CoverageBench CRUX: information coverage (Qwen3.6)

CoverageBench (arXiv 2603.20034, `hltcoe/coveragebench`) links each topic's nuggets (sub-questions a good report
must answer) to the documents that support them. A ranking can then be scored on how many different nuggets its top
documents cover, not only on relevance. We use its CRUX part: report-style requests over one 565,015-passage news
corpus (`DylanJHJ/crux-mds-corpus`), with 100 MultiNews and 50 DUC04 topics.

- **Metrics** (`utils/coverage_metrics.py`):
  - **nDCG@10/20:** binary; a document is relevant if it supports at least one nugget.
  - **α-nDCG@20:** α = 0.5, rewards covering nuggets not yet covered higher in the ranking.
  - **StRec@20:** subtopic recall, the share of a topic's nuggets covered by the top 20.
  - **R@100:** recall of relevant documents.
  - The evaluator reproduces every number the paper reports for its released runs. Our round 0 (BM25, pyserini defaults) is identical to the paper's BM25 run.
- **ThinkQE settings:** as above, with `--max_demo_len 512`.
- **Jobs:** chained on genoa: 29620656 (MultiNews, 2 h 22 min) and 29620657 (DUC04, 1 h 16 min).
- **Baselines:** the paper's released runs, evaluated with the same code. "Rerank" rows rerank the first-stage top 100.

**CRUX-MultiNews (100 topics)**

| Run | nDCG@10 | nDCG@20 | α-nDCG@20 | StRec@20 | R@100 |
|---|---|---|---|---|---|
| ThinkQE round 0 (BM25) | 41.2 | 42.9 | 46.9 | 63.4 | 58.6 |
| ThinkQE round 1 | 52.8 | 55.6 | 57.8 | 72.9 | 74.8 |
| ThinkQE round 2 | 53.7 | 56.4 | 59.3 | 74.5 | 76.2 |
| **ThinkQE round 3** | **54.0** | **56.9** | **59.6** | **74.4** | **76.7** |
| BM25 → Rank1 rerank | 53.5 | 53.3 | 57.0 | 71.0 | 52.8 |
| BM25 → Qwen3 rerank | 48.9 | 49.3 | 55.1 | 68.9 | 52.8 |
| Qwen3-8B dense | 57.7 | 61.0 | 64.8 | 81.3 | 79.3 |
| Qwen3-8B dense → Rank1 rerank | 57.9 | 61.6 | 61.3 | 83.1 | 73.5 |
| Qwen3-8B dense → Qwen3 rerank | 58.1 | 62.0 | 65.2 | 81.7 | 73.5 |

**CRUX-DUC04 (50 topics)**

| Run | nDCG@10 | nDCG@20 | α-nDCG@20 | StRec@20 | R@100 |
|---|---|---|---|---|---|
| ThinkQE round 0 (BM25) | 53.0 | 44.8 | 47.6 | 65.9 | 48.0 |
| **ThinkQE round 1** | **68.3** | **61.6** | **61.2** | **76.4** | **67.5** |
| ThinkQE round 2 | 67.0 | 60.6 | 60.2 | 74.7 | 65.0 |
| ThinkQE round 3 | 67.6 | 60.8 | 60.1 | 74.4 | 64.9 |
| BM25 → Rank1 rerank | 72.2 | 61.0 | 61.6 | 73.2 | 41.1 |
| BM25 → Qwen3 rerank | 64.0 | 52.4 | 55.4 | 69.6 | 41.1 |
| Qwen3-8B dense | 77.9 | 70.1 | 65.2 | 81.4 | 74.6 |
| Qwen3-8B dense → Rank1 rerank | 81.2 | 75.4 | 67.6 | 83.8 | 65.1 |
| Qwen3-8B dense → Qwen3 rerank | 78.7 | 71.3 | 66.3 | 82.6 | 65.1 |

What the tables show:
- **Coverage gain over BM25:** the final round adds +12.7 (MultiNews) and +12.5 (DUC04) α-nDCG@20, and +11.0 and +8.5 StRec@20.
- **Against reranking BM25:** ThinkQE beats the Qwen3 reranker on both sets. Against Rank1, a reasoning reranker, it is ahead on MultiNews (α-nDCG@20 59.6 vs 57.0) and level on DUC04 (60.1 vs 61.6), with higher StRec@20 on both. Its R@100 is far higher, because expansion changes which documents are retrieved, while reranking only reorders BM25's top 100.
- **Against dense retrieval:** ThinkQE stays behind Qwen3-8B dense retrieval by 5.2 (MultiNews) and 5.1 (DUC04) α-nDCG@20.
- **Rounds:** almost all of the gain comes in round 1. On MultiNews rounds 2–3 add +1.8 α-nDCG@20; on DUC04 they lose 1.1, as accumulated expansions drift.
- **Generation:** 900 generations in total: 9 hit the length limit, and 9 have no `</think>` split. The filter skipped 857 (MultiNews) and 435 (DUC04) documents, and the LLM never read below rank 12.
- **API usage:** 896 requests, 0.68M prompt and 3.22M completion tokens, 22 retries (all on MultiNews).
- **Caveats:** the topics are news events from 1990s–2010s that the LLM may know. Some reasoning traces call the retrieved passages "outdated" and draw on the model's own knowledge. The qrels are LLM-made, and the topic sets are small (50 topics for DUC04), so differences of 1–2 points are within noise.

Reproduce: `python snellius/coverage_eval.py rounds $WORK/outputs/crux_*_Qwen3.6__batched_s0` (setup: `snellius/README_SNELLIUS.md`, "CoverageBench CRUX").

## 3. Evaluation setup

### ThinkQE retrieval
- **Code:** `thinkqe.py` with `--generation_model litellm/Qwen3.6`, through `vllm_server/litellm_completion.py` (see `snellius/README_SNELLIUS.md`, "API models").
- **LLM:** Qwen3.6, served as `Qwen/Qwen3.6-27B-FP8` behind https://llmproxy.uva.nl. Thinking was on. The reasoning comes back separately and is rejoined as `<reasoning>\n</think>\n\n<answer>`.
- **ThinkQE hyperparameters (paper defaults):**
  - 3 rounds (`--num_interaction 3`), 2 samples per query per round (`--gen_num 2`)
  - temperature 0.7, top_p 1
  - 5 passages per prompt (`--keep_passage_num 5`), `--max_demo_len` 128 on DL and 512 on BRIGHT
  - accumulation on, redundancy filter on, query repeat weight 3
  - prompt plus output ≤ 32768 tokens
- **Seed:** 0. Each sample is a separate request with seed `seed*1000+i`.
- **Retrieval:** BM25 in pyserini 0.22.1, top 1000.
  - DL uses the `msmarco-v1-passage` index; BRIGHT uses the local indexes in `bright/data/pyserini_indexes/<domain>`.
  - The Lucene clause limit was raised to 16384. This changes nothing for queries under 1,024 unique terms. Without it, long round-3 queries were silently dropped.
- **Submission:**

  ```
  CHAIN=true BATCH=true EVAL_ARGS='--log_trajectory true --top_passages_format docids --top_passages_k 100' \
      bash snellius/submit_runs.sh litellm/Qwen3.6 <datasets>
  ```

  The jobs ran on genoa CPU nodes and were chained, so only one domain called the API at a time.
- **Metrics:**
  - DL: trec_eval (MAP and recall use relevance ≥ 2).
  - BRIGHT: nDCG@10 and R@100 against `bright/data/pyserini_qrels/<domain>.tsv`.

### RAG answer evaluation (`snellius/rag_eval.py`, `snellius/rag_eval.job`)
- **Input:** the original BRIGHT query and the top 5 documents of the final round (round 3), each capped at 1000 words.
  - In the short prompt, documents are shown as `Doc i: text` without titles, because BRIGHT doc ids leak the source question.
- **Reference answer:** BRIGHT `gold_answer`, the accepted StackExchange answer, taken from the HF `xlangai/BRIGHT` `examples` split. `$\begingroup$`/`$\endgroup$` markers are stripped.
- **Answer model:** Qwen3.6 with thinking on, temperature 0.6, top_p 0.95, n = 1, seed 0, max 32768 tokens. The answer is the text after `</think>`; the reasoning is stored as well.
- **Prompts:**
  - **short:** the FlashRAG default prompt (RUC-NLPIR/FlashRAG `base_prompt.py`): "Answer the question based on the given document. Only give me the answer and do not output any other words…"
  - **long:** BRIGHT's QA prompt (`generate_answer.py`): documents and the post in delimited blocks, ending with "Let's think step by step to address the query and give a detailed answer."
- **Judge:** gpt-oss-120b on the same proxy, temperature 0, using BRIGHT's judge prompt (`eval_claude.py`).
  - The judge scores 0–100 in steps of 10, as the percentage of the reference content the answer addresses.
  - The score is parsed from the last `SCORE:` in the output. A blank answer gets 0 without a judge call.
- **Baselines:** `--k 0` (closed book; FlashRAG's no-retrieval prompt for short answers, BRIGHT's prompt without documents for long ones) and `--round 0` (BM25 top 5); everything else is unchanged:

  ```
  sbatch snellius/rag_eval.job <the 7 BRIGHT run dirs> -- --k 0       # job 29617116 -> rag_eval_closedbook/
  sbatch snellius/rag_eval.job <the 7 BRIGHT run dirs> -- --round 0   # job 29617036 -> rag_eval_r0_k5/
  ```
- **Metrics:**
  - EM and token F1 use the FlashRAG/SQuAD normalisation (lowercase, strip punctuation and articles).
  - BERTScore F1 uses `bert-score` 0.3.13 with roberta-large, layer 17. Both raw and baseline-rescaled scores are reported.
  - All metrics are macro-averaged over domains.

## 4. Where everything is (Snellius, readable by everyone)

Root: `/scratch-shared/scur0920/thinkqe/outputs/`

| Run | Folder | Slurm job(s) |
|---|---|---|
| DL19 | `dl19_Qwen3.6__batched_s0/` | 29399650 |
| DL20 | `dl20_Qwen3.6__batched_s0/` | 29399651 |
| biology | `biology_Qwen3.6__batched_s0/` | 29425749 |
| earth_science | `earth_science_Qwen3.6__batched_s0/` | 29425750 (crashed in round 3), 29461412 (final) |
| economics | `economics_Qwen3.6__batched_s0/` | 29425751 (crashed in round 3), 29461413 (final) |
| psychology | `psychology_Qwen3.6__batched_s0/` | 29425752 (crashed in round 3), 29461414 (final) |
| robotics | `robotics_Qwen3.6__batched_s0/` | 29425753 |
| stackoverflow | `stackoverflow_Qwen3.6__batched_s0/` | 29425754 |
| sustainable_living | `sustainable_living_Qwen3.6__batched_s0/` | 29425755 |
| BrowseComp-Plus (R1-0528-Qwen3-8B) | `browsecomp_DeepSeek-R1-0528-Qwen3-8B__batched_s0/` | 29375184 |
| BrowseComp-Plus smoke test (20 queries) | `browsecomp_DeepSeek-R1-0528-Qwen3-8B__smoke20_batched_s0/` | 29331147, 29339037 |
| CRUX-MultiNews | `crux_multinews_Qwen3.6__batched_s0/` | 29620656 |
| CRUX-DUC04 | `crux_duc04_Qwen3.6__batched_s0/` | 29620657 |
| RAG evaluation logs | `_logs/rag_eval_29425767.out` (ThinkQE, 4 domains), `_logs/rag_eval_29461415.out` (ThinkQE: earth_science, economics, psychology), `_logs/rag_eval_29617036.out` (BM25 baseline), `_logs/rag_eval_29617116.out` (closed-book baseline) | |

`_logs/` also holds the logs of the first BRIGHT attempts, which failed on proxy errors (jobs 29421416–22 and 29424040), and of the DL19 smoke test (29396203).

The final jobs for earth_science, economics and psychology replayed every LLM response from the API cache, so their trajectories are identical to the generations of the crashed attempts.

Files in each run folder:
- `bm25-aug<r>_result_retrieval.trec`: TREC run file for round r (r = 0 is plain BM25, r = 3 is final).
- `*.metrics.json`: metrics per round.
- `*.responses.json`: the LLM outputs.
- `*.topics.jsonl`: the expanded queries.
- `*top-psgs.json`: the top-100 document ids.
- `trajectory.jsonl`: one line per query. Each round records the BM25 query, the doc ids the LLM read, the docs the filter skipped, the prompt, both responses, finish reasons, token counts, the expansions and the top-100 doc ids with scores.
- `run_config.json`: all arguments, git commit, package versions and the Slurm job id. `extra_args.txt` holds any extra arguments.
- `check_trajectory_report.txt` and `metrics_per_round.txt`: consistency check and per-round metrics.
- `slurm_<jobid>.out`: full job log, including reasoning traces and the per-round API usage line.
- `coverage_per_round.txt` (CRUX only): coverage metrics per round, with the paper's runs as reference rows.
- `*.metrics_gold.json` (BrowseComp-Plus only): per-round metrics against the gold qrels. `*.metrics.json` there uses the evidence qrels.
- `rag_eval_final_k5/` (BRIGHT only):
  - `prompts_{short,long}.jsonl`
  - `answers_{short,long}.jsonl` (answer, reasoning, finish reason, tokens)
  - `judge_{short,long}.jsonl` (judge text and parsed score)
  - `per_query.tsv` (all metrics per query and style)
  - `metrics.json`
- `rag_eval_r0_k5/` and `rag_eval_closedbook/` (BRIGHT only): the same files for the BM25 and closed-book baselines.

Other locations:
- **API response cache:** `/scratch-shared/scur0920/thinkqe/api_cache/`, readable by the owner only. Reruns with the same settings replay from it at no cost.
- **Original Slurm logs:** in the repo root on scur0920's home, named `thinkqe_tqe_<ds>_Qwen3.6_b_s0_<job>.out` and `rag_eval_<job>.out`. Home is not purged.

Reproduce the tables:

```
source snellius/common.sh
python snellius/rag_eval.py summary $WORK/outputs/*_Qwen3.6__batched_s0     # add --sub rag_eval_r0_k5 or --sub rag_eval_closedbook for the baselines
cat $WORK/outputs/<run>/metrics_per_round.txt
```

**BrowseComp-Plus folders** contain the decrypted benchmark questions in plain text, in `*.topics.jsonl`, `trajectory.jsonl`, `*.responses.json` and the Slurm logs. They are readable by Snellius users, but the dataset's terms require they stay off the web. Do not copy them off Snellius, commit them to git, paste them into chatbots or send them to an API. Aggregate metrics, such as the table in section 2b, are fine to share.

### Retention
`/scratch-shared` deletes files that have not been used for about 14 days.

- Everything under `/scratch-shared/scur0920/` was touched on 2026-11-23, so it is safe until about **2026-12-07**.
- To refresh, run `find /scratch-shared/scur0920 -exec touch -h {} +`. Only the owner can do this.
- Copy anything you want to keep long-term elsewhere.
- Many of scur0900's runs are already purged: the R1-14B DL runs and the ablations. The R1-14B BRIGHT, Qwen3-8B and Qwen3.8-27B DL results in the tables above still exist as of 2026-11-23.
