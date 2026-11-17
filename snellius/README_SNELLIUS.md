# Reproducing ThinkQE's main results (Tables 3 and 4) on Snellius

All commands run from the repo root (`~/Think_QE_IR2`), with `snellius/` inside it.

## One-time setup (login node)
    bash snellius/00_setup_env.sh            # env for R1-14B, OpenThinker2, Phi-4-reasoning
    bash snellius/00_setup_env.sh qwen3      # env for Qwen3-8B/14B (only needed for step 4)
    tmux                                     # so downloads survive disconnects
    bash snellius/01_download.sh             # BRIGHT data, MS MARCO index, 1.5B + 14B models
    sbatch snellius/02_index_bright.job      # BRIGHT Lucene indexes (CPU, ~minutes)

To use an existing uv environment in the repo's `.venv`, pass `--uv` as the first
argument to the script or job. Without the flag, the scripts activate Conda.

    bash snellius/01_download.sh --uv
    sbatch snellius/02_index_bright.job --uv
    sbatch snellius/bm25_rm3.job --uv
    sbatch snellius/run_thinkqe.job --uv dl19
    bash snellius/submit_runs.sh --uv deepseek-ai/DeepSeek-R1-Distill-Qwen-14B all

## Batched generation (TA-approved)
    git apply snellius/batching.patch       # once; adds --batch_generation and --seed (default path unchanged)
    BATCH=true bash snellius/submit_runs.sh <model> all        # SEED=1, 2, ... for repeat runs

## Evaluation logging (opt-in; retrieval results unchanged)
    EVAL_ARGS='--log_trajectory true --top_passages_format docids --top_passages_k 100' \
        BATCH=true bash snellius/submit_runs.sh <model> all
- `--log_trajectory true` writes `trajectory.jsonl` (one line per query; per round: `bm25_query`, `llm_input_docids`
  (what the LLM read, after the filter), `skipped_by_filter`, `prompt`, `responses`, `finish_reasons`,
  `completion_tokens`, `think_split_found`, `expansions`, `top_docids`/`top_scores` (top `--trajectory_top_k`, default 100))
  and `run_config.json` (arguments, git commit, package versions, Slurm job id). Rewritten after every round.
- `--top_passages_format docids` / `--top_passages_k N` shrink `*top-psgs.json` (default: full texts of all 1000 hits,
  which is ~25 GB per round on BrowseComp-Plus).
- `--quiet_log true` keeps the Slurm log small: no per-query prompts, reasoning traces, expansions or blacklisted
  passage texts (all in `trajectory.jsonl` when logging is on); one summary line per round instead.
  Recommended for BrowseComp-Plus: `EVAL_ARGS='--log_trajectory true --quiet_log true --top_passages_format docids --top_passages_k 100'`
- `EVAL_ARGS` does not change the output folder name (unlike `EXTRA`/`ETAG`).

## BrowseComp-Plus (extension)
Data: `Tevatron/browsecomp-plus` (queries, encrypted; decrypt with the dataset card's script into
`/scratch-shared/$USER/browsecomp-plus/decrypted.jsonl`) and `Tevatron/browsecomp-plus-corpus` (parquet, in
`/scratch-shared/$USER/browsecomp-plus-corpus/`). Then, once:
    python snellius/prepare_browsecomp.py          # login node, thinkqe_310 env -> .../browsecomp-plus/pyserini/
    sbatch snellius/index_browsecomp.job           # CPU, ~15 min: index (full text) + index_lean (docids only)
Run (dataset name `browsecomp`; 830 queries; metrics vs evidence qrels during the run, gold qrels at the end):
    EVAL_ARGS='--log_trajectory true --quiet_log true --top_passages_format docids --top_passages_k 100' \
        BATCH=true bash snellius/submit_runs.sh deepseek-ai/DeepSeek-R1-0528-Qwen3-8B browsecomp
    source snellius/common.sh; python snellius/check_trajectory.py $WORK/outputs/browsecomp_DeepSeek-R1-0528-Qwen3-8B__batched_s0 --slurm_log thinkqe_<job>.out
Documents are long (BM25's top hits average ~370K characters, up to ~10M), so the job adds three options that leave
the results unchanged: `--index index_lean --text_index index` (search without loading every hit's stored text: same
ranking and scores, 14-170x faster), `--passage_text_depth 5` (texts in memory; deeper ones loaded on demand) and
`--fast_truncate true` (spaCy tokenizer on a prefix; same truncated text, avoids spaCy's 1M-character limit).
The decrypted queries, topics, qrels, outputs and logs are plain text of a benchmark that must stay off the web:
keep them on Snellius, never commit or upload them.

## Runs
1. Smoke test (~1 job):   `bash snellius/submit_runs.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B dl19`
2. Main model (9 jobs):   `bash snellius/submit_runs.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B all`
3. Check:                 `python snellius/collect_results.py`
4. Other models (download first with `bash snellius/01_download.sh <model>`), then
   `bash snellius/submit_runs.sh <model> all` for each of:
   Qwen/Qwen3-8B, Qwen/Qwen3-14B, open-thoughts/OpenThinker2-7B, microsoft/Phi-4-reasoning

## API models via the UvA LiteLLM proxy (extension)
`bash snellius/submit_runs.sh litellm/Qwen3.6 dl19` runs ThinkQE with generation on https://llmproxy.uva.nl
(`vllm_server/litellm_completion.py`); the job runs on a CPU node (genoa). Key: `api_token=...` in the repo's `.env`
(gitignored) or `$LITELLM_API_KEY`; it never appears in argv, logs or `run_config.json`.
Same defaults as the local runs: thinking on, temperature 0.7, top_p 1, 2 samples, prompt + output <= 32768 tokens.
Differences: `--no_thinking true` sends `enable_thinking=False` instead of the `</think>` prefill; the server returns
the reasoning separately and it is put back as `<reasoning>\n</think>\n\n<answer>`; one request per sample
(seed `seed*1000+i`), so the API's seed is best-effort, not bit-reproducible.
Every request spends proxy credits: smoke-test with `EXTRA='--num_interaction 1' ETAG=r1` and watch the
"LiteLLM usage so far" lines in the log. Refuses BrowseComp-Plus (its data must stay off the web).

Monitor: `squeue -u $USER`; logs are `thinkqe_tqe_<dataset>_<model>_<jobid>.out` in the repo root.
Outputs: `/scratch-shared/$USER/thinkqe/outputs/<dataset>_<model>/` (cleaned after ~14 days unused; copy what you keep).

Expect small differences from the paper: generation samples at temperature 0.7 with no fixed seed.
