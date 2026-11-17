#!/bin/bash
# Submit one ThinkQE job per dataset for a model. From the repo root:
#   bash snellius/submit_runs.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B dl19     # smoke test
#   bash snellius/submit_runs.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B all       # all 9 datasets
#   bash snellius/submit_runs.sh <model> biology economics                          # pick datasets
# Options (environment variables):
#   BATCH=true   one vLLM call per round instead of per query (needs batching.patch applied)
#   SEED=1       vLLM seed (default 0)
#   EXTRA='--think_split "</think>"' ETAG=thinksplit   extra thinkqe.py args + a label for the output folder
#   EXTRA='--no_thinking true --num_interaction 1' ETAG=nothink1r   (ablations; ETAG required with EXTRA)
#   e.g.  BATCH=true SEED=1 bash snellius/submit_runs.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B all
#   EVAL_ARGS='--log_trajectory true --top_passages_format docids --top_passages_k 100'
#                logging only (trajectory.jsonl + run_config.json, small top-psgs); same results and output folder
#   PARTITION=gpu_a100   run on A100 (40 GB) instead of the default H100; fine for <=8B models on 1 GPU
#                        (14B on A100 needs 2 GPUs, see run_thinkqe.job). Don't mix GPU types within a comparison.
#   Model litellm/<name> (e.g. litellm/Qwen3.6): generation via the UvA LiteLLM proxy, job on a CPU node
#                        (genoa; PARTITION overrides). Key: api_token in .env or $LITELLM_API_KEY.
#                        Costs proxy credits: smoke-test first, e.g. EXTRA='--num_interaction 1' ETAG=r1
# Slurm only bills time actually used, so the time limits below are generous on purpose.
set -euo pipefail
ENV_ARGS=()
if [ "${1:-}" = --uv ]; then
  ENV_ARGS=(--uv)
  shift
fi
MODEL=${1:?usage: submit_runs.sh [--uv] <hf_model_id> <all|dataset...>}; shift
BRIGHT=(biology earth_science economics psychology robotics stackoverflow sustainable_living)
if [ "${1:-all}" = "all" ]; then DATASETS=(dl19 dl20 "${BRIGHT[@]}"); else DATASETS=("$@"); fi

for d in "${DATASETS[@]}"; do
  case $d in dl19|dl20) T=04:00:00 ;; browsecomp) T=16:00:00 ;; *) T=08:00:00 ;; esac
  T=${TIME_LIMIT:-$T}   # optional override, e.g. TIME_LIMIT=01:30:00 (short jobs backfill sooner)
  if [ "${BATCH:-false}" = "true" ] && ! grep -q batch_generation thinkqe.py; then
    echo "BATCH=true but batching.patch is not applied (run: git apply snellius/batching.patch)"; exit 1; fi
  if [ -n "${EXTRA:-}" ] && [ -z "${ETAG:-}" ]; then echo "Set ETAG (a short label) when using EXTRA"; exit 1; fi
  # API models (litellm/<name>): generation runs on the proxy, so a CPU node is enough (only BM25 runs locally)
  RES=(${PARTITION:+--partition=$PARTITION})
  case $MODEL in litellm/*) RES=(--partition=${PARTITION:-genoa} --gpus=0 --cpus-per-task=16) ;; esac
  MODEL=$MODEL BATCH=${BATCH:-false} SEED=${SEED:-0} EXTRA="${EXTRA:-}" ETAG="${ETAG:-}" EVAL_ARGS="${EVAL_ARGS:-}" sbatch --export=ALL --time=$T ${RES[@]+"${RES[@]}"} \
      --job-name="tqe_${d}_$(basename "$MODEL")${ETAG:+_$ETAG}$([ "${BATCH:-false}" = true ] && echo _b)_s${SEED:-0}" \
      snellius/run_thinkqe.job "${ENV_ARGS[@]}" "$d"
done
squeue -u "$USER"
