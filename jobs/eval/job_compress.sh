#!/bin/bash
#SBATCH --job-name=compress
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/compress_%j.out
#SBATCH --error=logs/compress_%j.err

# Shorten existing answers into the band the benchmark's humans actually rated
# (66-106 words, p50 88), so baseline-vs-injection can be scored where the judges
# were validated. One short call per existing answer -- no retrieval, no drafts,
# no regeneration. The untouched `baseline` rows are copied into the same file so
# the judge pairs them per question.
#
#   RESP=overton_responses_flat.jsonl COND=merge_v2 OUT=overton_responses_comp.jsonl \
#     sbatch jobs/eval/job_compress.sh
#
# Then judge the result with BOTH judges and compare in band:
#   RESP=$OUT SCORES=overton_scores_comp.csv       MODEL=<72B-AWQ> TP=4 MAXU=20 ...
#   RESP=$OUT SCORES=overton_scores_comp_q38.csv   MODEL=<27B-FP8> TP=2 MAXU=20 ...
#
# Knobs: RESP, COND, KEEP, LABEL, WORDS, OUT, MODEL, TP, PORT, MAXQ.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs

RESP="${RESP:-overton_responses_flat.jsonl}"
COND="${COND:-merge_v2}"
KEEP="${KEEP:-baseline}"
LABEL="${LABEL:-${COND}_compressed}"
WORDS="${WORDS:-90}"          # median length of the human-rated responses
# MATCH beats a fixed target: a global "about 90 words" came back at p50 128,
# and baseline is often below the 66-word band floor anyway. Matching each
# rewrite to its own question's baseline length is what makes the arms
# comparable. Empty = use WORDS for every row.
MATCH="${MATCH:-baseline}"
MATCH_ARG=""
[ -n "${MATCH}" ] && MATCH_ARG="--match_condition ${MATCH}"
OUT="${OUT:-overton_responses_comp.jsonl}"
MAXQ="${MAXQ:-0}"
# Same generator that wrote the originals, so the rewrite is not a second
# model's paraphrase on top of the arm being tested.
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"

echo "RESP=${RESP} COND=${COND} -> ${LABEL} @ ${WORDS} words  KEEP=${KEEP}  OUT=${OUT}"
[ -f "${RESP}" ] || { echo "MISSING ${RESP} in $(pwd)"; exit 1; }

"${VLLM}" serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len 8192 > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null; then break; fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM died -- see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }

CUDA_VISIBLE_DEVICES="" python -u -m evaluation.overton.compress_responses \
    --responses "${RESP}" --condition "${COND}" --keep "${KEEP}" \
    --label "${LABEL}" --target_words "${WORDS}" ${MATCH_ARG} \
    --max_questions "${MAXQ}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" --out "${OUT}" \
    || { echo "COMPRESSION FAILED (see .err)"; exit 1; }

echo ""
echo "Done -> ${OUT}"
echo "CHECK THE MANIPULATION FIRST: ${LABEL} must land in 66-106 words like"
echo "baseline does, or the arm is not length-matched and proves nothing."
