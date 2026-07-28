#!/bin/bash
#SBATCH --job-name=contestedness
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/contestedness_%j.out
#SBATCH --error=logs/contestedness_%j.err

# Adaptation 1 of docs/adaptive_injection.md: self-consistency contestedness.
# Sample K committed answers per question and measure how much the model's own
# STANCE varies -- then score that gate POST-HOC against a finished run
# (SCORES), so we learn whether the signal routes without generating anything
# new. This is the same design that killed the graph signals (route_signal).
#
# Cost: 60 questions x K samples (default 5) = ~300 short generations.
#
# Smoke:  MAXQ=5 sbatch jobs/eval/job_contestedness.sh
# Full:   SCORES=overton_scores_v6.csv sbatch jobs/eval/job_contestedness.sh
# Knobs:  MODEL, K, TEMP, THR, SCORES, INJECT, MAXQ, OUT
#
# Prereqs: MODEL weights + mpnet (the held-out embedder) in HF_HOME.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
K="${K:-5}"                      # samples per question (spread needs K>=4)
TEMP="${TEMP:-1.0}"              # must be >0 or every sample is identical
THR="${THR:-0.35}"               # inject iff contestedness > THR
SCORES="${SCORES:-overton_scores_v6.csv}"   # post-hoc gate evaluation target
INJECT="${INJECT:-scout}"        # which injected condition the gate would pick
MAXQ="${MAXQ:-0}"                # 0 = all 60
OUT="${OUT:-contestedness.json}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
echo "MODEL=${MODEL} K=${K} TEMP=${TEMP} THR=${THR} SCORES=${SCORES} INJECT=${INJECT}"

vllm serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len 8192 > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null" EXIT

for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
        echo "vLLM up after ~$((i * 10))s"; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "vLLM died — see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }

# The GPUs belong to vLLM; the mpnet embedder must stay on CPU.
export CUDA_VISIBLE_DEVICES=""

SCORES_FLAG=""
if [ -f "${SCORES}" ]; then SCORES_FLAG="--scores ${SCORES} --inject_cond ${INJECT}";
else echo "note: ${SCORES} not found — skipping post-hoc gate evaluation"; fi

python -m retrieval.contestedness \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --k "${K}" --temperature "${TEMP}" --threshold "${THR}" \
    --max_questions "${MAXQ}" --out "${OUT}" ${SCORES_FLAG} \
    || { echo "CONTESTEDNESS FAILED (see .err)"; exit 1; }

echo "Done. Scores: ${OUT}"
