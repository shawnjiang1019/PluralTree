#!/bin/bash
#SBATCH --job-name=goqa
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/goqa_%j.out
#SBATCH --error=logs/goqa_%j.err

# GlobalOpinionQA steerable + distributional evaluation (evaluation/overton/eval_goqa.py).
# Serve the generator, then one first-token logprob call per prompt: cheap.
#
#   sbatch jobs/eval/job_goqa.sh
#   MODEL=<llama-70b-awq> OUT=goqa_llama.jsonl sbatch jobs/eval/job_goqa.sh
#
# Re-print the summary later without model calls:
#   python -m evaluation.overton.eval_goqa --summarise_only --out goqa_steer.jsonl
#
# Knobs: MODEL, TP, PORT, SOURCE, MAXQ, MAXC, MAXSIM, SEED, OUT.

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

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
SOURCE="${SOURCE:-WVS}"          # WVS avoids Pew overlap with the ATP graph
MAXQ="${MAXQ:-150}"
MAXC="${MAXC:-8}"
MAXSIM="${MAXSIM:-0.9}"
SEED="${SEED:-0}"
OUT="${OUT:-goqa_steer.jsonl}"

echo "MODEL=${MODEL} SOURCE=${SOURCE} MAXQ=${MAXQ} MAXC=${MAXC} OUT=${OUT}"

vllm serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len 8192 --max-logprobs 20 > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null' EXIT
for _ in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/health" > /dev/null && break
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "vLLM died -- see logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }

CUDA_VISIBLE_DEVICES="" python -u -m evaluation.overton.eval_goqa \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --source "${SOURCE}" --max_questions "${MAXQ}" --max_countries "${MAXC}" \
    --max_sim "${MAXSIM}" --seed "${SEED}" --out "${OUT}" \
    || { echo "GOQA EVAL FAILED (see .err)"; exit 1; }
