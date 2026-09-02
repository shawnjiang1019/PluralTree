#!/bin/bash
#SBATCH --job-name=judge_only
#SBATCH --gres=gpu:4
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/judge_only_%j.out
#SBATCH --error=logs/judge_only_%j.err

# Judge an EXISTING responses file. Serves the judge model, scores, exits.
#
#   RESP=cad_responses_a.jsonl SCORES=cad_scores_a.csv sbatch jobs/eval/job_judge_only.sh
#
# job_overton_eval.sh cannot do this: it validates --conditions against
# retrieval.answer.CONDITIONS, which has no `cad*` entries, so a CAD file is
# rejected before stage 2 is reached. The judge itself does not care what the
# condition is called -- it scores whatever rows are in the file.
#
# Knobs: RESP, SCORES, MODEL, MAXU, KROLL, PORT, TP.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

RESP="${RESP:-cad_responses_a.jsonl}"
SCORES="${SCORES:-cad_scores_a.csv}"
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
MAXU="${MAXU:-20}"
# Per-cluster hit/miss. Free: the judge already computes the covered
# sets and discards them after printing the union table.
CLUSTERS="${CLUSTERS:-${SCORES%.csv}_clusters.csv}"
KROLL="${KROLL:-0}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"

[ -f "${RESP}" ] || { echo "MISSING ${RESP} in $(pwd)"; exit 1; }
echo "RESP=${RESP}  SCORES=${SCORES}  MODEL=${MODEL}  MAXU=${MAXU}"
echo "rows: $(grep -c . "${RESP}")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${VLLM}" serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len 8192 > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null' EXIT

for _ in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null; then break; fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM died — see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 20
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }
echo "vLLM up"

python -u -m evaluation.overton.judge_overtonbench --score "${RESP}" \
    --max_users "${MAXU}" --k_rollouts "${KROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --out "${SCORES}" \
    --dump_clusters "${CLUSTERS}" \
    || { echo "JUDGING FAILED"; exit 1; }

echo ""
echo "Done -> ${SCORES}"
echo "For CAD: compare arms to base7b/ctx0 ONLY. These are a 7B model; v9/v10/v11"
echo "are Qwen-72B-AWQ and are NOT a valid reference."
