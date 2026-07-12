#!/bin/bash
#SBATCH --job-name=hivemind_div
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/hivemind_div_%j.out
#SBATCH --error=logs/hivemind_div_%j.err

# INFINITY-CHAT mode-collapse eval (Artificial Hivemind, Fig 4): serve MODEL with
# vLLM, sample N responses/query per condition, then measure intra-pool
# self-similarity. Tests whether scout-injected divergence lowers mode collapse.
#
# Smoke test:  NQ=5 NS=8 CONDS=baseline sbatch jobs/eval/job_hivemind_diversity.sh
# Baseline:    CONDS=baseline sbatch jobs/eval/job_hivemind_diversity.sh
# Full:        sbatch jobs/eval/job_hivemind_diversity.sh   (100 q x 50 x 3 conds)
# Knobs: MODEL, CONDS, NS(samples), NQ(queries), EMB, FEATS, DATASET, TAU, TP, EVAL_MODEL
#
# Generation is resumable: rerunning tops each (query,condition) pool back up to
# NS samples, so a timeout just needs a resubmit.
#
# Prereq (login node, once): the held-out eval embedder must be in HF_HOME —
#   huggingface-cli download BAAI/bge-large-en-v1.5
# (distinct from the scout's MiniLM to keep the eval independent of retrieval).

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
CONDS="${CONDS:-baseline,scout,div_only}"
NS="${NS:-50}"                   # samples per (query, condition)
NQ="${NQ:-100}"                  # queries (INFINITY-CHAT100)
TAU="${TAU:-0.1}"
EMB="${EMB:-embeddings_opinionqa.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
DATASET="${DATASET:-opinionqa}"
GEN="${GEN:-hivemind_gen.jsonl}"
DIV="${DIV:-hivemind_diversity.csv}"
EVAL_MODEL="${EVAL_MODEL:-BAAI/bge-large-en-v1.5}"   # held-out eval embedder (NOT MiniLM)
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"
echo "MODEL=${MODEL} CONDS=${CONDS} NS=${NS} NQ=${NQ} DATASET=${DATASET}"

# baseline-only runs need no embeddings; pass them only for injected conditions.
EMB_ARGS=""
case "${CONDS}" in
  *scout*|*div_only*) EMB_ARGS="--embeddings ${EMB} --text_feat ${FEATS} --dataset ${DATASET} --tau ${TAU}";;
esac

"${VLLM}" serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
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

# Client stages must not touch CUDA (MiniLM would OOM on the vLLM-filled GPU).
export CUDA_VISIBLE_DEVICES=""

echo "=== stage 1: generate ==="
python -m evaluation.hivemind.generate_hivemind \
    --conditions "${CONDS}" --num_samples "${NS}" --num_queries "${NQ}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    ${EMB_ARGS} --out "${GEN}" \
    || { echo "GENERATION FAILED (see .err)"; exit 1; }
echo "fork fallbacks (injected pools that got the baseline prompt):"
grep -c "0 forks" logs/hivemind_div_${SLURM_JOB_ID}.err || true

echo "=== stage 2: diversity metrics ==="
python -m evaluation.hivemind.diversity_metrics "${GEN}" --out "${DIV}" \
    --eval_model "${EVAL_MODEL}" \
    || { echo "METRICS FAILED (see .err)"; exit 1; }

echo "Done. Generations: ${GEN}  Diversity: ${DIV}"
