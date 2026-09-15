#!/bin/bash
#SBATCH --job-name=vital
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/vital_%j.out
#SBATCH --error=logs/vital_%j.err

# VITAL: arm-vs-arm comparisons at ~27x OvertonBench's questions (docs/vital_task.tex).
# Same ATP graph and retrieval as the OvertonBench runs; the TASK is swapped.
# Serve the 72B-AWQ generator, generate, stop the server, then score with
# score_vital.py on the freed GPU (mpnet over tens of thousands of units).
#
# ARM-VS-ARM ONLY. baseline vs an injected arm is invalid here: coverage is capped
# by length (60 words per value), and injected answers run ~5x longer.
#
# Geometry (the main use): one job per arm, tagged, then score both files together.
#   CURV=0.5 EMB=embeddings_opinionqa_c0p5.pt TAG=c0p5 sbatch jobs/eval/job_vital.sh
#   CURV=0   EMB=embeddings_opinionqa_c0.pt   TAG=c0   sbatch jobs/eval/job_vital.sh
#   SKIPGEN=1 SCORE_RESP=vital_responses_c0p5.jsonl,vital_responses_c0.jsonl \
#     SCORES=vital_scores_geom.csv sbatch jobs/eval/job_vital.sh
#
# Resumable: rerunning skips (situation, condition, rollout) rows already in OUT,
# so a timeout needs only a resubmit with the same knobs.
#
# Knobs: SITU, EMB, FEATS, CURV, SEED, DATASET, CONDS, TAG, MAXQ, SEEDQ, NROLL,
#        OUT, SCORES, SCORE_RESP, SKIPGEN, MODEL, TP, PORT.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs

SITU="${SITU:-vital_overton_valuekaleidoscope.json}"   # ONLY this VITAL file:
                                 # the opinionqa/globalopinionqa ones are Pew,
                                 # this project's own graph source
CURV="${CURV:-0.5}"
CTAG="c${CURV//./p}"
EMB="${EMB:-embeddings_opinionqa_${CTAG}.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
SEED="${SEED:-42}"               # MUST match train.py --seed: node ids depend on it
DATASET="${DATASET:-opinionqa}"
CONDS="${CONDS:-merge_v2}"
TAG="${TAG:-${CTAG}}"
MAXQ="${MAXQ:-300}"              # 0 = all 1,648. merge_v2 is 4 generations per
                                 # answer; start with a shuffled subset
SEEDQ="${SEEDQ:-0}"              # subset shuffle -- keep IDENTICAL across arms
NROLL="${NROLL:-1}"
OUT="${OUT:-vital_responses_${TAG}.jsonl}"
SCORES="${SCORES:-vital_scores_${TAG}.csv}"
SCORE_RESP="${SCORE_RESP:-${OUT}}"
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"

echo "SITU=${SITU} EMB=${EMB} CURV=${CURV} CONDS=${CONDS} TAG=${TAG} MAXQ=${MAXQ} SEEDQ=${SEEDQ}"
[ -f "${SITU}" ] || { echo "MISSING ${SITU} in $(pwd)"; exit 1; }

if [ "${SKIPGEN:-0}" != "1" ]; then
    [ -f "${EMB}" ] || { echo "MISSING ${EMB} in $(pwd)"; exit 1; }
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

    echo "=== generate ==="
    # The client must not touch the GPUs vLLM holds.
    CUDA_VISIBLE_DEVICES="" python -u -m evaluation.overton.eval_vital \
        --situations "${SITU}" --embeddings "${EMB}" --text_feat "${FEATS}" \
        --dataset "${DATASET}" --curvature "${CURV}" --seed "${SEED}" \
        --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
        --conditions "${CONDS}" --tag "${TAG}" --n_rollouts "${NROLL}" \
        --max_questions "${MAXQ}" --seed_situations "${SEEDQ}" --out "${OUT}" \
        || { echo "GENERATION FAILED (see .err)"; exit 1; }

    # Free the GPUs for the scorer's embedder.
    kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null
    trap - EXIT
fi

echo "=== score ${SCORE_RESP} ==="
python -u scripts/analysis/score_vital.py --situations "${SITU}" \
    --responses "${SCORE_RESP}" --out "${SCORES}" \
    || { echo "SCORING FAILED (see .err)"; exit 1; }

echo ""
echo "Done -> ${SCORES}"
echo "Read arm-vs-arm deltas WITH their word difference. A delta between arms of"
echo "different length is partly length (docs/vital_task.tex, Proposition 1)."
