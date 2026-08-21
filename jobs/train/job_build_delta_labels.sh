#!/bin/bash
#SBATCH --job-name=delta_labels
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/delta_labels_%j.out
#SBATCH --error=logs/delta_labels_%j.err

# Delta labels on GRAPH questions, so the injection router can be FIT off
# OvertonBench and REPORTED on it. scripts/analysis/delta_regressor.py currently
# learns from 60 questions that are also its eval set; the graph holds ~1,492 ATP
# (~2,500 GOQA) questions that are not. See scripts/build_delta_labels.py.
#
# Cost: N questions x 2 conditions. N=500 -> ~1000 generations, the injected half
# of them at max_tokens=4096. Budget ~4-6h at N=500 on 4 GPUs; GENERATION IS
# RESUMABLE (keyed on (question_id, condition)), so a walltime kill just needs a
# resubmit with the same OUT.
#
# Smoke:  N=8 sbatch jobs/train/job_build_delta_labels.sh
# Full:   N=500 sbatch jobs/train/job_build_delta_labels.sh
# Knobs:  MODEL N TAU SEED EMB FEATS DATASET OUT INJECT MATCH_THR DEPTH EMBEDDER TP
#
# RESCORE ONLY (after the reward match_thr is recalibrated -- no endpoint, no
# vLLM, no regeneration; runs in minutes):
#   SCORE_ONLY=1 MATCH_THR=0.35 sbatch jobs/train/job_build_delta_labels.sh
#
# Prereqs (login node, once): MODEL weights, the mpnet reward embedder and MiniLM
# (graph load) in HF_HOME; OPINIONQA_DIR populated; EMB/FEATS built by the embed
# job at the SAME --seed (node ids depend on the clustering seed).

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
# raw-ATP OpinionQA (offline; no SubPOP gate) -- see data/loaders/opinionqa.py
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs results

# AWQ, not the unquantized 72B: on 4x A100-40GB the fp16 weights leave no room
# for prompt + KV at --max-model-len 8192 and vLLM refuses to start. Same model
# as jobs/eval/job_contestedness.sh and the merge run in submit_experiments.sh.
#
# THE LABELS ARE MODEL-SPECIFIC. "Did injection help" is a claim about a
# particular baseline -- a 72B that already covers the spectrum unaided has
# nothing to gain from injection where a 7B has plenty. Labels made here do NOT
# transfer to a 7B policy. Whatever model you route for must be MODEL below.
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
N="${N:-500}"                    # questions to label (2 generations each)
TAU="${TAU:-0.25}"               # scout gate: on-domain opinionqa 0.25, GOQA ~0.1
SEED="${SEED:-42}"               # graph split AND sampling seed; MUST match the embed job
EMB="${EMB:-embeddings_opinionqa.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
DATASET="${DATASET:-opinionqa}"
INJECT="${INJECT:-scout}"        # injected condition compared against baseline
OUT="${OUT:-results/delta_labels_reward_scores.csv}"
#            No 'vN' in that name on purpose: delta_regressor.run_tag would file
#            these REWARD labels under an OvertonBench run tag and pool them with
#            JUDGE labels. The script refuses such a name outright.
EMBEDDER="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"
# Reward knobs. Unset = RewardConfig defaults (match_thr 0.50, min_depth_words 60).
# match_thr=0.50 currently FAILS its gate -- docs/reward_gate_failure.md: it sits
# above the p75=0.475 of the cosines it thresholds, only 18% of positions clear
# it, and 92% of responses score exactly 0. Set MATCH_THR to regenerate labels at
# a recalibrated value; do not edit alignment/reward.py from this job.
MATCH_THR="${MATCH_THR:-}"
DEPTH="${DEPTH:-}"
SCORE_ONLY="${SCORE_ONLY:-}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"

REWARD_FLAGS=""
if [ -n "${MATCH_THR}" ]; then REWARD_FLAGS="${REWARD_FLAGS} --match_thr ${MATCH_THR}"; fi
if [ -n "${DEPTH}" ]; then REWARD_FLAGS="${REWARD_FLAGS} --min_depth_words ${DEPTH}"; fi

echo "MODEL=${MODEL} N=${N} TAU=${TAU} SEED=${SEED} DATASET=${DATASET} INJECT=${INJECT}"
echo "OUT=${OUT} REWARD_FLAGS=[${REWARD_FLAGS}] SCORE_ONLY=[${SCORE_ONLY}]"

# --- rescore path: no vLLM, no generation, GPU for the mpnet embedder --------
if [ -n "${SCORE_ONLY}" ]; then
    python scripts/build_delta_labels.py --score_only \
        --dataset "${DATASET}" --seed "${SEED}" --inject_cond "${INJECT}" \
        --model "${MODEL}" --embedder "${EMBEDDER}" --out "${OUT}" ${REWARD_FLAGS} \
        || { echo "RESCORE FAILED (see .err)"; exit 1; }
    echo "Done (rescore). Scores: ${OUT}"
    exit 0
fi

# --- offline sanity before burning a 4-GPU allocation ------------------------
python scripts/build_delta_labels.py --selftest \
    || { echo "SELFTEST FAILED -- not starting vLLM"; exit 1; }

"${VLLM}" serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len 8192 > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null" EXIT

for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
        echo "vLLM up after ~$((i * 10))s"; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "vLLM died -- see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }

# The GPUs belong to the vLLM server. The client stage is HTTP plus a CPU scout;
# MiniLM (graph load / question embedding) OOMs if it lands on a GPU that is 95%
# Qwen -- same reason job_overton_eval.sh does this.
export CUDA_VISIBLE_DEVICES=""

echo "=== stage 1: generate (resumable; resubmit this job to continue) ==="
python scripts/build_delta_labels.py --generate_only \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset "${DATASET}" \
    --curvature 0.5 --tau "${TAU}" --seed "${SEED}" --n "${N}" \
    --inject_cond "${INJECT}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --out "${OUT}" \
    || { echo "GENERATION FAILED (see .err)"; exit 1; }
echo "fork fallbacks (injected answers that got the baseline prompt):"
grep -c "scout returned 0 forks" logs/delta_labels_${SLURM_JOB_ID}.err || true
echo "tag failures (injected answers missing <answer> tags):"
grep -c "missing <answer> tags" logs/delta_labels_${SLURM_JOB_ID}.err || true

# Release the GPUs before scoring: stage 2 is a sentence-transformers pass over
# every response, ~3 texts/sec on CPU vs seconds on a GPU (measured in
# jobs/eval/job_reward_correlation.sh). Generation is already durable on disk, so
# a failure here costs a rescore, not a regeneration.
echo "=== releasing vLLM before scoring ==="
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null
trap - EXIT
unset CUDA_VISIBLE_DEVICES

echo "=== stage 2: score both answers with coverage_reward ==="
python scripts/build_delta_labels.py --score_only \
    --dataset "${DATASET}" --seed "${SEED}" --inject_cond "${INJECT}" \
    --model "${MODEL}" --embedder "${EMBEDDER}" --out "${OUT}" ${REWARD_FLAGS} \
    || { echo "SCORING FAILED (see .err)"; exit 1; }

echo ""
echo "Done. Scores: ${OUT}"
echo "WATCH both_zero_rate above -- questions where the reward scored BOTH"
echo "answers 0.0 carry no routing signal. They are emitted (both_zero=1), not"
echo "dropped. A high rate means the LABELS are thin, not that routing is hard;"
echo "rerun with SCORE_ONLY=1 MATCH_THR=<recalibrated> once the reward is fixed."
echo ""
echo "Next: python scripts/analysis/delta_regressor.py --scores ${OUT} \\"
echo "        --inject_conds ${INJECT} --features causal --embeddings ${EMB} \\"
echo "        --text_feat ${FEATS} --dataset ${DATASET} --seed ${SEED}"
