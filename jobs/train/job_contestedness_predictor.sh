#!/bin/bash
#SBATCH --job-name=contest_pred
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/contest_pred_%j.out
#SBATCH --error=logs/contest_pred_%j.err

# Fit the TEXT-KEYED contestedness predictor and score it as a routing signal.
#
#   sbatch jobs/train/job_contestedness_predictor.sh
#
# Context: injection helps +0.31 on contested questions and hurts -0.45 on
# consensus ones; oracle routing = 0.634 vs 0.497 always-baseline. Two routers
# already failed -- graph divergence (+0.19, killed by the scout's max-W
# SELECTION) and the model's own <think> self-report (0.072). This job trains a
# third candidate whose label comes from the survey leaves (free: no generation,
# no judge) and whose INPUT is question text only, so it also works on questions
# with no graph anchor.
#
# Stages:
#   0  --selftest on a synthetic fixture (seconds). Cheap gate: if the target
#      cannot rank hand-built contested/consensus questions, stop here.
#   1  fit + held-out-TOPIC CV on both graphs, ATP<->GOQA transfer both ways,
#      for every --target (z_level / entropy / w_mean).
#   2  if SCORES exists, score the PREDICTION against the OvertonBench
#      per-question help-delta. THE NUMBER THAT MATTERS: it must beat w_raw's
#      +0.19 from evaluation/overton/route_signal.py.
#
# Knobs: DATASET (primary graph), TARGET (z_level|entropy|w_mean|all), MODEL
#        (ridge|logistic), EMBEDDER, GROUND (ordinal|unordered), AXIS_AGG,
#        NULL_BINS, NTEST (topics held out per fold), SCORES, SEED, OUT, CROSS.
#
# GPU, not CPU: the script never pins a device, so SentenceTransformer takes
# CUDA when one is visible. ~4k questions through mpnet is seconds on a GPU and
# ~20 min on a login node. Everything after the embedding pass is numpy.
#
# SEED must match the run that produced SCORES: opinionqa's topic clustering
# (and therefore node ids and the topic folds) is seeded.
#
# Prereqs: mpnet + MiniLM and the overtonbench dataset pre-downloaded into
# HF_HOME; OPINIONQA_DIR populated; SCORES present in the repo root.
# (No scout is run here, so the opinionqa-0.25 / GOQA-0.1 TAU split does not
# apply -- this job never touches the retrieval gate.)

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs docs

DATASET="${DATASET:-opinionqa}"
TARGET="${TARGET:-all}"          # all => report which target best predicts the delta
MODEL="${MODEL:-ridge}"          # ~4k examples: linear only, no MLP
EMBEDDER="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"
                                 # deliberately NOT the scout's MiniLM: the router
                                 # must not be a function of the retrieval it gates
GROUND="${GROUND:-ordinal}"      # ordinal W1 on the option index (Likert scales);
                                 # `unordered` = TV, for nominal option sets
AXIS_AGG="${AXIS_AGG:-mean}"     # mean, NOT max -- max-over-candidates is the
                                 # selection effect that killed the graph router
NULL_BINS="${NULL_BINS:-4}"      # 1 reproduces route_signal's degenerate global
                                 # null, where z is just an affine map of w_mean
NNULL="${NNULL:-400}"
NTEST="${NTEST:-2}"              # 12 ATP topics -> 10 train / 2 test per fold
ALPHAS="${ALPHAS:-1,10,100,1000}"
SEED="${SEED:-42}"
SCORES="${SCORES:-overton_scores_v6.csv}"
INJECT="${INJECT:-scout}"
OUT="${OUT:-contest_${MODEL}_${DATASET}.npz}"
EMB_CACHE="${EMB_CACHE:-docs/contest_emb}"
CROSS="${CROSS:-1}"              # 1 = also load the other graph, transfer both ways
echo "DATASET=${DATASET} TARGET=${TARGET} MODEL=${MODEL} GROUND=${GROUND}"
echo "NULL_BINS=${NULL_BINS} AXIS_AGG=${AXIS_AGG} NTEST=${NTEST} SEED=${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== stage 0: selftest (synthetic fixture, no cluster data) ==="
python -m alignment.contestedness_predictor --selftest \
    || { echo "SELFTEST FAILED -- target logic is broken, not the data"; exit 1; }

CROSS_FLAG=""
if [ "${CROSS}" = "1" ]; then CROSS_FLAG="--cross"; fi
SCORES_FLAG=""
if [ -f "${SCORES}" ]; then
    SCORES_FLAG="--scores ${SCORES} --inject_cond ${INJECT}"
else
    echo "NOTE: ${SCORES} not found -- skipping the help-delta comparison."
    echo "      Topic-CV numbers alone do NOT justify wiring a router."
fi

echo ""
echo "=== stage 1+2: fit, topic CV, cross-dataset transfer, help-delta ==="
python -m alignment.contestedness_predictor \
    --dataset "${DATASET}" ${CROSS_FLAG} \
    --target "${TARGET}" --model "${MODEL}" \
    --ground "${GROUND}" --axis_agg "${AXIS_AGG}" \
    --null_bins "${NULL_BINS}" --n_null "${NNULL}" \
    --n_test_topics "${NTEST}" --alphas "${ALPHAS}" \
    --embed_model "${EMBEDDER}" --emb_cache "${EMB_CACHE}" \
    --seed "${SEED}" ${SCORES_FLAG} --save "${OUT}" \
    || { echo "FIT FAILED (see .err)"; exit 1; }

echo ""
echo "Done. Predictor: ${OUT}"
echo "Read the results in this order:"
echo "  1. corr(z, w_mean) per dataset. Near +1.0 means the stratified null did"
echo "     nothing and z_level is raw magnitude again -- the +0.19 failure."
echo "  2. held-out-TOPIC spearman. Topic-grouped because ATP items inside a"
echo "     cluster are near-paraphrases; a random split would score itself."
echo "  3. ATP<->GOQA transfer. Two different divergence structures (US"
echo "     demographics vs countries); this is the real generalization test."
echo "  4. pred[...] corr vs the help-delta. Must beat +0.19 (w_raw) to matter."
echo "Then wire it as a candidate signal with ONE line at the end of"
echo "evaluation/overton/route_signal.py main():"
echo "  from alignment.contestedness_predictor import report_route_signal; report_route_signal(rows)"
echo "and run with CONTEST_MODEL=${OUT}."
