#!/bin/bash
#SBATCH --job-name=mt_router
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/mt_router_%j.out
#SBATCH --error=logs/mt_router_%j.err

# Does auxiliary-task pretraining on FREE contestedness labels buy a better
# injection router than the ~60 expensive delta labels can buy alone?
#
#   sbatch jobs/train/job_multitask_router.sh
#
# THE LABEL ASYMMETRY, which is the whole reason this job exists:
#   auxiliary  ~4,000 contestedness labels (z_level), FREE -- computed from the
#              survey distributions themselves (ATP ~1,492 + GOQA ~2,500). No
#              generation, no judge, one embedding pass.
#   target     ~60 delta labels per scored run, each costing TWO generations
#              plus TWO judge passes.
# Plentiful auxiliary + scarce target is the textbook case for auxiliary
# pretraining, and the two targets share their causal axis: injection helps
# contested questions (+0.31) and hurts consensus ones (-0.45). Oracle routing
# scores 0.634 vs 0.497 always-baseline; that +0.137 is over 4x the 0.027 noise
# floor and nothing collects it. Graph divergence W got corr +0.19 (the scout
# SELECTS max-W forks, so the signal is pre-saturated) and the model's own
# <think> routing decision collapsed to 0.072.
#
# THE HEADLINE IS A COMPARISON, NOT A NUMBER: does `pretrain-ft` recover more of
# the oracle gap than `delta-only`? If it does not, the multi-task premise is
# wrong for this pair of tasks and the PR says so. Do not tune until it flips.
#
# Stages:
#   0  --selftest, 12 synthetic fixture draws (seconds, no cluster data). Gates
#      everything else: it asserts the transfer machinery works when the tasks
#      share a latent factor, that SHUFFLED auxiliary labels do NOT help, and
#      that a random split inflates the metric relative to leave-one-topic-out.
#   1  all three modes side by side + the length-only control + the split
#      contrast, on the real labels.
#   2  THE CONTROL RUN: --shuffle_aux. delta-only is NESTED inside pretrain-ft
#      at transfer strength s=0, so pretrain-ft can only lose to it by selection
#      noise. The comparison that actually separates a transferred signal from a
#      flattering extra hyperparameter is pretrain-ft vs pretrain-ft-shuffled.
#      Stage 1 without stage 2 is not evidence.
#
# Knobs: VERSIONS (space-separated score tags; the FIRST present one is the
#        headline, the rest are scored separately -- pooling prompt variants is
#        a choice this job does not make for you), MODE (headline mode),
#        COMPARE, AUX (auxiliary graphs to pool), NCOMP (PCA trunk width),
#        ALPHAS / SHRINKS / JOINT_LAMS / JOINT_TIES, NTEST (topics held out per
#        fold), TOPICS (k-means pseudo-topics over the eval questions),
#        COST_RATIO, EMBEDDER, SEED, INJECT, SPLIT.
#
# SEED must match the run that produced the scores: opinionqa's topic clustering
# (and therefore node ids and the auxiliary topic folds) is seeded.
#
# GPU, not CPU: the script never pins a device, so SentenceTransformer takes
# CUDA when one is visible. ~4k auxiliary questions through mpnet is seconds on
# a GPU and ~20 min on a login node. Everything after the embedding pass is
# closed-form numpy over a <=64-dim trunk.
#
# Prereqs: mpnet + MiniLM and the overtonbench dataset pre-downloaded into
# HF_HOME; OPINIONQA_DIR populated; overton_scores_vN.csv (and, for the
# length-only control to use ANSWER length rather than question length,
# overton_responses_vN.jsonl) in the repo root.
# No scout is run here, so the opinionqa-0.25 / GOQA-0.1 TAU split does not
# apply -- this job never touches the retrieval gate.

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

VERSIONS="${VERSIONS:-v6 v5 v9 v8 v4}"
MODE="${MODE:-pretrain-ft}"          # the mode under test
COMPARE="${COMPARE:-delta-only,pretrain-ft,joint}"
AUX="${AUX:-opinionqa,globalopinionqa}"   # pooled -> the full ~4,000 free labels
NCOMP="${NCOMP:-64}"                 # PCA trunk width. 768 raw dims against ~60
                                     # target labels identifies nothing; keep small
ALPHAS="${ALPHAS:-0.3,1,3,10,30,100}"
SHRINKS="${SHRINKS:-0,0.25,0.5,0.75,1.0}"   # transfer strength; 0 == delta-only
JOINT_LAMS="${JOINT_LAMS:-0.3,0.6,0.9}"     # auxiliary loss weight
JOINT_TIES="${JOINT_TIES:-1,10}"            # task-specific / shared penalty
NTEST="${NTEST:-2}"                  # topics held out per fold
TOPICS="${TOPICS:-10}"               # k-means pseudo-topics over the 60 eval
                                     # questions: they have no topic layer and
                                     # near-paraphrases must not straddle a split
COST_RATIO="${COST_RATIO:-1.4516}"   # 0.45 / 0.31, the measured asymmetry
EMBEDDER="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"
                                     # deliberately NOT the scout's MiniLM: the
                                     # router must not be a function of the
                                     # retrieval it gates
INJECT="${INJECT:-scout}"
SPLIT="${SPLIT:-topic}"              # NEVER report `random`; it is printed as a
                                     # contrast only
SEED="${SEED:-42}"
EMB_CACHE="${EMB_CACHE:-docs/mtr_emb}"
echo "MODE=${MODE} AUX=${AUX} NCOMP=${NCOMP} NTEST=${NTEST} TOPICS=${TOPICS}"
echo "SPLIT=${SPLIT} COST_RATIO=${COST_RATIO} SEED=${SEED} EMBEDDER=${EMBEDDER}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "================================================================"
echo "=== STAGE 0: selftest (synthetic fixtures, no cluster data)"
echo "================================================================"
# Asserts on the MEAN over 12 draws and on paired per-draw wins: 60 questions in
# 5 folds is noise-dominated, so a single draw decides nothing.
python -m alignment.multitask_router --selftest \
    || { echo "SELFTEST FAILED -- the transfer machinery or the metric is broken,"
         echo "not the data. Every number below would be meaningless; stopping."; exit 1; }

run () {   # run <scores> <tag> <extra-args...>
    local scores="$1" tag="$2"; shift 2
    echo ""
    echo "================================================================"
    echo "=== ${tag}"
    echo "================================================================"
    python -m alignment.multitask_router \
        --scores "${scores}" --inject_conds "${INJECT}" \
        --mode "${MODE}" --compare "${COMPARE}" \
        --aux_datasets "${AUX}" --n_comp "${NCOMP}" \
        --alphas "${ALPHAS}" --shrinks "${SHRINKS}" \
        --joint_lams "${JOINT_LAMS}" --joint_ties "${JOINT_TIES}" \
        --n_test_topics "${NTEST}" --target_topics "${TOPICS}" \
        --split "${SPLIT}" --cost_ratio "${COST_RATIO}" \
        --embed_model "${EMBEDDER}" --emb_cache "${EMB_CACHE}" \
        --seed "${SEED}" "$@" \
        || { echo "MULTITASK ROUTER FAILED (${tag}) -- see .err"; exit 1; }
}

HEAD=""
n_ok=0
for V in ${VERSIONS}; do
    S="overton_scores_${V}.csv"
    if [ ! -f "${S}" ]; then
        echo ""
        echo "### SKIP ${V}: ${S} not present"
        continue
    fi
    [ -z "${HEAD}" ] && HEAD="${V}"
    n_ok=$((n_ok + 1))

    # STAGE 1: the three modes, the length-only control, the split contrast.
    run "${S}" "[${V}] STAGE 1: delta-only vs pretrain-ft vs joint" \
        --out "docs/mtr_${V}.csv" \
        --save "mtr_${MODE}_${V}.npz"

    # STAGE 2: THE CONTROL. Same trunk, same grid (s=0 still available), only
    # the auxiliary SIGNAL destroyed. If stage 1's pretrain-ft gain survives
    # here too, it was the extra hyperparameter, not transfer.
    run "${S}" "[${V}] STAGE 2 CONTROL: --shuffle_aux (auxiliary labels permuted)" \
        --shuffle_aux --out "docs/mtr_${V}_shufaux.csv"
done

if [ "${n_ok}" -eq 0 ]; then
    echo "NO SCORE FILES FOUND in $(pwd) for VERSIONS='${VERSIONS}'"; exit 1
fi

echo ""
echo "================================================================"
echo "HOW TO READ THIS  (headline run: ${HEAD})"
echo "================================================================"
echo "1. aux head held-out-TOPIC spearman. If it is ~0 the auxiliary head"
echo "   learned nothing from the 4,000 free labels and there is nothing to"
echo "   transfer -- stop, and fix contestedness_predictor first."
echo "2. delta-only vs pretrain-ft, oracle gap recovered. THE COMPARISON."
echo "   Ceiling is the 0.634 oracle; 0.497 is always-baseline."
echo "3. STAGE 2 shuffled-aux. delta-only is NESTED inside pretrain-ft at s=0,"
echo "   so pretrain-ft beating it is weak evidence on its own. The claim is"
echo "   supported only if stage 1 pretrain-ft clearly beats stage 2's."
echo "4. length_only. Length has already reversed one conclusion here; if it"
echo "   recovers what pretrain-ft recovers, the result is verbosity."
echo "5. the [RANDOM split] row. Printed as a contrast and NEVER reportable:"
echo "   contestedness_predictor measured random k-fold at 1.000 against"
echo "   leave-one-topic-out at 0.007 on a label that IS the topic."
echo ""
echo "EVERY delta-head number is a CV ESTIMATE of an upper bound. The labels ARE"
echo "OvertonBench, so a router fit on them cannot be reported cleanly on them;"
echo "a fresh contested-question set is the only clean evaluation."
echo ""
echo "If pretrain-ft does NOT beat delta-only, that is the result. The free"
echo "contestedness labels do not transfer to the delta, the multi-task premise"
echo "is wrong for this pair, and the PR should say so rather than tune."
echo ""
echo "Router saved to mtr_${MODE}_<version>.npz; load it with"
echo "  alignment.multitask_router.MultiTaskRouter.load(path).route(texts)"
echo "which applies the cost-asymmetric threshold (break-even p = 0.592, not"
echo "0.5) rather than 'inject if you expect any gain'."
