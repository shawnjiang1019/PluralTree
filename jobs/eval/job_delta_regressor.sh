#!/bin/bash
#SBATCH --job-name=delta_reg
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/delta_reg_%j.out
#SBATCH --error=logs/delta_reg_%j.err

# Can a REGRESSOR on the per-question injection delta collect the routing gap?
#
# Measured, and the reason this job exists: injection helps contested questions
# (+0.31 coverage) and hurts consensus ones (-0.45); per-question oracle routing
# scores 0.634 vs 0.497 always-baseline. That +0.137 is over 4x the 0.027 noise
# floor and nothing currently collects any of it. Graph divergence W got +0.19
# correlation (the scout selects max-W forks, so the signal is pre-saturated) and
# the model's own <think> routing decision collapsed to 0.072.
#
# The target is CONTINUOUS delta, not sign(delta): magnitude is what lets the
# decision threshold come from the measured cost asymmetry (a wrong inject costs
# ~1.5x a wrong skip) instead of from a 0.5 accuracy break-even. See
# scripts/analysis/delta_regressor.py.
#
#   sbatch jobs/eval/job_delta_regressor.sh
#
# Knobs: VERSIONS (space-separated tags to POOL), FEATURES (headline set),
#        COMPARE (sets scored side by side), COST_RATIO, EMB/FEATS (enable the
#        live scout features g_z_level / g_driver_sim), CONTESTEDNESS (seam),
#        SEED (graph split -- must match the run that produced the responses).
#
# CPU-only: no vLLM, no GPU embedder. The graph load + MiniLM node features are
# the slow part; the fits are closed-form ridge over <=15 columns.
#
# Prereqs: OPINIONQA_DIR populated, overton_scores_vN.csv + overton_responses_
# vN.jsonl in the repo root, MiniLM in HF_HOME. EMB/FEATS are OPTIONAL -- without
# them z_level/driver_sim are dropped as missing (printed, not silently zeroed).

module load python/3.11 gcc arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs docs

VERSIONS="${VERSIONS:-v9 v8 v6 v5 v4}"
FEATURES="${FEATURES:-causal}"      # pre+baseline only: what a deployment can run
COMPARE="${COMPARE:-causal,graph,response,trace,all,length_only}"
COST_RATIO="${COST_RATIO:-1.4516}"  # 0.45 / 0.31, the measured asymmetry
FOLDS="${FOLDS:-0}"                 # 0 = leave-one-QUESTION-out
SEED="${SEED:-42}"
EMB="${EMB:-}"                      # e.g. embeddings_opinionqa.pt
FEATS="${FEATS:-}"                  # e.g. feats_opinionqa.pt
CONTESTEDNESS="${CONTESTEDNESS:-}"  # sibling's text-only predictions (seam)

# STEP 1: the metric itself, on a synthetic fixture with a planted signal. This
# asserts oracle >= best single condition, always-baseline recovers EXACTLY 0.0,
# a planted perfect router EXACTLY 1.0, and that a length-only feature set does
# NOT recover a signal planted as semantic. If this fails, every number in step 2
# and 3 is meaningless, so it gates them.
echo "================================================================"
echo "=== STEP 1: selftest (synthetic fixture, planted routing signal)"
echo "================================================================"
python scripts/analysis/delta_regressor.py --selftest --cost_ratio "${COST_RATIO}" \
    || { echo "SELFTEST FAILED -- the routing metric is miscoded; stopping"; exit 1; }

extra=()
[ -n "${EMB}" ] && extra+=(--embeddings "${EMB}")
[ -n "${FEATS}" ] && extra+=(--text_feat "${FEATS}")
[ -n "${CONTESTEDNESS}" ] && extra+=(--contestedness "${CONTESTEDNESS}")

# STEP 2: PER RUN, unpooled. v4/v5/v6/v8/v9 differ in prompt variant and
# condition set. Pooling is a real choice and the per-run numbers are what say
# whether it is defensible: if a run disagrees with the pool, the pooled headline
# is an average over different tasks, not more data for one task.
echo ""
echo "================================================================"
echo "=== STEP 2: each run alone (is the effect stable across runs?)"
echo "================================================================"
present=()
for V in ${VERSIONS}; do
    S="overton_scores_${V}.csv"
    if [ ! -f "${S}" ]; then
        echo "### SKIP ${V}: ${S} not present"
        continue
    fi
    present+=("${S}")
    echo ""
    echo "--- ${V} ---"
    python scripts/analysis/delta_regressor.py \
        --scores "${S}" --features "${FEATURES}" --compare "${COMPARE}" \
        --cost_ratio "${COST_RATIO}" --folds "${FOLDS}" --seed "${SEED}" \
        "${extra[@]}" --out "docs/delta_regressor_${V}.csv" \
        || { echo "DELTA REGRESSOR FAILED on ${V} (see .err)"; exit 1; }
done

if [ "${#present[@]}" -eq 0 ]; then
    echo "NO SCORE FILES FOUND in $(pwd) for VERSIONS='${VERSIONS}'"; exit 1
fi

# STEP 3: POOLED. n=60 questions per run is small enough that a single run's
# routing number is dominated by fold noise; pooling buys labels at the cost of
# assuming the delta means the same thing under different prompt variants. Every
# row carries its run tag and the per-run breakdown is reprinted inside this run
# so the assumption stays visible.
if [ "${#present[@]}" -gt 1 ]; then
    echo ""
    echo "================================================================"
    echo "=== STEP 3: POOLED across ${#present[@]} runs (a CHOICE -- see per-run rows)"
    echo "================================================================"
    args=()
    for S in "${present[@]}"; do args+=(--scores "${S}"); done
    python scripts/analysis/delta_regressor.py \
        "${args[@]}" --features "${FEATURES}" --compare "${COMPARE}" \
        --cost_ratio "${COST_RATIO}" --folds "${FOLDS}" --seed "${SEED}" \
        "${extra[@]}" --out "docs/delta_regressor_pooled.csv" \
        || { echo "DELTA REGRESSOR FAILED on the pooled run (see .err)"; exit 1; }
fi

echo ""
echo "================================================================"
echo "HOW TO READ THIS"
echo "  headline = fraction of the oracle gap recovered, NOT R^2."
echo "  It is a CROSS-VALIDATED estimate on the SAME 60 OvertonBench questions"
echo "  the router was designed against, so it bounds what is achievable rather"
echo "  than reporting a clean result. A router fit on OvertonBench cannot be"
echo "  reported cleanly on OvertonBench -- a fresh contested-question set is the"
echo "  only honest evaluation."
echo "  If length_only recovers what causal recovers, the result is verbosity."
echo "  If only 'all' recovers, the signal is in the <think> trace of the"
echo "  INJECTED answer: that is a reranker over two generations, not a router."
echo "================================================================"
