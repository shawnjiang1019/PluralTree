#!/bin/bash
#SBATCH --job-name=injection_triage
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/injection_triage_%j.out
#SBATCH --error=logs/injection_triage_%j.err

# WHICH questions should we have injected on, and where did merge_v2 lose?
#
#     sbatch jobs/eval/job_injection_triage.sh
#
# Per-question win/loss triage against an INDEPENDENT baseline run. The delta is
# inj - base, so correlating it with the SAME run's baseline is confounded by
# regression to the mean: a question whose baseline scored high did so partly by
# luck. BASE_FROM supplies a different run's baseline, whose noise is
# independent, and the script prints both so the gap (the artifact) is visible.
#
# GPU, not CPU: load_opinionqa embeds ~1500 question texts with MiniLM and
# never pins a device, so it takes CUDA when one is visible. Measured on a
# Narval LOGIN node that pass took 7m15s (47 batches @ 9.27 s/it, throttled to a
# couple of cores); on a GPU it is seconds. Everything after it -- the 75k-record
# graph build, anchor resolution for 60 questions, the arithmetic -- is fast.
#
# Knobs: SCORES, BASE_FROM, CONDS (space-separated), EMB, GROUPS, TOP, TOL.
#   e.g.  CONDS="merge_v2" EMB="" sbatch jobs/eval/job_injection_triage.sh
#
# NOTE: compute nodes have NO outbound network. Pull on a login node first --
# this script deliberately does not try, and the offline flags below keep
# sentence-transformers from stalling on a connect timeout.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
# Without this, load_opinionqa falls back to the GATED SubPOP HF dataset and
# dies on authentication; with it, the raw ATP parse is fully offline.
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

SCORES="${SCORES:-overton_scores_v9.csv}"
BASE_FROM="${BASE_FROM:-overton_scores_v8.csv}"
CONDS="${CONDS:-merge_v2 merge}"
GROUPS="${GROUPS:-graph,response,trace}"
TOP="${TOP:-10}"
TOL="${TOL:-0.027}"        # OvertonBench noise floor from two baseline draws
SEED="${SEED:-42}"         # graph split seed; must match the run being scored
# z_level / driver_sim need a live scout pass. Set EMB="" to skip them.
EMB="${EMB:-embeddings_opinionqa.pt}"

echo "SCORES=${SCORES}  BASE_FROM=${BASE_FROM}  CONDS='${CONDS}'  EMB='${EMB}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for f in "${SCORES}" "${BASE_FROM}"; do
    [ -f "${f}" ] || { echo "MISSING ${f} in $(pwd)"; exit 1; }
done

# An unreadable EMB is a silent 2-feature loss, not a crash -- say so up front.
EMB_ARG=()
if [ -n "${EMB}" ]; then
    if [ -f "${EMB}" ]; then
        EMB_ARG=(--embeddings "${EMB}")
    else
        echo "WARN: ${EMB} not found -- g_z_level/g_driver_sim will be NaN"
    fi
fi

# The pushed code must be the version that scores partial-coverage features;
# the older one drops 5 position features over 4 missing rows.
grep -q "RECOVERED" scripts/analysis/injection_triage.py \
    || echo "WARN: stale injection_triage.py (no partial-coverage recovery) -- git pull on a LOGIN node"

n_fail=0
for C in ${CONDS}; do
    OUT="docs/injection_triage_$(basename "${SCORES}" .csv)_${C}.txt"
    echo ""
    echo "================================================================"
    echo "=== ${C}   -> ${OUT}"
    echo "================================================================"
    python scripts/analysis/injection_triage.py \
        --scores "${SCORES}" --baseline-from "${BASE_FROM}" \
        --condition "${C}" --groups "${GROUPS}" \
        --top "${TOP}" --tol "${TOL}" --seed "${SEED}" \
        "${EMB_ARG[@]}" 2>&1 | tee "${OUT}"
    # tee is in the pipe, so check the PYTHON exit status, not tee's.
    rc="${PIPESTATUS[0]}"
    [ "${rc}" -eq 0 ] || { echo "### ${C} FAILED (rc=${rc})"; n_fail=$((n_fail + 1)); }

    # Same features, same rows, drawn with error bars. Cheap next to the graph
    # load that just happened, and it is the artifact worth putting in a slide.
    python scripts/analysis/plot_feature_routing.py         --scores "${SCORES}" --condition "${C}" --groups "${GROUPS}"         --tol "${TOL}" --seed "${SEED}" "${EMB_ARG[@]}"         || echo "### plot failed for ${C} (triage numbers above still stand)"
done

echo ""
[ "${n_fail}" -eq 0 ] || { echo "${n_fail} condition(s) failed"; exit 1; }
echo "OK -- docs/injection_triage_*.txt and docs/feature_routing_*.png"
