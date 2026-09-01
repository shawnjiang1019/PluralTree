#!/bin/bash
#SBATCH --job-name=rand_fork_analysis
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/rand_fork_analysis_%j.out
#SBATCH --error=logs/rand_fork_analysis_%j.err

# Analysis for the random-fork control (docs/random_fork_control.md). Chain it
# behind the eval that produces the scores:
#
#   JID=$(... sbatch --parsable jobs/eval/job_overton_eval.sh)
#   sbatch --dependency=afterok:$JID jobs/eval/job_random_fork_analysis.sh
#
# CPU only: coverage arithmetic, a bootstrap, and one matplotlib figure. No
# graph load, no embedder, no GPU -- pass --no_features to keep it that way.
#
# Knobs: SCORES, RESP, BASE_FROM, COND, RAND.

module load python/3.11 gcc arrow/24.0.0
source ~/pluraltree-env/bin/activate
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

SCORES="${SCORES:-overton_scores_v11.csv}"
RESP="${RESP:-overton_responses_v11.jsonl}"
BASE_FROM="${BASE_FROM:-overton_scores_v10.csv}"
COND="${COND:-merge_v2}"
RAND="${RAND:-merge_v2_rand}"

for f in "${SCORES}" "${RESP}"; do
    [ -f "${f}" ] || { echo "MISSING ${f} in $(pwd)"; exit 1; }
done
echo "SCORES=${SCORES}  RESP=${RESP}  ${COND} vs ${RAND}"

# --- 1. the manipulation check, FIRST -----------------------------------
# If the sampled forks are as relevant as the retrieved ones, this is not a
# control and nothing below means anything. Read this before the deltas.
echo ""
echo "=== manipulation check: are the random forks actually irrelevant? ==="
python -u scripts/analysis/check_random_fork.py --responses "${RESP}" \
    || { echo "MANIPULATION CHECK FAILED -- do not interpret the deltas"; exit 1; }

# --- 2. both arms on one axis -------------------------------------------
echo ""
echo "=== mean delta vs baseline, bootstrap CI ==="
python -u scripts/analysis/plot_delta_ci.py --scores "${SCORES}" \
    --conditions "${COND},${RAND}" --out "docs/delta_ci_randfork.png"

# --- 3. per-question triage for the control arm --------------------------
# --no_features: the feature table is a settled null (9 of 13 flip sign between
# arms) and loading the graph for it costs ~7 min for nothing.
echo ""
echo "=== triage: ${RAND} ==="
python -u scripts/analysis/injection_triage.py --scores "${SCORES}" \
    --condition "${RAND}" --baseline-from "${BASE_FROM}" --no_features \
    | tee "docs/injection_triage_randfork.txt"

echo ""
echo "READ IN THIS ORDER:"
echo "  1. rel_real vs rel_rand -- if the gap is small this is not a control"
echo "  2. len_real vs len_rand -- if these differ, prompt length is confounded"
echo "  3. ${RAND} vs ${COND} in the CI plot:"
echo "       overlapping  -> the graph is a randomizer; retrieval contributes"
echo "                       nothing beyond making the drafts differ"
echo "       ${RAND} clearly lower -> fork CONTENT is load-bearing, the first"
echo "                       direct evidence the graph earns its place"
