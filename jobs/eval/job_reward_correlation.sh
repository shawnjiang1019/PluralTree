#!/bin/bash
#SBATCH --job-name=reward_corr
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/reward_corr_%j.out
#SBATCH --error=logs/reward_corr_%j.err

# STAGE 0 GATE for the GRPO phase: does coverage_reward rank answers to the SAME
# question the way the OvertonBench judge does? GRPO's advantage is computed
# strictly within a group of rollouts sharing one prompt, so if within-question
# concordance is ~0.5 the reward is noise and training optimizes nothing. Run
# this before any training job.
#
#   sbatch jobs/eval/job_reward_correlation.sh
#
# Runs every eval version in VERSIONS (default: v6 and the earlier v5) so the
# gate is not read off a single OvertonBench run. v5 and v6 differ in prompt
# variant and condition set, so agreeing concordance across them is evidence the
# reward tracks the judge rather than one run's quirks.
#
# Knobs: VERSIONS (space-separated tags; RESP/SCORES derived), DEPTHS, EMBEDDER,
#        SEED (graph split, must match the run that produced the responses).
#        RESP/SCORES override the derived names for a one-off, non-vN file.
#
# GPU, not CPU: the script never pins a device -- default_embed_fn constructs
# SentenceTransformer with no `device`, so it takes CUDA when one is visible. On
# a login node this same work runs at ~3 texts/sec (>1h); on a GPU it is seconds.
#
# Prereqs: mpnet reward embedder + MiniLM (graph load) pre-downloaded into
# HF_HOME; OPINIONQA_DIR populated; RESP/SCORES present in the repo root.

# Same modules as the working interactive run, plus cuda for the GPU embedder.
# No opencv: this path is graph load + sentence-transformers only.
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

VERSIONS="${VERSIONS:-v6 v5}"
DEPTHS="${DEPTHS:-0,30,60,90,120,150}"
EMBEDDER="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"
SEED="${SEED:-42}"
echo "VERSIONS='${VERSIONS}' DEPTHS=${DEPTHS} SEED=${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run () {   # run <resp> <scores> <tag> <extra-args...>
    local resp="$1" scores="$2" tag="$3"; shift 3
    echo ""
    echo "================================================================"
    echo "=== ${tag}"
    echo "================================================================"
    python scripts/analysis/reward_eval_correlation.py \
        --responses "${resp}" --scores "${scores}" \
        --min_depth_words "${DEPTHS}" --embedder "${EMBEDDER}" --seed "${SEED}" \
        "$@" || { echo "CORRELATION CHECK FAILED (${tag}) -- see .err"; exit 1; }
}

n_ok=0
for V in ${VERSIONS}; do
    resp="${RESP:-overton_responses_${V}.jsonl}"
    scores="${SCORES:-overton_scores_${V}.csv}"
    if [ ! -f "${resp}" ] || [ ! -f "${scores}" ]; then
        echo ""
        echo "### SKIP ${V}: missing ${resp} or ${scores}"
        continue
    fi

    # ALL conditions: `route` collapsed to 0.072, so the reward only has to
    # notice a 0.4 gap. Easy, and NOT the regime GRPO runs in.
    run "${resp}" "${scores}" "[${V}] all conditions (includes route -- easy case)" \
        --out "docs/reward_eval_correlation_${V}_all.csv"

    # NEAR-TIES: every rollout in a GRPO group comes from the SAME policy and
    # they resemble each other far more than baseline resembles route. THE GATE.
    run "${resp}" "${scores}" "[${V}] excluding route (near-ties -- THE GATE, needs >=0.60)" \
        --exclude route --out "docs/reward_eval_correlation_${V}_noroute.csv"
    n_ok=$((n_ok + 1))
done

if [ "${n_ok}" -eq 0 ]; then
    echo "NO VERSIONS SCORED -- check VERSIONS / file names in \$(pwd)"; exit 1
fi

echo ""
echo "Done (${n_ok} version(s)). Gate: within-question concordance in each"
echo "'excluding route' block must clear 0.60, and the versions should AGREE."
echo "Below ~0.55 the reward cannot order same-prompt rollouts -- fix the reward"
echo "before launching jobs/train/job_grpo_align.sh."
