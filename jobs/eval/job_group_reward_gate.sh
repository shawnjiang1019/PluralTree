#!/bin/bash
#SBATCH --job-name=group_gate
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/group_gate_%j.out
#SBATCH --error=logs/group_gate_%j.err

# STAGE 1 GATE for group-diversity GRPO: does alignment/group_reward.py rank the
# rollouts of one question the way the OvertonBench judge does? Exit 2 on FAIL, so
# a training job chained with --dependency=afterok cannot start on a reward that
# does not track the judge. No generation, no judge calls: it reads files.
#
# Inputs come from stage 0 (7B generation, then the 72B judge TWICE at different
# seeds -- the second pass is the judge-vs-itself ceiling the bar is set against):
#
#   GEN=$(OUT=overton_responses_g8.jsonl SCORES=overton_scores_g8.csv NROLL=8 \
#         CONDS=baseline,scout SKIPJUDGE=1 sbatch --parsable jobs/eval/job_scale_7b.sh)
#   J0=$(RESP=overton_responses_g8.jsonl SCORES=overton_scores_g8.csv \
#        ROLLOUTS=overton_scores_g8_rollouts.csv SEED=0 \
#        sbatch --parsable --dependency=afterok:$GEN jobs/eval/job_judge_only.sh)
#   J1=$(RESP=overton_responses_g8.jsonl SCORES=overton_scores_g8_s1.csv \
#        ROLLOUTS=overton_scores_g8_s1_rollouts.csv SEED=1 \
#        sbatch --parsable --dependency=afterok:$GEN jobs/eval/job_judge_only.sh)
#   sbatch --dependency=afterok:$J0:$J1 jobs/eval/job_group_reward_gate.sh
#
# GPU only for the mpnet embedder (same reason as job_reward_correlation.sh).
#
# Knobs: RESP, ROLL_A, ROLL_B, COND, REF_RESP, REF_COND, LAMBDAS, DEPTHS, OUT.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

RESP="${RESP:-overton_responses_g8.jsonl}"
ROLL_A="${ROLL_A:-overton_scores_g8_rollouts.csv}"
ROLL_B="${ROLL_B:-overton_scores_g8_s1_rollouts.csv}"
COND="${COND:-baseline}"
REF_RESP="${REF_RESP:-}"         # frozen-base samples; enables mode "reference"
REF_COND="${REF_COND:-baseline}"
LAMBDAS="${LAMBDAS:-0,0.5,1,2}"
DEPTHS="${DEPTHS:-0,30}"
OUT="${OUT:-docs/group_reward_gate_${COND}.csv}"

for f in "${RESP}" "${ROLL_A}" "${ROLL_B}"; do
    [ -f "${f}" ] || { echo "MISSING ${f} in $(pwd)"; exit 1; }
done
REF_ARG=""
[ -n "${REF_RESP}" ] && REF_ARG="--ref_responses ${REF_RESP} --ref_condition ${REF_COND}"

set +e
python -u scripts/analysis/group_reward_gate.py \
    --responses "${RESP}" --rollouts "${ROLL_A}" --rollouts_b "${ROLL_B}" \
    --condition "${COND}" ${REF_ARG} \
    --lambdas "${LAMBDAS}" --depths "${DEPTHS}" --out "${OUT}" --gate
rc=$?
set -e
if [ "${rc}" -eq 2 ]; then
    echo ""
    echo "GATE NOT MET (exit 2) -- a RESULT, not a crash. Do not train on this reward."
    exit 2
elif [ "${rc}" -ne 0 ]; then
    echo "GATE CRASHED (rc=${rc}) -- see .err"; exit 1
fi
echo "GATE PASSED -> train with the chosen LAMBDA/DEPTH:"
echo "  REWARD=group INJECT=0 LAMBDA=<chosen> DEPTH=<chosen> MAXSTEPS=20 sbatch jobs/train/job_grpo_align.sh"
