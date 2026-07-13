#!/bin/bash
#SBATCH --job-name=overton_analysis
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/overton_analysis_%j.out
#SBATCH --error=logs/overton_analysis_%j.err

# Post-hoc analysis of an OvertonBench run (CPU-only, stdlib-only): paired
# per-question stats + the triage-retention x win/loss mechanism table.
# Chain it after the eval job:
#   sbatch --dependency=afterok:<EVAL_JOB_ID> jobs/eval/job_overton_analysis.sh
# Knobs: RESP (responses JSONL), SCORES (scores CSV) — must match the eval job's
# OUT/SCORES values.

module load python/3.11
source ~/pluraltree-env/bin/activate
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

RESP="${RESP:-overton_responses_v4.jsonl}"
SCORES="${SCORES:-overton_scores_v4.csv}"
echo "RESP=${RESP}  SCORES=${SCORES}"

python -m evaluation.overton.analyze_overtonbench "${RESP}" "${SCORES}" \
    || { echo "ANALYSIS FAILED (see .err)"; exit 1; }
