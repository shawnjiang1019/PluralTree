#!/bin/bash
# Submit the WN18RR message-passing sweep: four variants of the structural
# encoder (knowledge injection off), each as its own SLURM job with a distinct
# name and log file. Run from the repo root on the Narval login node:
#
#     bash jobs/submit_wn18rr_sweep.sh
#
# Compare afterwards with:
#   grep -h '^RESULT' logs/wn18rr_*_*.out   # link prediction (MRR / Hits)
#   grep -h '^STRUCT' logs/wn18rr_*_*.out   # geometry (subtree_ap, ancestor_auc, ...)

set -euo pipefail
mkdir -p logs

submit () {
    local name="$1"
    sbatch \
        --job-name="wn18rr_${name}" \
        --output="logs/wn18rr_${name}_%j.out" \
        --error="logs/wn18rr_${name}_%j.err" \
        --export=ALL,RUN_NAME="${name}" \
        jobs/kgc/job_wn18rr.sh
}

# RUN_NAME values are mapped to flags inside job_wn18rr.sh (no spaces exported).
submit up
submit updown
submit lat
submit both

echo "Submitted 4 jobs. Track with: squeue -u \$USER"
