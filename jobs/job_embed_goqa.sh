#!/bin/bash
#SBATCH --job-name=embed_goqa
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/embed_goqa_%j.out
#SBATCH --error=logs/embed_goqa_%j.err

# Export GlobalOpinionQA Poincare embeddings for branch-divergence analysis.
# This is the dataset where divergence SHOULD exist (per-country opinion clouds),
# so it's the one that can actually validate the Wasserstein metric.
#
# Anti-collapse defaults (we saw CulturalBench saturate to the boundary):
#   CURV   curvature c (lower = more interior room)  default 0.5
#   LSTR   structure-fidelity loss weight            default 0.1
# Both overridable: CURV=1.0 LSTR=0.0 sbatch jobs/job_embed_goqa.sh
#
# Requires the dataset cached in HF_HOME (compute nodes are offline). Pre-download
# on the login node once, into the SAME HF_HOME:
#   export HF_HOME=~/projects/def-enaskt/shawnj/hf_cache
#   python -c "from datasets import load_dataset; load_dataset('Anthropic/llm_global_opinions')"

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

CURV="${CURV:-0.5}"
LSTR="${LSTR:-0.1}"
LDIV="${LDIV:-0.1}"        # sibling-separation floor (diversity-as-objective)
DIVM="${DIVM:-1.0}"        # min geodesic distance between siblings
EMB="${EMB:-embeddings_goqa.pt}"
echo "CURV=${CURV}  LSTR=${LSTR}  LDIV=${LDIV}  DIVM=${DIVM}  EMB=${EMB}"

python scripts/train.py --dataset globalopinionqa \
    --curvature "${CURV}" --lambda_struct "${LSTR}" \
    --lambda_div "${LDIV}" --div_margin "${DIVM}" \
    --save_embeddings "${EMB}" --device cuda \
    || { echo "TRAIN FAILED (see .err)"; exit 1; }

if [[ -f "${EMB}" ]]; then
    echo "Done. Wrote ${EMB} ($(du -h ${EMB} | cut -f1))"
else
    echo "ERROR: training exited 0 but ${EMB} was not written"; exit 1
fi
