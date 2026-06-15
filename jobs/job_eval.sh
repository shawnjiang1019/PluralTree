#!/bin/bash
#SBATCH --job-name=pt_eval
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Frozen baselines for a dataset, both axes, in one job:
#   - link-prediction floor  (frozen_baseline.py  -> MRR / Hits)
#   - geometry floor         (eval_structure.py   -> subtree_ap / ancestor_auc / ...)
#
# The heavy step is the sentence-transformer encoding of all entity texts, which
# wants a GPU — hence a job rather than the login node.
#
# Env vars (all optional):
#   DATASET      dataset to evaluate            (default: wn18rr)
#   EMBED_MODEL  sentence-transformer           (default: all-mpnet-base-v2)
#   EMBEDDINGS   path to a trained (N,d) tensor  (default: none) -> if set, eval_structure
#                prints floor vs trained side by side
#
# Examples:
#   sbatch jobs/job_eval.sh
#   EMBEDDINGS=embeddings_up.pt sbatch jobs/job_eval.sh

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/envs/pluraltree/bin/activate

# Compute nodes have no internet — use the local HuggingFace cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree

DATASET="${DATASET:-wn18rr}"
EMBED_MODEL="${EMBED_MODEL:-all-mpnet-base-v2}"
echo "DATASET=${DATASET}  EMBED_MODEL=${EMBED_MODEL}  EMBEDDINGS=${EMBEDDINGS:-<none>}"

echo "=================== link-prediction floor ==================="
python scripts/frozen_baseline.py \
    --dataset "${DATASET}" --embed_model "${EMBED_MODEL}" --device cuda

echo "=================== geometry floor ==================="
if [ -n "${EMBEDDINGS:-}" ]; then
    python scripts/eval_structure.py \
        --dataset "${DATASET}" --embed_model "${EMBED_MODEL}" \
        --embeddings "${EMBEDDINGS}"
else
    python scripts/eval_structure.py \
        --dataset "${DATASET}" --embed_model "${EMBED_MODEL}"
fi
