#!/bin/bash
#SBATCH --job-name=embed_cb
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/embed_cb_%j.out
#SBATCH --error=logs/embed_cb_%j.err

# Export CultureBank Poincare embeddings for branch-divergence analysis.
# CultureBank is small (~1.3K entities), so this is quick. Requires the dataset
# + all-MiniLM-L6-v2 model to be pre-cached on the login node (compute nodes are
# offline): the HF_*_OFFLINE flags below force cache-only.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

python scripts/train.py --dataset culturalbench \
    --save_embeddings embeddings_cb.pt --device cuda || { echo "TRAIN FAILED (see .err)"; exit 1; }

if [[ -f embeddings_cb.pt ]]; then
    echo "Done. Wrote embeddings_cb.pt ($(du -h embeddings_cb.pt | cut -f1))"
else
    echo "ERROR: training exited 0 but embeddings_cb.pt was not written"; exit 1
fi
