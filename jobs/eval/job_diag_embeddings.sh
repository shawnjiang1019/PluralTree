#!/bin/bash
#SBATCH --job-name=diag_emb
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/diag_emb_%j.out
#SBATCH --error=logs/diag_emb_%j.err

# What each embedding hands persona drafting, no LLM. Read the "scout personas
# N/60" lines: an embedding with few personas generates mostly baseline answers.

module load python/3.11 gcc arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
FEATS="${FEATS:-feats_opinionqa.pt}"

for e in tc1_ld03 simple_qgp simple_qg; do
    echo "=== ${e} ==="
    python scripts/analysis/retrieval_diag.py --embeddings "embeddings_opinionqa_${e}.pt" \
        --text_feat "${FEATS}" --out "retrieval_diag_${e}.csv" \
        || { echo "DIAGNOSTIC FAILED for ${e}"; exit 1; }
done
