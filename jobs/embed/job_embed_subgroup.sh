#!/bin/bash
#SBATCH --job-name=embed_subgroup
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/embed_subgroup_%j.out
#SBATCH --error=logs/embed_subgroup_%j.err

# Simple subgroup encoder (scripts/train/train_subgroup.py), the comparison arm
# for the Tree-GRU. Same graph and node ids, so EMB works with retrieval as is.
#
#   INPUTS=qgp SPLIT=axis EMB=embeddings_opinionqa_simple_qgp.pt sbatch jobs/embed/job_embed_subgroup.sh
#   INPUTS=qg  SPLIT=item EMB=embeddings_opinionqa_simple_qg.pt  sbatch jobs/embed/job_embed_subgroup.sh

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs

INPUTS="${INPUTS:-qgp}"
SPLIT="${SPLIT:-axis}"
FEATS="${FEATS:-feats_opinionqa.pt}"
EMB="${EMB:-embeddings_opinionqa_simple_${INPUTS}.pt}"
STEPS="${STEPS:-6000}"
EXTRA="${EXTRA:-}"

python scripts/train/train_subgroup.py --inputs "${INPUTS}" --split_by "${SPLIT}" \
    --text_feat "${FEATS}" --save_embeddings "${EMB}" --steps "${STEPS}" \
    --device cuda ${EXTRA} \
    || { echo "TRAIN FAILED (see .err)"; exit 1; }
