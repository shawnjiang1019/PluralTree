#!/bin/bash
#SBATCH --job-name=wn18rr_ind
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/wn18rr_ind_%j.out
#SBATCH --error=logs/wn18rr_ind_%j.err

# Inductive WN18RR run: hold out a fraction of LEAF entities' links from training
# and evaluate link prediction on them (entities embedded from text + structure
# alone, never trained on). Prints an INDUCTIVE | line next to RESULT | / STRUCT |.
#
# Env vars (optional):
#   RUN_NAME   up|updown|lat|both  -> message-passing flow (default: up)
#   HOLDOUT    held-out leaf fraction (default: 0.1)
#
# Examples:
#   sbatch jobs/kgc/job_inductive.sh                       # up, 10% holdout
#   RUN_NAME=lat HOLDOUT=0.2 sbatch jobs/kgc/job_inductive.sh

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/envs/pluraltree/bin/activate

# Compute nodes have no internet — use the local HuggingFace cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree

# Map RUN_NAME -> message-passing flags (default: bottom-up only).
case "${RUN_NAME:-up}" in
    up)     FLOW_FLAGS="" ;;
    updown) FLOW_FLAGS="--bidirectional" ;;
    lat)    FLOW_FLAGS="--lateral" ;;
    both)   FLOW_FLAGS="--bidirectional --lateral" ;;
    *) echo "Unknown RUN_NAME='${RUN_NAME}' (use up|updown|lat|both)"; exit 1 ;;
esac
HOLDOUT="${HOLDOUT:-0.1}"
TAG="${RUN_NAME:-up}_ind"
echo "RUN_NAME=${RUN_NAME:-up}  FLOW_FLAGS=${FLOW_FLAGS:-<none>}  HOLDOUT=${HOLDOUT}"

python scripts/train/train.py --dataset wn18rr --device cuda \
    --n_epochs 100 --d_hidden 128 \
    --warmup1 400 --warmup2 1600 --batch_size 1024 --lr 3e-3 \
    --embed_model all-mpnet-base-v2 \
    --inductive_holdout "${HOLDOUT}" \
    --save_embeddings "embeddings_${TAG}.pt" \
    --metrics_csv "metrics_${TAG}.csv" \
    --no_gki ${FLOW_FLAGS}
