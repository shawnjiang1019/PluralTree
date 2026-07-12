#!/bin/bash
#SBATCH --job-name=grail_ind
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/grail_%j.out
#SBATCH --error=logs/grail_%j.err

# Inductive WN18RR on the standard GraIL splits (v1-v4), both protocols:
#   RESULT | ... -> filtered MRR / Hits@{1,3,10}   (modern, rank vs all)
#   GRAIL  | ... -> Hits@10 / MRR vs 50 negs + AUC-PR  (original GraIL 2020)
#
# First download the data on the LOGIN node: python scripts/fetch/get_grail_wn18rr.py
#
# Env vars (optional):
#   VERSION  v1|v2|v3|v4|all  (default: all)
#   FLOW     up|updown|lat|both  (default: up)

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/envs/pluraltree/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree

case "${FLOW:-up}" in
    up)     FLOW_FLAGS="" ;;
    updown) FLOW_FLAGS="--bidirectional" ;;
    lat)    FLOW_FLAGS="--lateral" ;;
    both)   FLOW_FLAGS="--bidirectional --lateral" ;;
    *) echo "Unknown FLOW='${FLOW}' (use up|updown|lat|both)"; exit 1 ;;
esac
VERSION="${VERSION:-all}"
echo "VERSION=${VERSION}  FLOW=${FLOW:-up}  FLOW_FLAGS=${FLOW_FLAGS:-<none>}"

python scripts/train/train_inductive.py --version "${VERSION}" --device cuda \
    --d_hidden 128 --n_epochs 100 --batch_size 1024 --lr 3e-3 \
    --embed_model all-mpnet-base-v2 ${FLOW_FLAGS}
