#!/bin/bash
#SBATCH --job-name=wn18rr
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/wn18rr_%j.out
#SBATCH --error=logs/wn18rr_%j.err

# WN18RR run. Knowledge injection is OFF (--no_gki) — we are testing the
# structural message-passing flows, not GKI. The extra flows are selected by the
# RUN_NAME env var (mapped to flags below) so one script drives the whole sweep:
#
#   RUN_NAME=up      -> (none)                      bottom-up only
#   RUN_NAME=updown  -> --bidirectional             + top-down
#   RUN_NAME=lat     -> --lateral                   + same-depth siblings
#   RUN_NAME=both    -> --bidirectional --lateral   + both
#
# RUN_NAME defaults to "up". Submit all four with jobs/submit_wn18rr_sweep.sh.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/envs/pluraltree/bin/activate

# Compute nodes have no internet — force HuggingFace to use the local cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Unbuffer stdout so the .out updates live and survives a time-limit kill.
export PYTHONUNBUFFERED=1
# Reduce CUDA fragmentation on the large 40K-node encode.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree

# Map RUN_NAME -> message-passing flags (default: bottom-up only).
case "${RUN_NAME:-up}" in
    up)     FLOW_FLAGS="" ;;
    updown) FLOW_FLAGS="--bidirectional" ;;
    lat)    FLOW_FLAGS="--lateral" ;;
    both)   FLOW_FLAGS="--bidirectional --lateral" ;;
    *) echo "Unknown RUN_NAME='${RUN_NAME}' (use up|updown|lat|both)"; exit 1 ;;
esac
echo "RUN_NAME=${RUN_NAME:-up}  FLOW_FLAGS=${FLOW_FLAGS:-<none>}"

# Faster *convergence* config: encode every step (default), no gradient
# checkpointing (GKI is off, so activations fit -> no recompute), and a large
# batch so each expensive encode trains on many triples. If this OOMs, add
# --checkpoint back (and/or drop to --d_hidden 64).
python scripts/train.py --dataset wn18rr --device cuda \
    --n_epochs 100 --d_hidden 128 \
    --warmup1 400 --warmup2 1600 --batch_size 1024 --lr 3e-3 \
    --embed_model all-mpnet-base-v2 \
    --save_embeddings "embeddings_${RUN_NAME:-up}.pt" \
    --metrics_csv "metrics_${RUN_NAME:-up}.csv" \
    --no_gki ${FLOW_FLAGS}
