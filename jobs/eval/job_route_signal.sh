#!/bin/bash
#SBATCH --job-name=route_signal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/route_signal_%j.out
#SBATCH --error=logs/route_signal_%j.err

# Decide whether ANY inference-time graph signal predicts when scout injection
# helps OvertonBench coverage — before spending a generation run on a router.
# CPU-only (no vLLM): runs the scout over the 60 questions, correlates candidate
# signals (raw W, calibrated z, relevance, driver-question match) with the known
# v5 (scout - baseline) coverage deltas. See evaluation/overton/route_signal.py.
#
# Needs: embeddings_opinionqa.pt (embed job) + overton_scores_v5.csv (eval job).

module load python/3.11 gcc arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
# raw-ATP OpinionQA (offline; no SubPOP gate) — see data/loaders/opinionqa.py
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

EMB="${EMB:-embeddings_opinionqa.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
SCORES="${SCORES:-overton_scores_v5.csv}"
TAU="${TAU:-0.25}"
echo "EMB=${EMB}  SCORES=${SCORES}  TAU=${TAU}"

python -m evaluation.overton.route_signal \
    --embeddings "${EMB}" --text_feat "${FEATS}" --scores "${SCORES}" \
    --dataset opinionqa --curvature 0.5 --tau "${TAU}" \
    || { echo "ROUTE SIGNAL FAILED (see .err)"; exit 1; }
