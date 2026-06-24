#!/bin/bash
#SBATCH --job-name=map_reader
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/map_reader_%j.out
#SBATCH --error=logs/map_reader_%j.err

# Phase-1 Map Reader: generate data -> QLoRA SFT -> per-fact ablation eval.
# Assumes the encoder embeddings already exist (EMB). See pluraltree/sft/map_reader.md.
#
# Env vars (optional):
#   EMB     embeddings .pt                (default: embeddings_up_ind.pt)
#   BASE    HF base model                 (default: Qwen/Qwen2.5-7B-Instruct)
#   DATASET wn18rr | culturalbench        (default: wn18rr)
#   SMOKE   1 -> 0.5B model + 200 nodes + 1 epoch (cheap plumbing test)
#
# Examples:
#   sbatch jobs/job_map_reader.sh                                  # full 7B run
#   SMOKE=1 sbatch jobs/job_map_reader.sh                          # dry run
#   EMB=embeddings_both_ind.pt sbatch jobs/job_map_reader.sh

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

# Compute nodes have no internet — model + tokenizer must be in the local HF cache.
# Pre-download on the login node once (same HF_HOME as below):
#   export HF_HOME=~/projects/def-enaskt/shawnj/hf_cache
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
# Honor an HF_HOME passed at submit time (point it at wherever your model cache
# lives); otherwise default to project space. Use a local --base dir to skip the
# cache entirely.
export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree

EMB="${EMB:-embeddings_up.pt}"
DATASET="${DATASET:-wn18rr}"
DATA_DIR="data/wn18rr"
RUN="runs/map_reader"
SFT="data/map_reader_sft.jsonl"
SPLIT="runs/holdout.json"

if [[ ! -f "${EMB}" ]]; then
    echo "ERROR: embeddings '${EMB}' not found. Run step 0 (train.py --save_embeddings) first."
    exit 1
fi

# --- SMOKE vs full config ------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    BASE="${BASE:-Qwen/Qwen2.5-7B-Instruct}"
    GEN_FLAGS="--max_nodes 200"
    TRAIN_FLAGS="--epochs 1 --batch_size 2 --grad_accum 2"
    RUN="runs/map_reader_smoke"
    SFT="data/map_reader_smoke.jsonl"
    echo "=== SMOKE run: ${BASE} ==="
else
    BASE="${BASE:-Qwen/Qwen2.5-7B-Instruct}"
    GEN_FLAGS=""
    TRAIN_FLAGS="--epochs 2 --batch_size 4 --grad_accum 4"
    echo "=== FULL run: ${BASE} ==="
fi

mkdir -p runs logs
echo "EMB=${EMB}  DATASET=${DATASET}  BASE=${BASE}  RUN=${RUN}"

# --- 1. generate Map Reader data (CPU; deterministic) --------------------
echo "### Step 1: generate data"
python scripts/generate_map_reader_sft.py \
    --dataset "${DATASET}" --data_dir "${DATA_DIR}" \
    --embeddings "${EMB}" --split "${SPLIT}" \
    --out "${SFT}" ${GEN_FLAGS} || exit 1

# --- 2. QLoRA SFT (GPU) --------------------------------------------------
echo "### Step 2: train"
python scripts/train_map_reader.py \
    --base "${BASE}" --embeddings "${EMB}" \
    --data "${SFT}" --out "${RUN}" ${TRAIN_FLAGS} || exit 1

# --- 3. per-fact ablation eval (GPU) -------------------------------------
echo "### Step 3: evaluate"
python scripts/eval_map_reader.py \
    --run "${RUN}" --embeddings "${EMB}" --data "${SFT}"

echo "Done."
