#!/bin/bash
#SBATCH --job-name=grpo_align
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/grpo_align_%j.out
#SBATCH --error=logs/grpo_align_%j.err

# GRPO alignment phase (docs/grpo_alignment.txt): train a LoRA policy over a
# TRAINABLE bf16 base to produce pluralistic answers, rewarded by graph-grounded
# coverage of real subgroup positions (NOT the OvertonBench judge -- that stays
# held-out eval). The scout retrieval is baked into the prompts at build time;
# the graph encoder / scout are frozen here.
#
# Smoke:   DRY=1 sbatch jobs/train/job_grpo_align.sh          (CPU-ish; no trl/GPU)
# Full:    sbatch jobs/train/job_grpo_align.sh
# Knobs:   BASE, GROUP, LR, KL, MAXSTEPS, PROMPTS, EMB, FEATS, OUT
#
# Prereqs (login node, once): trl + peft installed in pluraltree-env; BASE model
# weights + the mpnet reward embedder pre-downloaded into HF_HOME; embeddings_*.pt
# from the embed job. Verify the reward first: `DRY=1 sbatch ...`.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
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
BASE="${BASE:-Qwen/Qwen2.5-7B-Instruct}"
GROUP="${GROUP:-8}"
LR="${LR:-1e-6}"
KL="${KL:-0.04}"
MAXSTEPS="${MAXSTEPS:-0}"          # 0 = run to n_epochs
PROMPTS="${PROMPTS:-0}"            # 0 = all usable graph questions
OUT="${OUT:-grpo_lora}"
echo "BASE=${BASE} GROUP=${GROUP} LR=${LR} KL=${KL} MAXSTEPS=${MAXSTEPS} PROMPTS=${PROMPTS} EMB=${EMB}"

if [ "${DRY:-0}" = "1" ]; then
    echo "=== DRY RUN (reward + advantage on real prompts; no trl/GPU) ==="
    python -m alignment.train_grpo \
        --embeddings "${EMB}" --text_feat "${FEATS}" --dataset opinionqa \
        --curvature 0.5 --prompts_max 20 --dry_run --stub_embed \
        || { echo "DRY RUN FAILED (see .err)"; exit 1; }
    exit 0
fi

python -m alignment.train_grpo \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset opinionqa \
    --curvature 0.5 \
    --base_model "${BASE}" --group_size "${GROUP}" --lr "${LR}" --kl_coef "${KL}" \
    --max_steps "${MAXSTEPS}" --prompts_max "${PROMPTS}" --save_dir "${OUT}" \
    || { echo "GRPO TRAIN FAILED (see .err)"; exit 1; }

echo "Done. LoRA adapter: ${OUT}"
