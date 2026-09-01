#!/bin/bash
#SBATCH --job-name=cad_eval
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/cad_eval_%j.out
#SBATCH --error=logs/cad_eval_%j.err

# Context-aware decoding on OvertonBench (docs/cad_experiment.md).
#
#   Smoke:  MAXQ=3 ARMS="base7b,ctx0,cad-0.5" sbatch jobs/eval/job_cad.sh
#   Tune:   HALF=a sbatch jobs/eval/job_cad.sh          # sweep alpha here
#   Report: HALF=b ARMS="base7b,ctx0,cad<best>" sbatch jobs/eval/job_cad.sh
#
# TWO STAGES: generate, then judge. The judge needs a served model, so it is NOT
# run here -- generation holds the GPU with a local 7B and the two cannot share
# it. Judge afterwards with jobs/eval/job_overton_eval.sh's stage 2, or directly:
#
#   python -m evaluation.overton.judge_overtonbench --score cad_responses.jsonl \
#       --max_users 20 --out cad_scores.csv --base_url ... --model ...
#
# WHY 7B AND NOT THE 72B AWQ: CAD contrasts two forward passes, so it needs
# LOGITS. The vLLM endpoint exposes none. Consequence, and it is not optional:
# these numbers are NOT comparable to v9/v10. `base7b` is the only valid
# reference, which is why it is in the default ARMS.
#
# Knobs: MODEL, ARMS, HALF, MAXQ, TEMP, TOPP, MAXNEW, EMB, FEATS, DATASET, TAU.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

# Prefer a local clone: huggingface_hub 1.19.0+computecanada breaks
# snapshot_download, so weights were pulled with `git clone`. Passing a DIRECTORY
# skips the hub resolver and the offline flags stop mattering.
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/projects/def-enaskt/shawnj}"
if [ -z "${MODEL:-}" ] && [ -d "${LOCAL_ROOT}/Qwen2.5-7B-Instruct" ]; then
    MODEL="${LOCAL_ROOT}/Qwen2.5-7B-Instruct"
fi
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

EMB="${EMB:-embeddings_opinionqa.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
DATASET="${DATASET:-opinionqa}"
TAU="${TAU:-0.25}"
ARMS="${ARMS:-base7b,ctx0,cad-0.5,cad-0.25,cad0.25,cad0.5}"
HALF="${HALF:-all}"
MAXQ="${MAXQ:-0}"
TEMP="${TEMP:-0.7}"
TOPP="${TOPP:-0.95}"
MAXNEW="${MAXNEW:-1024}"
OUT="${OUT:-cad_responses_${HALF}.jsonl}"
LABELS="${LABELS:-contestedness_labels.json}"

echo "MODEL=${MODEL}"
echo "ARMS='${ARMS}'  HALF=${HALF}  MAXQ=${MAXQ}  OUT=${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "${MODEL}" in
    /*|./*|~*) [ -d "${MODEL}" ] || { echo "MISSING local model dir: ${MODEL}"; exit 1; } ;;
esac
[ -f "${EMB}" ] || { echo "MISSING ${EMB}"; exit 1; }

# cad_soft consumes the probe run's labels. Same model, same prompt, so they are
# reusable -- do NOT regenerate them (that is K=8 samples x 60 questions).
case "${ARMS}" in
    *cad_soft*)
        [ -f "${LABELS}" ] || { echo "cad_soft needs ${LABELS} (jobs/train/job_probe.sh)"; exit 1; } ;;
esac

# Fail on the algebra before spending a GPU-hour on generation.
python -m retrieval.cad --selftest || { echo "CAD SELFTEST FAILED"; exit 1; }

python -u -m evaluation.overton.eval_cad \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset "${DATASET}" \
    --model "${MODEL}" --tau "${TAU}" \
    --arms "${ARMS}" --half "${HALF}" --max_questions "${MAXQ}" \
    --temperature "${TEMP}" --top_p "${TOPP}" --max_new_tokens "${MAXNEW}" \
    --labels "${LABELS}" --out "${OUT}" \
    || { echo "CAD GENERATION FAILED"; exit 1; }

echo ""
echo "Generated ${OUT}. NEXT: judge it, then compare arms to base7b/ctx0 only."
echo "READ THE SAMPLES before the scores -- contrastive decoding degenerates at"
echo "large |alpha|, and fluent nonsense scores 0 without looking broken."
