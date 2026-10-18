#!/bin/bash
#SBATCH --job-name=vital_nli
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/vital_nli_%j.out
#SBATCH --error=logs/vital_nli_%j.err

# Score VITAL Overton answers with the published NLI protocol (Modular Pluralism's
# evaluate_overton_valuekaleidoscope.py, as used by VITAL). Reads files written
# by job_vital.sh / eval_vital.py; no generation, no vLLM.
#
# One-time setup on a LOGIN node (compute nodes are offline):
#   source ~/pluraltree-env/bin/activate
#   pip install sentence-splitter
#   python -c "from transformers import AutoModelForSequenceClassification as M, \
#     AutoTokenizer as T; n='microsoft/deberta-v2-xlarge-mnli'; T.from_pretrained(n); \
#     M.from_pretrained(n)"
#
#   RESP=vital_responses_vk.jsonl REF=plain_merge sbatch jobs/eval/job_vital_nli.sh
#
# Knobs: RESP (space- or comma-separated), VITAL, REF, CONDS, OUT, BATCH.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs

RESP="${RESP:-vital_responses_vk.jsonl}"
VITAL="${VITAL:-vital_overton_valuekaleidoscope.json}"
REF="${REF:-plain_merge}"          # length-matched to the persona arms
CONDS="${CONDS:-}"
OUT="${OUT:-vital_nli_scores.csv}"
BATCH="${BATCH:-32}"

python -c "import sentence_splitter" 2>/dev/null \
  || echo "WARNING: sentence-splitter missing -> regex fallback, not comparable"

python -u scripts/analysis/score_vital_nli.py \
    --responses ${RESP//,/ } --vital "${VITAL}" --reference "${REF}" \
    --conditions "${CONDS}" --batch "${BATCH}" --out "${OUT}" \
    || { echo "SCORING FAILED (see .err)"; exit 1; }
