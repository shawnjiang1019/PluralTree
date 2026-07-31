#!/bin/bash
#SBATCH --job-name=hivemind_metrics
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/hivemind_metrics_%j.out
#SBATCH --error=logs/hivemind_metrics_%j.err

# Stage 2 of the INFINITY-CHAT eval, standalone: score an EXISTING generations
# file. No vLLM -- so a generation run that hit the walltime still yields results
# without waiting for it to finish (the panel is per-(query, condition) pool, so
# a partial file is simply fewer pools).
#
# NEEDS A GPU. A full run is ~13.5k responses through bge-large (335M); on CPU
# that alone is hours, and a first attempt died at the 1.5h walltime having
# printed nothing after "loading eval embedder". The second cost is self-BLEU,
# O(n^2) per pool in pure python -- lower BLEUCAP if that dominates.
#
# Usage:  GEN=hivemind_gen.jsonl DIV=hivemind_diversity.csv \
#             sbatch jobs/eval/job_hivemind_metrics.sh
# Knobs:  GEN, DIV, EVAL_MODEL, DEVICE, BLEUCAP
#
# Prereq: BAAI/bge-large-en-v1.5 in HF_HOME (see job_hivemind_diversity.sh).

module load python/3.11 gcc arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

GEN="${GEN:-hivemind_gen.jsonl}"
DIV="${DIV:-hivemind_diversity.csv}"
EVAL_MODEL="${EVAL_MODEL:-BAAI/bge-large-en-v1.5}"   # held-out (NOT the scout's MiniLM)
DEVICE="${DEVICE:-auto}"         # auto -> cuda when a GPU is present
BLEUCAP="${BLEUCAP:-30}"         # samples/pool for the O(n^2) self-BLEU
echo "GEN=${GEN}  DIV=${DIV}  EVAL_MODEL=${EVAL_MODEL}  DEVICE=${DEVICE}  BLEUCAP=${BLEUCAP}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "pools present:"
python -c "
import json, collections, sys
c = collections.Counter()
for line in open('${GEN}', encoding='utf-8'):
    r = json.loads(line); c[r['condition']] += 1
qs = set()
for line in open('${GEN}', encoding='utf-8'):
    qs.add(json.loads(line)['query_id'])
print('  queries:', len(qs), ' samples by condition:', dict(c))
"

python -m evaluation.hivemind.diversity_metrics "${GEN}" --out "${DIV}" \
    --eval_model "${EVAL_MODEL}" --device "${DEVICE}" --bleu_cap "${BLEUCAP}" \
    || { echo "METRICS FAILED (see .err)"; exit 1; }

echo "Done. Diversity: ${DIV}"
