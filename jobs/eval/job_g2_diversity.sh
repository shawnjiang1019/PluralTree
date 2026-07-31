#!/bin/bash
#SBATCH --job-name=g2_diversity
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/g2_diversity_%j.out
#SBATCH --error=logs/g2_diversity_%j.err

# Replicate G2 (arXiv:2511.00432) on INFINITY-CHAT, then score with our existing
# diversity panel. G2 = contrastive decoding between a Diversity Guide and a
# Dedupe Guide, entropy-gated; see retrieval/g2.py.
#
# Motivation: our earlier hivemind run measured vendi 1.4/8 and mean_cos 0.92 --
# near-total mode collapse. G2's paper reports Distinct 4.02 -> 5.80 at almost no
# quality cost, and shows that PROMPTING for diversity (our `route`/`expand`) is
# what wrecks quality. This tests the logit-level alternative on our generator.
#
# ONE GPU: 7B bf16 is ~15GB; the three streams share the model (3 KV caches).
# NOT the 72B AWQ -- G2 needs logits, which the vLLM endpoint does not expose.
#
# Smoke:  NQ=3 NS=4 sbatch jobs/eval/job_g2_diversity.sh
# Run:    NQ=20 NS=8 sbatch jobs/eval/job_g2_diversity.sh
# Knobs:  MODEL, NQ, NS, THETA, BETA, KREPR, MAXNEW, GEN, DIV, EVAL_MODEL
#
# COST: G2 is 3 forward passes/token AND sequential across samples (answer i
# conditions on answers < i), so it does not batch. Budget ~1 min per answer at
# 384 tokens on an A100. NQ x NS x 2 conditions answers total.
#
# Prereqs (login node, once): MODEL + BAAI/bge-large-en-v1.5 +
# sentence-transformers/all-mpnet-base-v2 in HF_HOME; infinite-chats-eval cached.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NQ="${NQ:-20}"
NS="${NS:-8}"
THETA="${THETA:-0.3}"            # paper sweeps 0.15 / 0.3 / 0.5 / 0.7
BETA="${BETA:-0.1}"              # entropy gate, fixed at 0.1 in the paper
KREPR="${KREPR:-3}"              # representative priors (Center Selection)
MAXNEW="${MAXNEW:-384}"
CONDS="${CONDS:-baseline,g2}"    # baseline = identical path with theta=0
GEN="${GEN:-hivemind_g2.jsonl}"
DIV="${DIV:-hivemind_g2_diversity.csv}"
EVAL_MODEL="${EVAL_MODEL:-BAAI/bge-large-en-v1.5}"   # held-out scorer
echo "MODEL=${MODEL} NQ=${NQ} NS=${NS} THETA=${THETA} BETA=${BETA} KREPR=${KREPR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== stage 1: generate (G2) ==="
python -m evaluation.hivemind.generate_g2 \
    --model "${MODEL}" --conditions "${CONDS}" \
    --num_queries "${NQ}" --num_samples "${NS}" \
    --theta "${THETA}" --beta "${BETA}" --k_repr "${KREPR}" \
    --max_new_tokens "${MAXNEW}" --out "${GEN}" \
    || { echo "G2 GENERATION FAILED (see .err)"; exit 1; }

echo "=== stage 2: diversity metrics ==="
python -m evaluation.hivemind.diversity_metrics "${GEN}" --out "${DIV}" \
    --eval_model "${EVAL_MODEL}" \
    || { echo "METRICS FAILED (see .err)"; exit 1; }

echo "Done. Generations: ${GEN}  Diversity: ${DIV}"
