#!/bin/bash
#SBATCH --job-name=retrieval_diag
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/retrieval_diag_%j.out
#SBATCH --error=logs/retrieval_diag_%j.err

# Offline retrieval diagnostic (checklist B): no LLM, no vLLM. Builds the
# second encoder's node-feature cache on the GPU if it is missing, runs
# scripts/analysis/retrieval_diag.py for both encoders, then compares them.
#
#   sbatch jobs/eval/job_retrieval_diag.sh
#   EMBED2=BAAI/bge-m3 FEATS2=feats_opinionqa_bge.pt FOCUS="russia china" sbatch jobs/eval/job_retrieval_diag.sh
#
# Knobs: DATASET, SEED, EMB, FEATS1/EMBED1 (reference), FEATS2/EMBED2, FOCUS.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs

DATASET="${DATASET:-opinionqa}"
SEED="${SEED:-42}"                       # must match the embed job's train.py --seed
EMB="${EMB:-embeddings_opinionqa.pt}"
EMBED1="${EMBED1:-all-MiniLM-L6-v2}"
FEATS1="${FEATS1:-feats_opinionqa.pt}"
EMBED2="${EMBED2:-BAAI/bge-m3}"
FEATS2="${FEATS2:-feats_opinionqa_bge.pt}"
FOCUS="${FOCUS:-russia}"
TAG1="${EMBED1##*/}"
TAG2="${EMBED2##*/}"
echo "DATASET=${DATASET} EMB=${EMB} ${EMBED1}:${FEATS1} vs ${EMBED2}:${FEATS2}"

for pair in "${EMBED1}|${FEATS1}" "${EMBED2}|${FEATS2}"; do
    model="${pair%%|*}"; feats="${pair##*|}"
    if [ ! -f "${feats}" ]; then
        echo "=== building ${feats} with ${model} (GPU) ==="
        python -c "from data.loaders.graphs import load_graph; from retrieval.scout import load_or_compute_text_feat; g = load_graph('${DATASET}', split_seed=${SEED}, leakage_safe=True); f = load_or_compute_text_feat(g, '${DATASET}', '${feats}', model_name='${model}'); print('features', tuple(f.shape))" \
            || { echo "FEATURE BUILD FAILED (see .err)"; exit 1; }
    fi
done

for pair in "${EMBED1}|${FEATS1}|${TAG1}" "${EMBED2}|${FEATS2}|${TAG2}"; do
    IFS='|' read -r model feats tag <<< "${pair}"
    echo "=== diagnostic: ${model} ==="
    python scripts/analysis/retrieval_diag.py --dataset "${DATASET}" --seed "${SEED}" \
        --embeddings "${EMB}" --text_feat "${feats}" --embed_model "${model}" \
        --out "retrieval_diag_${tag}.csv" \
        || { echo "DIAGNOSTIC FAILED for ${model} (see .err)"; exit 1; }
done

echo "=== compare ==="
python scripts/analysis/compare_retrieval_diag.py \
    --csv "retrieval_diag_${TAG1}.csv" "retrieval_diag_${TAG2}.csv" \
    --focus ${FOCUS} --out retrieval_diag_compare.md
