#!/bin/bash
#SBATCH --job-name=text_feat
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/text_feat_%j.out
#SBATCH --error=logs/text_feat_%j.err

# Build a node-feature cache (retrieval relevance features) for one encoder.
# Node ids depend on SEED, so it must match the embed job's train.py --seed.
#
#   sbatch jobs/embed/job_text_feat.sh                                   (bge-m3)
#   EMBED=all-MiniLM-L6-v2 FEATS=feats_opinionqa.pt sbatch jobs/embed/job_text_feat.sh

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
SEED="${SEED:-42}"
EMBED="${EMBED:-BAAI/bge-m3}"
FEATS="${FEATS:-feats_opinionqa_bge.pt}"
echo "DATASET=${DATASET} SEED=${SEED} EMBED=${EMBED} FEATS=${FEATS}"

if [ -f "${FEATS}" ]; then
    echo "${FEATS} already exists -- delete it first to rebuild"; exit 0
fi

python -c "from data.loaders.graphs import load_graph; from retrieval.scout import load_or_compute_text_feat; g = load_graph('${DATASET}', split_seed=${SEED}, leakage_safe=True); print('subgroup leaves', len(g.opinion_dist), 'with option_ordinal', len(getattr(g, 'opinion_ordinal', {}))); f = load_or_compute_text_feat(g, '${DATASET}', '${FEATS}', model_name='${EMBED}'); print('features', tuple(f.shape))" || { echo "FEATURE BUILD FAILED (see .err)"; exit 1; }
echo "Done. Wrote ${FEATS}"
