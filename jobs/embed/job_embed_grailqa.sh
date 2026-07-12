#!/bin/bash
#SBATCH --job-name=embed_grailqa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/embed_grailqa_%j.out
#SBATCH --error=logs/embed_grailqa_%j.err

# Export GrailQA (dki-lab/grail_qa) Poincare embeddings with the anti-collapse
# recipe. GrailQA is a large Freebase-typed KG (root -> domain -> type -> entity,
# tens of thousands of nodes, thousands of relations), so we gradient-checkpoint
# the encoder and re-encode the tree every few steps to fit memory/time.
#
# Anti-collapse recipe (keep mass off the Poincare boundary so the geometry is
# usable — we saw CulturalBench saturate to the rim):
#   CURV   curvature c (lower = more interior room)     default 0.5
#   LSTR   structure-fidelity loss weight               default 0.1
#   LBND   boundary penalty weight (0 = off)            default 0.0
#     Set LBND>0 (e.g. 0.1) to let the boundary penalty do the anti-saturation
#     work instead of hand-tuning CURV/LSTR.
# All overridable: CURV=1.0 LSTR=0.0 LBND=0.1 sbatch jobs/embed/job_embed_grailqa.sh
#
# Requires the GrailQA JSON in data/grailqa/ and the MiniLM feature model in
# HF_HOME (compute nodes are offline). Pre-fetch both on the login node once:
#   python scripts/fetch/get_grailqa.py --out data/grailqa
#   export HF_HOME=~/projects/def-enaskt/shawnj/hf_cache
#   python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

CURV="${CURV:-0.5}"
LSTR="${LSTR:-0.1}"
LBND="${LBND:-0.0}"
LDIV="${LDIV:-0.1}"        # sibling-separation floor (diversity-as-objective)
DIVM="${DIVM:-1.0}"        # min geodesic distance between siblings
EMB="${EMB:-embeddings_grailqa.pt}"
echo "CURV=${CURV}  LSTR=${LSTR}  LBND=${LBND}  LDIV=${LDIV}  DIVM=${DIVM}  EMB=${EMB}"

python scripts/train/train.py --dataset grailqa \
    --curvature "${CURV}" --lambda_struct "${LSTR}" --lambda_boundary "${LBND}" \
    --lambda_div "${LDIV}" --div_margin "${DIVM}" \
    --checkpoint --encode_every 4 \
    --save_embeddings "${EMB}" --device cuda \
    || { echo "TRAIN FAILED (see .err)"; exit 1; }

if [[ -f "${EMB}" ]]; then
    echo "Done. Wrote ${EMB} ($(du -h ${EMB} | cut -f1))"
else
    echo "ERROR: training exited 0 but ${EMB} was not written"; exit 1
fi
