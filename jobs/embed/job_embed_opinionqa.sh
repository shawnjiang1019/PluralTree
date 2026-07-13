#!/bin/bash
#SBATCH --job-name=embed_opinionqa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/embed_opinionqa_%j.out
#SBATCH --error=logs/embed_opinionqa_%j.err

# Export OpinionQA/SubPOP Poincare embeddings — the US-demographic opinion
# graph for the OvertonBench experiment (docs/overtonbench_eval.txt).
# Depth-6 hierarchy: US Public -> topic -> subtopic -> question -> axis ->
# subgroup opinion (~100k nodes, ~4x GOQA — hence mem=32G / 3h).
#
# Anti-collapse + diversity defaults, same recipe as GOQA:
#   CURV=0.5 LSTR=0.1 LDIV=0.1 DIVM=1.0    (all overridable)
#
# SubPOP is GATED on HF. Login node, once:
#   1. accept terms at https://huggingface.co/datasets/jjssuh/subpop
#   2. huggingface-cli login
#   3. export HF_HOME=~/projects/def-enaskt/shawnj/hf_cache
#      python -c "from datasets import load_dataset; load_dataset('jjssuh/subpop')"

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
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

CURV="${CURV:-0.5}"
LSTR="${LSTR:-0.1}"
LDIV="${LDIV:-0.1}"        # sibling-separation floor (diversity-as-objective)
DIVM="${DIVM:-1.0}"        # min geodesic distance between siblings
EPOCHS="${EPOCHS:-12}"     # val MRR plateaus ~epoch 9 (docs/opinionqa_train_metrics.png)
EMB="${EMB:-embeddings_opinionqa.pt}"
echo "CURV=${CURV}  LSTR=${LSTR}  LDIV=${LDIV}  DIVM=${DIVM}  EPOCHS=${EPOCHS}  EMB=${EMB}"

python scripts/train/train.py --dataset opinionqa \
    --curvature "${CURV}" --lambda_struct "${LSTR}" \
    --lambda_div "${LDIV}" --div_margin "${DIVM}" \
    --n_epochs "${EPOCHS}" \
    --save_embeddings "${EMB}" --device cuda \
    || { echo "TRAIN FAILED (see .err)"; exit 1; }

if [[ -f "${EMB}" ]]; then
    echo "Done. Wrote ${EMB} ($(du -h ${EMB} | cut -f1))"
else
    echo "ERROR: training exited 0 but ${EMB} was not written"; exit 1
fi
