#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

# Embedding-vs-interpretation diagnosis:
#   1. probe   -> how much of each fact is IN the embedding (the ceiling)
#   2. eval    -> how much the Map Reader actually READS (d_shuf)        [if a run exists]
#   3. compare -> one verdict table per fact (EMBEDDING vs INTERPRETATION)
#
# Env vars (optional):
#   EMB      embeddings .pt            (default: embeddings_up.pt)
#   DATASET  wn18rr|culturalbench|globalopinionqa  (default: wn18rr)
#   RUN      trained Map Reader dir    (default: runs/map_reader; eval skipped if absent)
#   HIDDEN   MLP probe width           (default: 256; 0 = linear only)
#
# Examples:
#   sbatch jobs/eval/job_probe.sh
#   EMB=embeddings_cb.pt DATASET=culturalbench sbatch jobs/eval/job_probe.sh

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

# The Map Reader eval loads the (4-bit) LM offline; the probe itself needs no LM.
export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p runs logs

EMB="${EMB:-embeddings_up.pt}"
DATASET="${DATASET:-wn18rr}"
DATA_DIR="data/wn18rr"
RUN="${RUN:-runs/map_reader}"
SPLIT="runs/holdout.json"
SFT="data/map_reader_sft.jsonl"
HIDDEN="${HIDDEN:-256}"

PROBE_JSON="runs/probe_${DATASET}.json"
MR_JSON="runs/mapreader_eval_${DATASET}.json"

if [[ ! -f "${EMB}" ]]; then
    echo "ERROR: embeddings '${EMB}' not found."; exit 1
fi
echo "EMB=${EMB}  DATASET=${DATASET}  RUN=${RUN}  HIDDEN=${HIDDEN}"

# --- 1. probe: the embedding's information ceiling -----------------------
echo "### Step 1: probe (info ceiling)"
python scripts/analysis/probe_embeddings.py \
    --embeddings "${EMB}" --dataset "${DATASET}" --data_dir "${DATA_DIR}" \
    --split "${SPLIT}" --hidden "${HIDDEN}" --json "${PROBE_JSON}" \
    || { echo "PROBE FAILED"; exit 1; }

# --- 2. Map Reader eval: what the LM actually reads (if a run exists) -----
MR_ARG=""
if [[ -d "${RUN}" && -f "${RUN}/bridge_config.json" ]]; then
    echo "### Step 2: Map Reader ablation eval"
    python scripts/analysis/eval_map_reader.py \
        --run "${RUN}" --embeddings "${EMB}" --data "${SFT}" --json "${MR_JSON}" \
        && MR_ARG="--mapreader ${MR_JSON}" \
        || echo "WARN: eval failed; comparing probe only"
else
    echo "### Step 2: skipped (no trained Map Reader at ${RUN})"
fi

# --- 3. comparison verdict table ----------------------------------------
echo "### Step 3: probe vs Map Reader verdict"
python scripts/analysis/compare_probe_mapreader.py --probe "${PROBE_JSON}" ${MR_ARG}

echo "Done."
