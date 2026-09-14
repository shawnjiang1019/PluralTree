#!/bin/bash
#SBATCH --job-name=anchor_cov
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/anchor_cov_%j.out
#SBATCH --error=logs/anchor_cov_%j.err

# THE GATE for any new benchmark. Before spending generation and judge hours on
# a question set, ask whether the ATP graph resolves anchors for it at all: if it
# does not, injection is inert and the benchmark cannot test the method however
# many questions it has. Costs one MiniLM pass, no generation, no judge.
#
# Reports resolution against an OvertonBench reference, so the number is read as
# "relative to a set we know the graph handles" rather than in the abstract.
#
#   SRC=valueprism RAW=valueprism.csv sbatch jobs/eval/job_anchor_coverage.sh
#   SRC=wildscope  RAW=wildscope.jsonl sbatch jobs/eval/job_anchor_coverage.sh
#   SRC=none QUESTIONS=my_questions.jsonl sbatch jobs/eval/job_anchor_coverage.sh
#
# GPU, not CPU: anchor_coverage embeds every question plus the graph's node text
# with MiniLM. default_embed_fn pins no device, so it takes CUDA when one is
# visible -- on a login node this is the ~3 texts/sec path that makes the job
# look hung. That is the whole reason this file exists.
#
# Knobs: SRC (valueprism|wildscope|none), RAW, QUESTIONS, MAXQ, EMB, DATASET,
#        TAU, SEED, MINVALS, GATE (exit 2 below this resolution rate).

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Without this load_opinionqa falls through to the gated Hub copy and dies on
# OfflineModeIsEnabled -- how job 2252726 failed.
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"
# load_issp reads this; without it DATASET=issp exits immediately.
export ISSP_DIR="${ISSP_DIR:-$HOME/projects/def-enaskt/shawnj/data/issp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

SRC="${SRC:-valueprism}"
RAW="${RAW:-}"
MAXQ="${MAXQ:-500}"              # gate only needs enough to estimate a rate
MINVALS="${MINVALS:-2}"          # valueprism: drop situations with <2 values
EMB="${EMB:-embeddings_opinionqa.pt}"
DATASET="${DATASET:-opinionqa}"
TAU="${TAU:-0.25}"
SEED="${SEED:-0}"
# Exit 2 below this resolution rate, so a dependent eval submitted with
# --dependency=afterok is genuinely GATED rather than merely sequenced: without
# it this job exits 0 on any result and the eval fires even when the graph
# reaches nothing. 0 disables; 0.60 is anchor_coverage.py's NOT-USABLE boundary.
GATE="${GATE:-0}"
if [ "${SRC}" = "none" ]; then QUESTIONS="${QUESTIONS:-}"
else QUESTIONS="${QUESTIONS:-${SRC}_questions.jsonl}"; fi
OUT="${OUT:-docs/anchor_cov_${SRC}.csv}"

echo "SRC=${SRC} RAW=${RAW} QUESTIONS=${QUESTIONS} MAXQ=${MAXQ}"
[ -f "${EMB}" ] || { echo "MISSING ${EMB} in $(pwd)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- stage 1: raw release -> {question_id, question} jsonl ------------------
if [ "${SRC}" != "none" ]; then
    [ -n "${RAW}" ] || { echo "SRC=${SRC} needs RAW=<downloaded file>"; exit 1; }
    [ -f "${RAW}" ] || { echo "MISSING ${RAW} in $(pwd)"; exit 1; }
    echo ""
    echo "=== stage 1: ${SRC} -> ${QUESTIONS} ==="
    case "${SRC}" in
        valueprism)
            python -u -m data.loaders.valueprism --path "${RAW}" \
                --out "${QUESTIONS}" --min_values "${MINVALS}" \
                --max_situations "${MAXQ}" \
                || { echo "LOADER FAILED -- the schema is guessed; the error names"; \
                     echo "the keys actually present. Edit _SITUATION/_TEXT."; exit 1; } ;;
        wildscope)
            python -u -m data.loaders.wildscope --path "${RAW}" \
                --out "${QUESTIONS}" \
                || { echo "LOADER FAILED -- see the key list in the error."; exit 1; } ;;
        *) echo "unknown SRC='${SRC}' (valueprism|wildscope|none)"; exit 1 ;;
    esac
fi

# QUESTIONS is OPTIONAL. anchor_coverage's --reference overton loads
# OvertonBench itself, so "does THIS graph resolve OvertonBench" needs no
# question file at all -- which is exactly the gate for a SWAPPED GRAPH
# (DATASET=issp). Only a NEW question set needs one.
QARG=""
if [ -n "${QUESTIONS}" ] && [ "${QUESTIONS}" != "none" ]; then
    [ -f "${QUESTIONS}" ] || { echo "MISSING ${QUESTIONS}"; exit 1; }
    echo "questions: $(grep -c . "${QUESTIONS}")"
    QARG="--questions ${QUESTIONS}"
else
    echo "no --questions: reference-only gate (does ${DATASET} reach OvertonBench?)"
fi

# --- stage 2: does the graph resolve anchors for them? ----------------------
echo ""
echo "=== stage 2: anchor resolution vs the OvertonBench reference ==="
set +e
python -u scripts/analysis/anchor_coverage.py \
    ${QARG} --embeddings "${EMB}" \
    --dataset "${DATASET}" --tau "${TAU}" --seed "${SEED}" \
    --max_questions "${MAXQ}" --reference overton --out "${OUT}" \
    --gate "${GATE}"
rc=$?
set -e
# Distinguish the two non-zero cases. `|| exit 1` would collapse them, and a
# GATE result reported as "FAILED" reads like a crash. Either way a dependent
# job submitted with --dependency=afterok will not start, which is the point.
if [ "${rc}" -eq 2 ]; then
    echo ""
    echo "GATE NOT MET (exit 2) -- this is a RESULT, not a crash. The graph does"
    echo "not reach this question set well enough to test anything on it."
    exit 2
elif [ "${rc}" -ne 0 ]; then
    echo "ANCHOR COVERAGE CRASHED (rc=${rc}) -- see .err"; exit 1
fi

echo ""
echo "Done -> ${OUT}"
echo "READ IT AS A GATE, not a result. Resolution far below the OvertonBench"
echo "reference means injection is inert on this set and no amount of extra"
echo "questions will test the method -- stop here rather than generating."
