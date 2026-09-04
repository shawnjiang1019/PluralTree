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
#        TAU, SEED, MINVALS.

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
QUESTIONS="${QUESTIONS:-${SRC}_questions.jsonl}"
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

[ -f "${QUESTIONS}" ] || { echo "MISSING ${QUESTIONS}"; exit 1; }
echo "questions: $(grep -c . "${QUESTIONS}")"

# --- stage 2: does the graph resolve anchors for them? ----------------------
echo ""
echo "=== stage 2: anchor resolution vs the OvertonBench reference ==="
python -u scripts/analysis/anchor_coverage.py \
    --questions "${QUESTIONS}" --embeddings "${EMB}" \
    --dataset "${DATASET}" --tau "${TAU}" --seed "${SEED}" \
    --max_questions "${MAXQ}" --reference overton --out "${OUT}" \
    || { echo "ANCHOR COVERAGE FAILED"; exit 1; }

echo ""
echo "Done -> ${OUT}"
echo "READ IT AS A GATE, not a result. Resolution far below the OvertonBench"
echo "reference means injection is inert on this set and no amount of extra"
echo "questions will test the method -- stop here rather than generating."
