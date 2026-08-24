#!/bin/bash
#SBATCH --job-name=contest_probe
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/contest_probe_%j.out
#SBATCH --error=logs/contest_probe_%j.err

# Adaptation 2 of docs/adaptive_injection.md: train the CtrlA-style
# contestedness probe end to end.
#
#   stage 0  self-tests (CPU, seconds) -- crash gate before any GPU work
#   stage 1  LABELS: K committed samples/question -> self-consistency score
#            + k-means topics            scripts/generate_contestedness_labels.py
#   stage 2  PROBE: hidden states -> logistic regression, leave-one-topic-out,
#            then scored post-hoc against a finished OvertonBench run
#                                        alignment/probe.py
#
# Why local HF weights and not the vLLM endpoint: the endpoint does not expose
# hidden states, which is the entire premise (self-report scored 0.072 because
# the signal is gone by the time it is a token). Stage 1 also runs locally so
# the LABEL describes the same model the probe reads -- labelling with the 72B
# AWQ and probing the 7B would measure two different models.
#
# COST: generation is paid ONCE, here. NQ x K short samples in stage 1
# (~15 min for 60 x 8 at 256 tokens on an A100). Stage 2 is 60 forward passes
# with no generation, and probe inference later is one forward pass per query.
#
# Smoke:  MAXQ=6 K=4 KTOP=2 sbatch jobs/train/job_probe.sh
# Run:    SCORES=overton_scores_v5.csv sbatch jobs/train/job_probe.sh
# Knobs:  MODEL, EMBEDDER, K, KTOP, TEMP, LAYER, MAXQ, SCORES, INJECT,
#         LABELS, OUT, L2, EPOCHS, PROMPT, FORCE_LABELS
#
# Prereqs (login node, once): MODEL and EMBEDDER cloned as DIRECTORIES; the
# OvertonBench dataset cached in HF_HOME (stage 1 calls load_questions).

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs runs

# Prefer local clones over hub ids. huggingface_hub 1.19.0+computecanada has a
# circular-import bug that breaks snapshot_download, so weights were pulled with
# `git clone https://huggingface.co/<repo>`; passing a DIRECTORY skips the hub
# resolver entirely and the offline flags stop mattering. Resolve at submit time
# rather than relying on the caller exporting MODEL.
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/projects/def-enaskt/shawnj}"
if [ -z "${MODEL:-}" ] && [ -d "${LOCAL_ROOT}/Qwen2.5-7B-Instruct" ]; then
    MODEL="${LOCAL_ROOT}/Qwen2.5-7B-Instruct"
fi
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Held-out embedder: mpnet, deliberately NOT the scout's MiniLM, so neither the
# stance-spread label nor the topic clustering is a function of the retrieval
# the gate governs.
if [ -z "${EMBEDDER:-}" ] && [ -d "${LOCAL_ROOT}/all-mpnet-base-v2" ]; then
    EMBEDDER="${LOCAL_ROOT}/all-mpnet-base-v2"
fi
EMBEDDER="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"

K="${K:-8}"                    # samples/question; spread is unstable below ~5
TEMP="${TEMP:-1.0}"            # must be >0 or every sample is identical
KTOP="${KTOP:-8}"              # topics = leave-one-topic-out folds (~7/topic)
LAYER="${LAYER:--1}"
PROMPT="${PROMPT:-chat}"       # read states under the prompt the label used
L2="${L2:-1e-3}"
EPOCHS="${EPOCHS:-400}"
MAXQ="${MAXQ:-0}"              # 0 = all 60
SCORES="${SCORES:-overton_scores_v5.csv}"
INJECT="${INJECT:-scout}"
LABELS="${LABELS:-contestedness_labels.json}"
OUT="${OUT:-runs/contestedness_probe.pt}"
FEATS="${FEATS:-runs/contestedness_feats.pt}"
FORCE_LABELS="${FORCE_LABELS:-0}"

echo "MODEL=${MODEL}"
echo "EMBEDDER=${EMBEDDER}"
echo "K=${K} TEMP=${TEMP} KTOP=${KTOP} LAYER=${LAYER} PROMPT=${PROMPT} MAXQ=${MAXQ}"
echo "LABELS=${LABELS} OUT=${OUT} SCORES=${SCORES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail in 1 second instead of 100 lines of traceback 3 minutes in.
for M in "${MODEL}" "${EMBEDDER}"; do
    case "${M}" in
        /*|./*|~*) [ -d "${M}" ] || { echo "MISSING local model dir: ${M}"; exit 1; } ;;
        *) echo "NOTE: '${M}' is a hub id; with HF_HUB_OFFLINE=1 this needs a "\
                "populated HF_HOME cache. Clone it and pass a directory if it fails." ;;
    esac
done

# --- stage 0: self-tests (CPU, seconds) ---------------------------------
# Both are synthetic (planted signal + shuffled-label control). They verify the
# math, not the model; they cost nothing and catch a broken checkout before the
# GPU is spent.
echo "=== stage 0: self-tests ==="
python scripts/generate_contestedness_labels.py --selftest \
    || { echo "LABEL SELFTEST FAILED"; exit 1; }
python -m alignment.probe --selftest \
    || { echo "PROBE SELFTEST FAILED"; exit 1; }

# --- stage 1: labels ----------------------------------------------------
# Self-consistency, NOT graph divergence: the scout selects max-Wasserstein
# forks, so divergence has no variance left and corr(w_raw, help-delta) is only
# +0.19. Training on it inherits that ceiling by construction.
if [ -f "${LABELS}" ] && [ "${FORCE_LABELS}" != "1" ]; then
    echo "=== stage 1: reusing existing ${LABELS} (FORCE_LABELS=1 to regenerate) ==="
else
    echo "=== stage 1: generate contestedness labels (${K} samples/question) ==="
    python scripts/generate_contestedness_labels.py \
        --model "${MODEL}" --embedder "${EMBEDDER}" \
        --k "${K}" --temperature "${TEMP}" --k_topics "${KTOP}" \
        --max_questions "${MAXQ}" --seed 0 \
        --scores "${SCORES}" --inject_cond "${INJECT}" \
        --out "${LABELS}" \
        || { echo "LABEL GENERATION FAILED (see .err)"; exit 1; }
fi

# --- stage 2: probe -----------------------------------------------------
# Holdout is by TOPIC. With ~60 questions the train AUC is ~1.0 no matter what;
# the pooled out-of-fold AUC is the only number worth reading, and --scores
# re-scores those out-of-fold predictions through route_signal's own
# best-single-threshold gate so the probe sits in the same table as w_raw.
echo "=== stage 2: train + evaluate probe ==="
python -m alignment.probe \
    --labels "${LABELS}" --model "${MODEL}" --layer "${LAYER}" \
    --prompt "${PROMPT}" --group_by topic --l2 "${L2}" --epochs "${EPOCHS}" \
    --scores "${SCORES}" --inject_cond "${INJECT}" \
    --features_out "${FEATS}" --out "${OUT}" \
    || { echo "PROBE TRAINING FAILED (see .err)"; exit 1; }

echo ""
echo "Done. Labels: ${LABELS}  Probe: ${OUT}  Features: ${FEATS}"
echo "READ: held-out (out-of-fold) AUC, not train AUC; and whether the probe's"
echo "corr(delta) beats the graph ceiling +0.19 with helped/hurt separating."
