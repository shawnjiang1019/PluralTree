#!/bin/bash
#SBATCH --job-name=scale_7b
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/scale_7b_%j.out
#SBATCH --error=logs/scale_7b_%j.err

# DOES merge_v2's GAIN SURVIVE AT 7B?
#
# CAD measured what injecting retrieved forks costs, by model size:
#
#     7B    base7b 0.3942  ->  ctx0 0.0992    (-75%)
#    72B   baseline 0.497  ->  scout 0.393    (-21%)
#
# Over-anchoring is strongly size-dependent (docs/results_2026_09.md §2). That
# matters because merge_v2 IS those two conditions stitched together: draft 1 is
# `plain` (= base7b) and drafts 2-3 are fork-injected under PLURALISM_INSTRUCTION
# (= ctx0). At 7B, two of its three drafts are the collapsing condition.
#
# So if merge_v2's +0.04-0.05 at 72B is largely an anchoring workaround -- a
# guard that quarantines a harmful channel -- the gain should shrink or invert
# here, where the channel is four times more harmful and the merge has more
# damage to keep out. If instead it holds, the gain is coming from the merged
# UNION and not from damage control. Either outcome is informative; there is no
# result of this job that is uninteresting.
#
# THESE NUMBERS ARE NOT COMPARABLE TO v10 / v11 / v12. Those are
# Qwen2.5-72B-Instruct-AWQ, generator AND judge. This job is a 7B, generator and
# judge (one GPU cannot hold both, and a second allocation would not be the same
# judge pass). The absolute OvertonScores here live on a different scale. THE
# ONLY VALID COMPARISON IS baseline VS merge_v2 WITHIN THIS RUN -- both arms
# share this job's questions, this job's generator and this job's judge pass.
# Do not put a number from this file in a table next to v10/v11/v12.
#
#   Smoke:  MAXQ=5 MAXU=10 sbatch jobs/eval/job_scale_7b.sh
#   Run:    sbatch jobs/eval/job_scale_7b.sh
#
# Knobs: MODEL, LOCAL_ROOT, TAU, SEED, MAXQ, MAXU, NROLL, KROLL, CONDS, EMB,
#        FEATS, DATASET, OUT, SCORES, TP, PORT.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
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
mkdir -p logs

# Prefer a local clone over the hub id. huggingface_hub 1.19.0+computecanada has
# a circular-import bug that breaks snapshot_download, so the weights were pulled
# with `git clone https://huggingface.co/<repo>`; passing a DIRECTORY skips the
# hub resolver entirely and the offline flags stop mattering. Falling back to the
# repo id would fail under HF_HUB_OFFLINE, so resolve it here rather than relying
# on the caller exporting MODEL.
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/projects/def-enaskt/shawnj}"
if [ -z "${MODEL:-}" ] && [ -d "${LOCAL_ROOT}/Qwen2.5-7B-Instruct" ]; then
    MODEL="${LOCAL_ROOT}/Qwen2.5-7B-Instruct"
fi
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

TAU="${TAU:-0.25}"               # on-domain opinionqa gate (GOQA cross-domain: 0.1)
SEED="${SEED:-42}"               # MUST match the embed job's train.py --seed
                                 # (default 42): opinionqa node ids depend on
                                 # the clustering seed, and a mismatch makes the
                                 # scout return the wrong subgroups silently
MAXQ="${MAXQ:-0}"                # 0 = all 60 questions -- the SAME question set
                                 # v10/v11/v12 used, so the two arms here are
                                 # comparable to each other on the same items
MAXU="${MAXU:-20}"               # participants per question, as the merge runs used
EMB="${EMB:-embeddings_opinionqa.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
DATASET="${DATASET:-opinionqa}"
CONDS="${CONDS:-baseline,merge_v2}"
NROLL="${NROLL:-1}"
KROLL="${KROLL:-0}"
OUT="${OUT:-overton_responses_7b.jsonl}"
SCORES="${SCORES:-overton_scores_7b.csv}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                    # 7B bf16 is ~15GB: one GPU, no sharding. The
                                 # 72B AWQ jobs need gpu:4 / TP=4; this one asks
                                 # for a quarter of that allocation.
VLLM="${VLLM:-vllm}"

echo "MODEL=${MODEL}  (7B -- NOT comparable to v10/v11/v12)"
echo "TAU=${TAU} SEED=${SEED} MAXQ=${MAXQ} MAXU=${MAXU} CONDS=${CONDS} TP=${TP}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail in 1 second instead of after the model loads.
[ -f "${EMB}" ] || { echo "MISSING ${EMB} in $(pwd) -- run jobs/embed/job_embed_opinionqa.sh"; exit 1; }
case "${MODEL}" in
    /*|./*|~*) [ -d "${MODEL}" ] || { echo "MISSING local model dir: ${MODEL}"; exit 1; } ;;
    *) echo "NOTE: '${MODEL}' is a hub id; with HF_HUB_OFFLINE=1 this needs a "\
            "populated HF_HOME cache. Clone it and pass a directory if it fails." ;;
esac

"${VLLM}" serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len 8192 > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null" EXIT

for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
        echo "vLLM up after ~$((i * 10))s"; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "vLLM died — see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }

# The GPU belongs to the vLLM server (launched above, env already inherited).
# Client stages talk to it over HTTP and must not touch CUDA themselves —
# MiniLM OOMs if it lands on a GPU that is 95% Qwen. This matters MORE at TP=1:
# there is no second device to land on.
export CUDA_VISIBLE_DEVICES=""

echo "=== stage 1: generate (both arms, one pass, same questions) ==="
python -m evaluation.overton.eval_overtonbench \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset "${DATASET}" \
    --curvature 0.5 --tau "${TAU}" --seed "${SEED}" \
    --conditions "${CONDS}" --n_rollouts "${NROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --max_questions "${MAXQ}" --out "${OUT}" \
    || { echo "GENERATION FAILED (see .err)"; exit 1; }
echo "fork fallbacks (scout answers that got the baseline prompt):"
grep -c "scout returned 0 forks" logs/scale_7b_${SLURM_JOB_ID}.err || true
# Tag compliance is a READOUT here, not just a warning: CAD found <answer> tag
# compliance and coverage moving in OPPOSITE directions at 7B (well-formed
# answers about the wrong things), so a clean tag count is not reassurance.
echo "tag failures (injected answers missing <answer> tags):"
grep -c "missing <answer> tags" logs/scale_7b_${SLURM_JOB_ID}.err || true
echo "merge_v2 lossy fallbacks (merge compressed, concatenation used instead):"
grep -c "merge_v2 LOSSY" logs/scale_7b_${SLURM_JOB_ID}.err || true

echo "=== stage 2: judge (SAME 7B server) ==="
python -m evaluation.overton.judge_overtonbench --score "${OUT}" \
    --max_users "${MAXU}" --k_rollouts "${KROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --out "${SCORES}" \
    --dump_clusters "${SCORES%.csv}_clusters.csv" \
    || { echo "JUDGING FAILED (see .err)"; exit 1; }

echo ""
echo "Done. Responses: ${OUT}  Scores: ${SCORES}"
echo ""
echo "READ ONE NUMBER: merge_v2 - baseline, within this file."
echo "  shrunk or negative -> merge_v2's 72B gain is largely damage control on"
echo "                        an injection channel that hurts more the smaller"
echo "                        the model gets; the retrieval is not the engine"
echo "  holds (~+0.04)     -> the gain survives a 4x worse injection channel,"
echo "                        so it comes from the merged union, not the guard"
echo "The absolute scores are on a 7B judge's scale. They do NOT belong in a"
echo "table with v10/v11/v12 (Qwen2.5-72B-Instruct-AWQ)."
