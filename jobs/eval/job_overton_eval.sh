#!/bin/bash
#SBATCH --job-name=overton_eval
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/overton_eval_%j.out
#SBATCH --error=logs/overton_eval_%j.err

# OvertonBench generate + score as one batch job (docs/overtonbench_eval.txt §4):
# serve MODEL with vLLM, generate baseline/scout/div_only answers, judge them,
# print the OvertonScore-by-condition table.
#
# Smoke test:   MAXQ=5 MAXU=10 sbatch jobs/eval/job_overton_eval.sh
# Full run:     sbatch jobs/eval/job_overton_eval.sh            (60 q, all users; hours)
# Other knobs:  MODEL, TAU (scout gate; GOQA needs ~0.1), EMB, FEATS, DATASET, TP
#
# Prereqs (login node, once): overtonbench dataset + MODEL weights in HF_HOME;
# vLLM installed in pluraltree-env (needs the opencv module loaded BEFORE the
# env is activated). Generation is resumable: rerunning skips (question,
# condition) rows already in OUT, so a timeout just needs a resubmit.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
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

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TAU="${TAU:-0.1}"                # scout relevance gate (GOQA cross-domain ~0.1; on-domain opinionqa: 0.25)
SEED="${SEED:-42}"               # MUST match the embed job's train.py --seed (default 42):
                                 # opinionqa node ids depend on the clustering seed
MAXQ="${MAXQ:-0}"                # 0 = all 60 questions
MAXU="${MAXU:-0}"                # 0 = all participants per question
EMB="${EMB:-embeddings_goqa.pt}"
FEATS="${FEATS:-feats_goqa.pt}"
DATASET="${DATASET:-globalopinionqa}"
OUT="${OUT:-overton_responses.jsonl}"
SCORES="${SCORES:-overton_scores.csv}"
CONDS="${CONDS:-baseline,scout,div_only}"   # any of: baseline,scout,div_only,route,expand,distributional
NROLL="${NROLL:-1}"              # samples per (question,condition); >1 -> coverage@K
KROLL="${KROLL:-0}"             # judge: rollouts used for union coverage@K (0 = all)
UNION="${UNION:-}"              # cross-condition union groups, e.g.
                                # "baseline+scout,baseline+scout+expand"
                                # (empty = auto: baseline+X for each X, plus all)
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"                     # pluraltree-env's vllm (already activated)
echo "MODEL=${MODEL} TAU=${TAU} MAXQ=${MAXQ} MAXU=${MAXU} EMB=${EMB} DATASET=${DATASET}"

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

# The GPUs belong to the vLLM server (launched above, env already inherited).
# Client stages talk to it over HTTP and must not touch CUDA themselves —
# MiniLM OOMs if it lands on a GPU that is 95% Qwen.
export CUDA_VISIBLE_DEVICES=""

echo "=== stage 1: generate ==="
python -m evaluation.overton.eval_overtonbench \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset "${DATASET}" \
    --curvature 0.5 --tau "${TAU}" --seed "${SEED}" \
    --conditions "${CONDS}" --n_rollouts "${NROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --max_questions "${MAXQ}" --out "${OUT}" \
    || { echo "GENERATION FAILED (see .err)"; exit 1; }
echo "fork fallbacks (scout answers that got the baseline prompt):"
grep -c "scout returned 0 forks" logs/overton_eval_${SLURM_JOB_ID}.err || true
echo "tag failures (injected answers missing <answer> tags):"
grep -c "missing <answer> tags" logs/overton_eval_${SLURM_JOB_ID}.err || true

echo "=== stage 2: judge ==="
UNION_FLAG=""
if [ -n "${UNION}" ]; then UNION_FLAG="--union ${UNION}"; fi

python -m evaluation.overton.judge_overtonbench --score "${OUT}" \
    --max_users "${MAXU}" --k_rollouts "${KROLL}" ${UNION_FLAG} \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --out "${SCORES}" \
    || { echo "JUDGING FAILED (see .err)"; exit 1; }

echo "Done. Responses: ${OUT}  Scores: ${SCORES}"
