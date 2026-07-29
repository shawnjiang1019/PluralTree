#!/bin/bash
#SBATCH --job-name=judge_validate
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/judge_validate_%j.out
#SBATCH --error=logs/judge_validate_%j.err

# Validate an open-weight OvertonBench judge (docs/overtonbench_eval.txt §2):
# serve MODEL with vLLM, predict held-out human ratings, report MAE/Spearman
# vs the mean-of-others baseline and the paper's Gemini numbers.
#
# Overridable:  MODEL=Qwen/Qwen2.5-32B-Instruct N=150 sbatch jobs/eval/job_judge_validate.sh
#
# Login node, once (compute nodes are offline):
#   export HF_HOME=~/projects/def-enaskt/shawnj/hf_cache
#   python -c "from datasets import load_dataset; load_dataset('elinorpd/overtonbench')"
#   huggingface-cli download "$MODEL"

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
mkdir -p logs

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct}"
N="${N:-300}"                    # held-out ratings for per-rating validation;
                                 # 0 = SKIP it (not "all" — 0 samples is NaN)
REL="${REL:-1}"                  # 1 = human split-half reliability (free, no
                                 # judge calls): the ceiling any judge could hit
WITHIN="${WITHIN:-40}"           # participants for within-participant discrimination
                                 # (~8 judge calls each); 0 = skip
AGG="${AGG:-1}"                  # 1 = run the aggregate judge-vs-human OvertonScore
                                 # check (the paper's rho=0.88 level); 0 = skip
AGGU="${AGGU:-5}"                # participants/question for the aggregate check
                                 # (cost ~= 60 questions x AGGU x 8 models)
PORT="${PORT:-8000}"
TP="${TP:-4}"                    # tensor parallel = GPUs requested above
MAXLEN="${MAXLEN:-8192}"
GPUUTIL="${GPUUTIL:-0.90}"
VLLM_EXTRA="${VLLM_EXTRA:-}"     # e.g. --kv-cache-dtype fp8
# MEMORY: bf16 72B is ~145GB of weights. 4x A100-40GB = 160GB total, ~144GB
# usable at 0.90 util -> does NOT fit before any KV cache. To run bf16 72B:
#     sbatch --gres=gpu:8 --nodes=1 ... TP=8      (if 8-GPU nodes exist)
# Otherwise prefer Int8 (~73GB, fits on 4 GPUs) to test the quantization
# hypothesis: MODEL=Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8
echo "MODEL=${MODEL}  N=${N}  WITHIN=${WITHIN}  AGG=${AGG}(u=${AGGU})  TP=${TP}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

vllm serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len "${MAXLEN}" --gpu-memory-utilization "${GPUUTIL}" \
    ${VLLM_EXTRA} > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null" EXIT

# Wait for the server (72B load takes several minutes).
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

FLAGS=""
if [ "${N}" != "0" ]; then FLAGS="--validate --n ${N}"; fi
if [ "${REL}" = "1" ]; then FLAGS="${FLAGS} --human_reliability"; fi
if [ "${AGG}" = "1" ]; then FLAGS="${FLAGS} --validate_aggregate --agg_max_users ${AGGU}"; fi

python -m evaluation.overton.judge_overtonbench ${FLAGS} \
    --within_n "${WITHIN}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    || { echo "VALIDATION FAILED (see .err)"; exit 1; }
