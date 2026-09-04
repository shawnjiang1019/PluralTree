#!/bin/bash
#SBATCH --job-name=g2_overton
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=11:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/g2_overton_%j.out
#SBATCH --error=logs/g2_overton_%j.err

# G2 decoding on OvertonBench, including the GRAPH-GUIDED variant that has never
# run (generate_g2.py only accepts baseline,g2). See evaluation/overton/
# eval_g2_overton.py for why this is the benchmark that variant needs.
#
# Stage 2 judges with a 7B; the 72B-AWQ judge needs 4 GPUs and would double the
# allocation for a job whose generation is single-GPU. Run the 72B judge
# separately if the arms separate:
#   RESP=g2_responses_a.jsonl SCORES=g2_scores_a.csv KROLL=4 \
#       sbatch jobs/eval/job_judge_only.sh
#
# Smoke:  NQ=2 NS=2 sbatch jobs/eval/job_g2_overton.sh
# Run:    sbatch jobs/eval/job_g2_overton.sh
# Knobs:  MODEL, ARMS, NQ, NS, THETA, BETA, KREPR, MAXNEW, HALF, EMB, RESP, JUDGE
#
# COST: G2 is 3 forward passes/token and SEQUENTIAL across pool members (answer i
# conditions on answers < i), so it does not batch. g2_base (theta=0) skips the
# guide passes entirely and is ~3x cheaper. Budget ~1.3 min per contrastive
# answer at 512 tokens on an A100: 30 questions x 2 contrastive arms x 4 samples
# ~ 5.2h, plus ~0.9h for g2_base. Halve NS if the queue is tight.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Without this the loader falls through to `load_dataset("jjssuh/subpop")` and
# dies on OfflineModeIsEnabled: the HF copy is gated, and the cluster reads the
# raw ATP release from disk instead. Every other job that builds the graph sets
# it; job 2252726 failed for exactly this reason.
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs docs

# Prefer local clones over hub ids: huggingface_hub 1.19.0+computecanada has a
# circular-import bug that breaks snapshot_download, so passing a DIRECTORY skips
# the hub resolver entirely and the offline flags stop mattering.
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/projects/def-enaskt/shawnj}"
if [ -z "${MODEL:-}" ] && [ -d "${LOCAL_ROOT}/Qwen2.5-7B-Instruct" ]; then
    MODEL="${LOCAL_ROOT}/Qwen2.5-7B-Instruct"
fi
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE="${JUDGE:-${MODEL}}"
ARMS="${ARMS:-g2_base,g2,g2_graph}"
NQ="${NQ:-0}"                    # 0 = every question in the half
NS="${NS:-4}"
THETA="${THETA:-0.3}"
BETA="${BETA:-0.1}"
KREPR="${KREPR:-3}"
MAXNEW="${MAXNEW:-512}"
HALF="${HALF:-a}"                # tune on a, report on b
EMB="${EMB:-embeddings_opinionqa.pt}"
RESP="${RESP:-g2_responses_${HALF}.jsonl}"
SCORES="${SCORES:-g2_scores_${HALF}.csv}"
CLUSTERS="${CLUSTERS:-${SCORES%.csv}_clusters.csv}"
PORT="${PORT:-8000}"
VLLM="${VLLM:-vllm}"

echo "MODEL=${MODEL}  ARMS=${ARMS}  HALF=${HALF}  NS=${NS}  THETA=${THETA}"
[ -f "${EMB}" ] || { echo "MISSING ${EMB} in $(pwd)"; exit 1; }
case "${MODEL}" in
    /*|./*|~*) [ -d "${MODEL}" ] || { echo "MISSING local model dir: ${MODEL}"; exit 1; } ;;
    *) echo "NOTE: '${MODEL}' is a hub id; with HF_HUB_OFFLINE=1 this needs a populated HF_HOME." ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== stage 1: generate (G2) ==="
python -u -m evaluation.overton.eval_g2_overton \
    --embeddings "${EMB}" --model "${MODEL}" --dataset opinionqa \
    --arms "${ARMS}" --half "${HALF}" --n_samples "${NS}" \
    --theta "${THETA}" --beta "${BETA}" --k_repr "${KREPR}" \
    --max_new_tokens "${MAXNEW}" --max_questions "${NQ}" \
    --out "${RESP}" \
    || { echo "G2 GENERATION FAILED (see .err)"; exit 1; }

echo ""
echo "=== stage 2: judge (k_rollouts=${NS}) ==="
"${VLLM}" serve "${JUDGE}" --port "${PORT}" --tensor-parallel-size 1 \
    --max-model-len 8192 > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null' EXIT

for _ in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null; then break; fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM died - see logs/vllm_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 20
done
curl -sf "http://localhost:${PORT}/health" > /dev/null \
    || { echo "vLLM never became healthy"; exit 1; }
echo "vLLM up"

python -u -m evaluation.overton.judge_overtonbench --score "${RESP}" \
    --max_users 20 --k_rollouts "${NS}" \
    --base_url "http://localhost:${PORT}/v1" --model "${JUDGE}" \
    --out "${SCORES}" --dump_clusters "${CLUSTERS}" \
    || { echo "JUDGING FAILED"; exit 1; }

echo ""
echo "Done -> ${SCORES}"
echo "Read coverage@K, not coverage: G2 is a pool method and its value is the"
echo "union over the pool. Compare arms to g2_base ONLY -- 7B, not the 72B runs."
echo "Then: python scripts/analysis/cluster_overlap.py --clusters ${CLUSTERS} \\"
echo "          --baseline g2_base"
