#!/bin/bash
#SBATCH --job-name=sel_ablation
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/sel_ablation_%j.out
#SBATCH --error=logs/sel_ablation_%j.err

# DOES MAX-WASSERSTEIN FORK SELECTION BEAT PICKING A SIBLING PAIR AT RANDOM?
#
# merge_v2 retrieves an anchor subtree and forks on the sibling pair with the
# largest Wasserstein gap between their opinion distributions. That selection
# rule is a claim: that WHICH pair we fork on matters. merge_v2_divrand is the
# same pipeline with the pair drawn uniformly at random from the same anchor
# subtree -- same graph, same anchor, same number of forks, same merge, only the
# selection rule swapped. If the two tie, the selection rule is decoration and
# the contribution is just "two drafts that differ".
#
# NOT the same control as merge_v2_rand (docs/random_fork_control.md): that one
# swaps the fork CONTENT for an unrelated subtree. This one keeps the content
# on-anchor and only randomises the CHOICE within it.
#
# ONE generation run and ONE judge pass for all three arms. Splitting them
# across jobs would give the arms different questions and different judge
# samples, and a 0.04-0.05 effect (v10 +0.0389, v11 +0.0498) does not survive
# that kind of noise.
#
#   Smoke:  MAXQ=5 MAXU=10 NROLL=1 sbatch jobs/eval/job_selection_ablation.sh
#   Run:    sbatch jobs/eval/job_selection_ablation.sh
#
# Knobs: MODEL, TAU, SEED, MAXQ, MAXU, NROLL, KROLL, CONDS, EMB, FEATS,
#        DATASET, OUT, SCORES, TP, PORT.

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

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"   # the generator of v10/v11/v12
TAU="${TAU:-0.25}"               # on-domain opinionqa gate (GOQA cross-domain: 0.1)
SEED="${SEED:-42}"               # MUST match the embed job's train.py --seed
                                 # (default 42): opinionqa node ids depend on
                                 # the clustering seed, and a mismatch makes the
                                 # scout return the wrong subgroups silently
MAXQ="${MAXQ:-0}"                # 0 = all 60 questions
MAXU="${MAXU:-20}"               # participants per question in the judge
EMB="${EMB:-embeddings_opinionqa.pt}"
FEATS="${FEATS:-feats_opinionqa.pt}"
DATASET="${DATASET:-opinionqa}"
CONDS="${CONDS:-baseline,merge_v2,merge_v2_divrand}"
NROLL="${NROLL:-3}"              # 3 rollouts: the replication runs use 3, and
                                 # random selection is a HIGH-VARIANCE arm --
                                 # one draw per question would confound the
                                 # selection rule with the luck of the draw
KROLL="${KROLL:-0}"              # judge: rollouts used for union coverage@K (0 = all)
OUT="${OUT:-overton_responses_sel.jsonl}"
SCORES="${SCORES:-overton_scores_sel.csv}"
CLUSTERS="${CLUSTERS:-${SCORES%.csv}_clusters.csv}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"

echo "MODEL=${MODEL} TAU=${TAU} SEED=${SEED} MAXQ=${MAXQ} MAXU=${MAXU} NROLL=${NROLL}"
echo "CONDS=${CONDS}  EMB=${EMB}  DATASET=${DATASET}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail in 1 second instead of after the model loads.
[ -f "${EMB}" ] || { echo "MISSING ${EMB} in $(pwd) -- run jobs/embed/job_embed_opinionqa.sh"; exit 1; }
case "${MODEL}" in
    /*|./*|~*) [ -d "${MODEL}" ] || { echo "MISSING local model dir: ${MODEL}"; exit 1; } ;;
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

# The GPUs belong to the vLLM server (launched above, env already inherited).
# Client stages talk to it over HTTP and must not touch CUDA themselves —
# MiniLM OOMs if it lands on a GPU that is 95% Qwen.
export CUDA_VISIBLE_DEVICES=""

echo "=== stage 1: generate (all arms, one pass) ==="
python -m evaluation.overton.eval_overtonbench \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset "${DATASET}" \
    --curvature 0.5 --tau "${TAU}" --seed "${SEED}" \
    --conditions "${CONDS}" --n_rollouts "${NROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --max_questions "${MAXQ}" --out "${OUT}" \
    || { echo "GENERATION FAILED (see .err)"; exit 1; }
echo "fork fallbacks (scout answers that got the baseline prompt):"
grep -c "scout returned 0 forks" logs/sel_ablation_${SLURM_JOB_ID}.err || true
echo "tag failures (injected answers missing <answer> tags):"
grep -c "missing <answer> tags" logs/sel_ablation_${SLURM_JOB_ID}.err || true

echo "=== stage 2: judge (one pass over every arm) ==="
# --dump_clusters is not optional here: the score alone cannot say WHETHER the
# two selection rules recover the same clusters by different routes or genuinely
# different ones, and cluster_overlap.py below reads this file.
python -m evaluation.overton.judge_overtonbench --score "${OUT}" \
    --max_users "${MAXU}" --k_rollouts "${KROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --out "${SCORES}" \
    --dump_clusters "${CLUSTERS}" \
    || { echo "JUDGING FAILED (see .err)"; exit 1; }

echo ""
echo "Done. Responses: ${OUT}  Scores: ${SCORES}  Clusters: ${CLUSTERS}"
echo ""
echo "RUN THESE TWO, IN THIS ORDER, BEFORE READING THE SCORES:"
echo ""
echo "  python scripts/analysis/check_random_fork.py --responses ${OUT}"
echo ""
echo "     The manipulation check. It EXITS NON-ZERO on a fork-count mismatch"
echo "     between the real and randomised arms -- exactly the failure that"
echo "     invalidated the previous control arm (v12: 1.00 vs 4.85 forks, so"
echo "     the 'tie' was comparing different injection VOLUMES, not different"
echo "     selection rules). If it fails, the deltas below mean nothing; fix"
echo "     the arm rather than interpreting them."
echo ""
echo "  python scripts/analysis/cluster_overlap.py --clusters ${CLUSTERS} \\"
echo "      --conditions merge_v2,merge_v2_divrand"
echo ""
echo "     Which clusters each rule uniquely recovers. Equal scores with"
echo "     DISJOINT unique clusters is a different finding from equal scores"
echo "     over the same clusters."
