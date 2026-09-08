#!/bin/bash
#SBATCH --job-name=geom_ablation
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/geom_ablation_%j.out
#SBATCH --error=logs/geom_ablation_%j.err

# DOES THE HYPERBOLIC GEOMETRY BUY COVERAGE?
#
# Link prediction already says the manifold fits the graph (opinionqa val MRR
# 0.456). That is NOT the claim the method makes. The claim is that hyperbolic
# structure produces better FORKS, and the only measurement of that is
# OvertonScore. So: train the same model at two curvatures and run the same
# OvertonBench pipeline on each.
#
#   CURV=0.5  hyperbolic arm (the project default; c is the ball curvature)
#   CURV=0    Euclidean arm. PoincareBall clamps c to |c|+MIN_NORM (1e-15), so
#             c=0 is the flat limit: the exp/log maps degrade to the identity
#             and every geodesic distance becomes the Euclidean one. Nothing
#             NaNs, and no separate Euclidean code path had to be invented.
#
# TWO STAGES in one allocation: (a) train.py exports embeddings at CURV, exactly
# as jobs/embed/job_embed_opinionqa.sh does and honouring its knobs; (b) the
# OvertonBench generate+judge pipeline of jobs/eval/job_overton_eval.sh runs on
# them. Stage (a) is skipped when the embeddings file already exists, so a
# re-run only re-evals. The 4-GPU allocation is sized for the vLLM server;
# train.py takes one of those GPUs for ~1h and leaves the rest idle.
#
#   sbatch jobs/eval/job_geometry_ablation.sh                 # hyperbolic
#   CURV=0 sbatch jobs/eval/job_geometry_ablation.sh          # Euclidean
#
# Knobs: CURV, LSTR, LDIV, DIVM, EPOCHS, MODEL, TAU, SEED, MAXQ, MAXU, NROLL,
#        CONDS, DATASET, TP, PORT.

module load python/3.11 gcc cuda/13.2 arrow/24.0.0 opencv/4.13.0
source ~/pluraltree-env/bin/activate

export HF_HOME="${HF_HOME:-$HOME/projects/def-enaskt/shawnj/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Without this load_opinionqa falls through to the gated Hub copy and dies on
# OfflineModeIsEnabled -- how job 2252726 failed. BOTH stages load the graph.
export OPINIONQA_DIR="${OPINIONQA_DIR:-$HOME/projects/def-enaskt/shawnj/data/human_resp}"

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree || exit 1
mkdir -p logs

# --- stage (a) knobs: same recipe as jobs/embed/job_embed_opinionqa.sh -------
CURV="${CURV:-0.5}"
LSTR="${LSTR:-0.1}"
LDIV="${LDIV:-0.1}"        # sibling-separation floor (diversity-as-objective)
DIVM="${DIVM:-1.0}"        # min geodesic distance between siblings
EPOCHS="${EPOCHS:-12}"     # val MRR plateaus ~epoch 9 (docs/opinionqa_train_metrics.png)

# THE FILENAME MUST ENCODE THE CURVATURE. Both arms run in the repo root with
# the same defaults; one shared name means arm 2 silently re-evals arm 1's
# embeddings and the "ablation" reports the same number twice. 0.5 -> c0p5.
CTAG="c${CURV//./p}"
EMB="${EMB:-embeddings_opinionqa_${CTAG}.pt}"
# Text features are raw MiniLM node text -- curvature-independent, so the two
# arms SHARE this cache on purpose (load_or_compute_text_feat writes it if it
# is missing).
FEATS="${FEATS:-feats_opinionqa.pt}"

# --- stage (b) knobs: same as jobs/eval/job_overton_eval.sh ------------------
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"   # the generator of v10/v11/v12
TAU="${TAU:-0.25}"               # on-domain opinionqa gate (GOQA cross-domain: 0.1)
SEED="${SEED:-42}"               # ONE seed, used by BOTH stages -- see below
MAXQ="${MAXQ:-0}"                # 0 = all 60 questions
MAXU="${MAXU:-20}"               # participants per question in the judge
DATASET="${DATASET:-opinionqa}"
CONDS="${CONDS:-baseline,merge_v2}"
NROLL="${NROLL:-1}"
KROLL="${KROLL:-0}"
OUT="${OUT:-overton_responses_geom_${CTAG}.jsonl}"
SCORES="${SCORES:-overton_scores_geom_${CTAG}.csv}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
VLLM="${VLLM:-vllm}"

# SEED IS ONE VARIABLE, NOT TWO KNOBS. train.py --seed and eval_overtonbench
# --seed both feed load_graph(split_seed=...), and the opinionqa node ids come
# out of that clustering seed. Train at 42 and eval at 0 and row i of the
# embedding tensor is a different subgroup than the scout believes it is:
# retrieval returns garbage forks, quietly, with no error. job_overton_eval.sh
# carries the same warning on its SEED knob. Passing one variable to both stages
# is the enforcement.
echo "CURV=${CURV} (${CTAG})  SEED=${SEED}  EMB=${EMB}"
echo "MODEL=${MODEL} TAU=${TAU} DATASET=${DATASET} CONDS=${CONDS} MAXQ=${MAXQ} MAXU=${MAXU}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "${MODEL}" in
    /*|./*|~*) [ -d "${MODEL}" ] || { echo "MISSING local model dir: ${MODEL}"; exit 1; } ;;
esac

# --- stage (a): embeddings at this curvature --------------------------------
if [ -f "${EMB}" ]; then
    echo "=== stage a: SKIP -- ${EMB} exists ($(du -h "${EMB}" | cut -f1)) ==="
    echo "    delete it to retrain; as-is this run only re-evals"
else
    echo "=== stage a: train embeddings at curvature ${CURV} ==="
    python scripts/train/train.py --dataset opinionqa \
        --curvature "${CURV}" --lambda_struct "${LSTR}" \
        --lambda_div "${LDIV}" --div_margin "${DIVM}" \
        --n_epochs "${EPOCHS}" --seed "${SEED}" \
        --save_embeddings "${EMB}" --device cuda \
        || { echo "TRAIN FAILED (see .err)"; exit 1; }
    [ -f "${EMB}" ] \
        || { echo "ERROR: training exited 0 but ${EMB} was not written"; exit 1; }
    echo "wrote ${EMB} ($(du -h "${EMB}" | cut -f1))"
fi

# --- stage (b): OvertonBench on those embeddings ----------------------------
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

echo "=== stage b1: generate ==="
python -m evaluation.overton.eval_overtonbench \
    --embeddings "${EMB}" --text_feat "${FEATS}" --dataset "${DATASET}" \
    --curvature "${CURV}" --tau "${TAU}" --seed "${SEED}" \
    --conditions "${CONDS}" --n_rollouts "${NROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --max_questions "${MAXQ}" --out "${OUT}" \
    || { echo "GENERATION FAILED (see .err)"; exit 1; }
echo "fork fallbacks (scout answers that got the baseline prompt):"
grep -c "scout returned 0 forks" logs/geom_ablation_${SLURM_JOB_ID}.err || true

echo "=== stage b2: judge ==="
python -m evaluation.overton.judge_overtonbench --score "${OUT}" \
    --max_users "${MAXU}" --k_rollouts "${KROLL}" \
    --base_url "http://localhost:${PORT}/v1" --model "${MODEL}" \
    --out "${SCORES}" \
    --dump_clusters "${SCORES%.csv}_clusters.csv" \
    || { echo "JUDGING FAILED (see .err)"; exit 1; }

# --- the other arm, copy-paste ----------------------------------------------
if [ "${CURV}" = "0" ]; then OTHER=0.5; else OTHER=0; fi
echo ""
echo "Done. Responses: ${OUT}  Scores: ${SCORES}"
echo "THIS IS HALF AN EXPERIMENT. The other arm:"
echo ""
echo "  CURV=${OTHER} MODEL=${MODEL} TAU=${TAU} SEED=${SEED} MAXQ=${MAXQ} MAXU=${MAXU} \\"
echo "      sbatch jobs/eval/job_geometry_ablation.sh"
echo ""
echo "READ THE DELTA, NOT THE ABSOLUTE SCORES: compare merge_v2 - baseline"
echo "across the two arms. Both arms share a generator and a judge, so the"
echo "delta is the only quantity the geometry moved. If the Euclidean delta"
echo "matches the hyperbolic one, curvature is not doing the work and MRR 0.456"
echo "was measuring fit, not usefulness."
