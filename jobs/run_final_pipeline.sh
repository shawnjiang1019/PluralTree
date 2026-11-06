#!/bin/bash
# Submit the whole final evaluation as one SLURM dependency chain, on a login node:
#
#   bash jobs/run_final_pipeline.sh
#   DRY=1 bash jobs/run_final_pipeline.sh          # print the jobs, submit nothing
#
# Order: validate judge -> train simple encoders -> retrieval diagnostic ->
# generate arms -> merge responses -> judge everything once. Each step starts only
# if the steps it needs succeeded (afterok), and finished generations are skipped.
#
# Knobs:
#   JUDGE_MODEL   default Qwen/Qwen2.5-72B-Instruct-AWQ
#   JUDGE_TP      GPUs for the judge (default 4; q38 uses 2)
#   JUDGE_TAG     file tag (default q25)
#   THINK_OFF=1   q38: send enable_thinking=false
#   VAL_OLD=1     also validate the old prompt (default on)
#
#   JUDGE_MODEL=<q38 path> JUDGE_TP=2 JUDGE_TAG=q38 THINK_OFF=1 bash jobs/run_final_pipeline.sh

set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_TAG="${JUDGE_TAG:-q25}"
THINK_OFF="${THINK_OFF:-0}"
VAL_OLD="${VAL_OLD:-1}"
DRY="${DRY:-0}"
FEATS="${FEATS:-feats_opinionqa.pt}"
FINAL="overton_responses_final.jsonl"

if [ "${THINK_OFF}" = "1" ]; then
    export PLURALTREE_CHAT_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
else
    unset PLURALTREE_CHAT_EXTRA
fi

N_SUB=0
sub() {   # sub <name> <sbatch args...>: prints the job id
    local name="$1"; shift
    if [ "${DRY}" = "1" ]; then
        N_SUB=$((N_SUB + 1))
        echo "  [dry] ${name}: sbatch --parsable $*" >&2
        echo "${name}_id"
    else
        local id
        id=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED: ${name}" >&2; exit 1; }
        echo "  ${name}: ${id}" >&2
        echo "${id}"
    fi
}

dep() {   # dep id1 id2 ...: the --dependency flag, or nothing with no ids
    local ids="" i
    for i in "$@"; do
        [ -n "${i}" ] && ids="${ids}:${i}"
    done
    [ -n "${ids}" ] && echo "--dependency=afterok${ids}"
}

have_rows() {   # have_rows file n: file exists with at least n rows
    [ -f "$1" ] && [ "$(grep -c . "$1")" -ge "$2" ]
}

for f in overton_responses_m80.jsonl overton_responses_all.jsonl \
         overton_responses_js.jsonl embeddings_opinionqa.pt \
         embeddings_opinionqa_tc1_ld03.pt "${FEATS}"; do
    [ -f "${f}" ] || { echo "MISSING ${f} in $(pwd)"; exit 1; }
done
echo "judge=${JUDGE_MODEL} tag=${JUDGE_TAG} tp=${JUDGE_TP} think_off=${THINK_OFF}"

# 1. Judge validation. Not a dependency of anything: read it, it does not gate.
sub validate_new --export=ALL,MODEL="${JUDGE_MODEL}",TP="${JUDGE_TP}",N=300,AGGU=0,JUDGE_PROMPT=benchmark \
    --gres=gpu:"${JUDGE_TP}" --time=12:00:00 jobs/eval/job_judge_validate.sh > /dev/null
if [ "${VAL_OLD}" = "1" ]; then
    sub validate_old --export=ALL,MODEL="${JUDGE_MODEL}",TP="${JUDGE_TP}",N=300,AGGU=5,JUDGE_PROMPT=ours \
        --gres=gpu:"${JUDGE_TP}" --time=06:00:00 jobs/eval/job_judge_validate.sh > /dev/null
fi

# 2. Simple encoders.
E_QGP=""; E_QG=""
[ -f embeddings_opinionqa_simple_qgp.pt ] || E_QGP=$(sub embed_qgp \
    --export=ALL,INPUTS=qgp,SPLIT=axis,EMB=embeddings_opinionqa_simple_qgp.pt,FEATS="${FEATS}" \
    jobs/embed/job_embed_subgroup.sh)
[ -f embeddings_opinionqa_simple_qg.pt ] || E_QG=$(sub embed_qg \
    --export=ALL,INPUTS=qg,SPLIT=item,EMB=embeddings_opinionqa_simple_qg.pt,FEATS="${FEATS}" \
    jobs/embed/job_embed_subgroup.sh)

# 3. Retrieval diagnostic on each embedding. Informational, does not gate.
sub diag $(dep "${E_QGP}" "${E_QG}") --export=ALL,FEATS="${FEATS}" jobs/eval/job_diag_embeddings.sh > /dev/null

# 4. Generation (180 rows per arm).
gen() {   # gen <name> <cond> <embedding> <out> [dependency ids]
    local name="$1" cond="$2" emb="$3" out="$4"; shift 4
    have_rows "${out}" 180 && { echo "  ${name}: ${out} complete, skipped" >&2; return; }
    sub "${name}" $(dep "$@") \
        --export=ALL,DATASET=opinionqa,EMB="${emb}",FEATS="${FEATS}",CONDS="${cond}",NROLL=3,SKIPJUDGE=1,OUT="${out}" \
        jobs/eval/job_overton_eval.sh
}
G_AXIS=$(gen gen_axis persona_axis embeddings_opinionqa.pt overton_responses_axis_minilm.jsonl)
G_LD03=$(gen gen_ld03 persona_merge embeddings_opinionqa_tc1_ld03.pt overton_responses_pm_ld03.jsonl)
G_QGP=$(gen gen_qgp persona_merge embeddings_opinionqa_simple_qgp.pt overton_responses_pm_qgp.jsonl "${E_QGP}")
G_QG=$(gen gen_qg persona_merge embeddings_opinionqa_simple_qg.pt overton_responses_pm_qg.jsonl "${E_QG}")

# 5. One responses file, arms renamed.
MERGE=$(sub merge $(dep "${G_AXIS}" "${G_LD03}" "${G_QGP}" "${G_QG}") \
    --account=def-enaskt --mem=8G --time=00:30:00 --output=logs/merge_final_%j.out \
    --wrap="bash jobs/eval/merge_final_responses.sh ${FINAL}")

# 6. Judge everything once, all raters.
JUDGE=$(sub judge --dependency=afterok:"${MERGE}" \
    --export=ALL,RESP="${FINAL}",SCORES=overton_scores_final_${JUDGE_TAG}.csv,ROLLOUTS=overton_rollouts_final_${JUDGE_TAG}.csv,MODEL="${JUDGE_MODEL}",TP="${JUDGE_TP}",MAXU=0,JUDGE_PROMPT=benchmark \
    --gres=gpu:"${JUDGE_TP}" --time=20:00:00 jobs/eval/job_judge_only.sh)

echo "submitted; watch with: squeue --me"
echo "scores -> overton_scores_final_${JUDGE_TAG}.csv"
[ "${DRY}" = "1" ] || echo "last job: ${JUDGE}"
