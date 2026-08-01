#!/bin/bash
# Submit the GRPO alignment phase as a dependency DAG. Run from the repo root on
# the login node (this is NOT an sbatch script -- it only calls sbatch):
#
#     bash jobs/submit_grpo_phase.sh              # gates + independent evals
#     TRAIN=1 bash jobs/submit_grpo_phase.sh      # ...and chain training + eval
#
# The DAG (see the staged plan in docs/):
#
#   [A] job_reward_correlation.sh   ──┐   Stage 0  : can the reward RANK?
#   [A] job_reward_ceiling.sh       ──┤   Stage 0b : does the policy have HEADROOM?
#   [A] overton CONDS=distributional  │   (independent; never evaluated)
#   [A] job_g2_diversity.sh           │   (independent branch)
#                                     │
#                                     └─> [B] smoke (DRY + 20 steps)
#                                             └─> [C] arms: scout | plain | v1
#                                                     └─> [D] per-arm eval
#
# Stage 0 and 0b are independent measurements -- both gate, neither feeds the
# other, so they run concurrently. B/C/D chain with `--dependency=afterok`, which
# is a REAL gate: job_reward_correlation.sh exits 2 when within-question
# concordance is below GATE, so a chance-level reward cannot launch a training
# run. Stages whose scripts do not exist yet are skipped with a notice.

set -uo pipefail
mkdir -p logs docs

VERSIONS="${VERSIONS:-v6 v5}"
GATE="${GATE:-0.60}"
ARMS="${ARMS:-scout plain v1}"
TRAIN="${TRAIN:-0}"

pending=()
note () { echo "  SKIP  $1"; pending+=("$1"); }

# sub <var-name> <script> [extra sbatch args...] -> echoes job id, sets var
sub () {
    local _var="$1" _script="$2"; shift 2
    if [ ! -f "${_script}" ]; then note "${_script} (not written yet)"; return 1; fi
    local _id
    _id=$(sbatch --parsable "$@" "${_script}") || { echo "  FAIL  ${_script}"; return 1; }
    printf -v "${_var}" '%s' "${_id}"
    echo "  ${_id}  ${_script} $*"
    return 0
}

echo "=== [A] gates + independent evals (parallel) ==================="

JID_CORR=""
sub JID_CORR jobs/eval/job_reward_correlation.sh \
    --export=ALL,VERSIONS="${VERSIONS}",GATE="${GATE}"

JID_CEIL=""
sub JID_CEIL jobs/eval/job_reward_ceiling.sh --export=ALL

# Built but never evaluated -- unrelated to the GRPO chain, just idle capacity.
sub _J_DIST jobs/eval/job_overton_eval.sh \
    --job-name=overton_distributional \
    --output=logs/overton_distributional_%j.out \
    --error=logs/overton_distributional_%j.err \
    --export=ALL,CONDS=distributional,OUT=overton_responses_dist.jsonl,SCORES=overton_scores_dist.csv

# Independent branch; needs Qwen2.5-7B-Instruct + mpnet already in HF_HOME.
sub _J_G2 jobs/eval/job_g2_diversity.sh --export=ALL

if [ "${TRAIN}" != "1" ]; then
    echo ""
    echo "Stage A submitted. Training NOT chained (TRAIN=1 to include it)."
    echo "Read the gate first:  grep -h 'GATE:' logs/reward_corr_*.out"
    [ ${#pending[@]} -gt 0 ] && printf 'Still to write: %s\n' "${pending[*]}"
    echo "Track: squeue -u \$USER"
    exit 0
fi

# --- B/C/D: only reachable behind BOTH gates ---------------------------------
deps=""
[ -n "${JID_CORR}" ] && deps="${deps}:${JID_CORR}"
[ -n "${JID_CEIL}" ] && deps="${deps}:${JID_CEIL}"
if [ -z "${deps}" ]; then
    echo ""
    echo "ABORT: neither gate was submitted -- refusing to chain training."; exit 1
fi

echo ""
echo "=== [B] smoke (afterok on the gates) ==========================="
JID_SMOKE=""
sub JID_SMOKE jobs/train/job_grpo_align.sh \
    --dependency="afterok${deps}" --job-name=grpo_smoke \
    --output=logs/grpo_smoke_%j.out --error=logs/grpo_smoke_%j.err \
    --time=00:40:00 --export=ALL,MAXSTEPS=20,PROMPTS=32,OUT=grpo_lora_smoke
[ -z "${JID_SMOKE}" ] && { echo "ABORT: smoke not submitted."; exit 1; }

echo ""
echo "=== [C] arms (afterok on smoke) + [D] per-arm eval ============="
for ARM in ${ARMS}; do
    JID_ARM=""
    sub JID_ARM jobs/train/job_grpo_align.sh \
        --dependency="afterok:${JID_SMOKE}" --job-name="grpo_${ARM}" \
        --output="logs/grpo_${ARM}_%j.out" --error="logs/grpo_${ARM}_%j.err" \
        --export=ALL,ARM="${ARM}",OUT="grpo_lora_${ARM}" || continue

    # Both evals depend only on THIS arm, and are parallel with each other.
    sub _J_OV jobs/eval/job_grpo_eval.sh \
        --dependency="afterok:${JID_ARM}" --job-name="grpo_eval_${ARM}" \
        --output="logs/grpo_eval_${ARM}_%j.out" \
        --error="logs/grpo_eval_${ARM}_%j.err" \
        --export=ALL,ARM="${ARM}",ADAPTER="grpo_lora_${ARM}"
    sub _J_HM jobs/eval/job_hivemind_metrics.sh \
        --dependency="afterok:${JID_ARM}" --job-name="hivemind_${ARM}" \
        --output="logs/hivemind_${ARM}_%j.out" \
        --error="logs/hivemind_${ARM}_%j.err" \
        --export=ALL,ARM="${ARM}",ADAPTER="grpo_lora_${ARM}"
done

echo ""
if [ ${#pending[@]} -gt 0 ]; then
    printf 'Still to write (their stages were skipped): %s\n' "${pending[*]}"
fi
echo "Track: squeue -u \$USER    Gate verdict: grep -h 'GATE:' logs/reward_corr_*.out"
echo "A gate FAIL (exit 2) cancels every dependent job -- that is intended."
