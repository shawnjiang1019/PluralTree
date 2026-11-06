#!/bin/bash
# Build the single responses file for the final judge pass: the re-judged arms,
# persona_self, persona_js, persona_axis, and persona_merge under each embedding,
# renamed so the arms stay distinct. Fails, rather than writing a partial file, if
# an input is missing or an arm does not have 180 rows.
#
#   bash jobs/eval/merge_final_responses.sh overton_responses_final.jsonl

module load python/3.11 2>/dev/null || true     # batch shells start without python

set -eu
OUT="${1:-overton_responses_final.jsonl}"
TMP="${OUT}.tmp"

for f in overton_responses_m80.jsonl overton_responses_all.jsonl overton_responses_js.jsonl \
         overton_responses_axis_minilm.jsonl overton_responses_pm_ld03.jsonl \
         overton_responses_pm_qgp.jsonl overton_responses_pm_qg.jsonl; do
    [ -f "${f}" ] || { echo "MISSING ${f}"; exit 1; }
done

cp overton_responses_m80.jsonl "${TMP}"
grep '"condition": "persona_self"' overton_responses_all.jsonl >> "${TMP}"
cat overton_responses_js.jsonl overton_responses_axis_minilm.jsonl >> "${TMP}"
sed 's/"condition": "persona_merge"/"condition": "pm_ld03"/' overton_responses_pm_ld03.jsonl >> "${TMP}"
sed 's/"condition": "persona_merge"/"condition": "pm_simple_qgp"/' overton_responses_pm_qgp.jsonl >> "${TMP}"
sed 's/"condition": "persona_merge"/"condition": "pm_simple_qg"/' overton_responses_pm_qg.jsonl >> "${TMP}"

python - "${TMP}" <<'EOF'
import collections, json, sys
n = collections.Counter(json.loads(l)["condition"] for l in open(sys.argv[1], encoding="utf-8"))
print(dict(sorted(n.items())))
bad = {c: k for c, k in n.items() if k != 180}
if bad:
    sys.exit(f"arms without exactly 180 rows: {bad}")
EOF

mv "${TMP}" "${OUT}"
echo "Done -> ${OUT}"
