"""Judge our answers with the OvertonBench authors' own LLM-judge code.

Faithful to elinorp-d/overtonbench (vendored at third_party/overtonbench, commit
a3f221c): the prompt is built by THEIR helper_functions.format_prompt with
THEIR "fs+fr" template (the participant's free response + their ratings of every
other response to the same question, untruncated), sent by THEIR
llm_api.generate (system prompt prepended, temperature 0, their digit parsing).
Two things are substituted, at the API call only, and both are recorded:

  model     their code hard-codes gemini-2.5-pro; --model picks the judge
            (gemini-2.5-flash: the authors report it comparable to Pro)
  thinking  their call sets no thinking_config (model default: on);
            --thinking_budget 0 turns it off, -1 leaves their default

Our answers never enter the few-shot pool: examples come from the original
human ratings only, so a participant sees the benchmark's own responses with
their real ratings, never another arm's answer with a missing rating.

Protocol (Poole-Dayan et al. 2026, Sec. 5-6 and App. C.1): every rater on the
question (about 60; --users_per_q samples fewer), --source modelslant for their
15 questions (the set their judge scores, Table 14; "full" = all 60), and three
predictions per rating, averaged and rounded (--repeats 3).
--with_references also re-rates the benchmark's own model responses for the same
participants, so every "model" in the scoring is rated by the same judge.

Output is a CSV in their schema (user, question_id, model, representation_rating
= the predicted rating, cluster_kmeans, ...), scored with THEIR script:

    python third_party/overtonbench/src/benchmark_overton_pipeline.py --data ob_gemini.csv --outdir ob_scores
    python third_party/overtonbench/src/benchmark_overton_pipeline.py --data ob_gemini.csv --outdir ob_scores --weighted

Run where Gemini is reachable (not a narval compute node). Key from the
environment only (GEMINI_API_KEY or GOOGLE_API_KEY):

    python -m evaluation.overton.overtonbench_gemini --responses overton_responses_all.jsonl --conditions baseline,persona_merge --source modelslant --repeats 3 --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OB_DIR = os.path.join(ROOT, "third_party", "overtonbench", "src")
OUT_COLS = ["user", "question_id", "model", "representation_rating",
            "cluster_kmeans", "question", "rollout", "source"]


class _JudgeClient:
    """Wraps their Gemini client: swaps the model and thinking budget, keeps
    everything else (contents, temperature) exactly as their code sent it, and
    tallies token usage for the cost estimate."""

    def __init__(self, inner, model: str, thinking_budget: int):
        self._inner, self._model, self._budget = inner, model, thinking_budget
        self.models = self
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "prompt": 0, "output": 0, "thoughts": 0}

    def generate_content(self, model, contents, config=None):
        from google.genai import types
        if self._budget >= 0:
            config = types.GenerateContentConfig(
                temperature=getattr(config, "temperature", 0),
                thinking_config=types.ThinkingConfig(thinking_budget=self._budget))
        resp = self._inner.models.generate_content(
            model=self._model, contents=contents, config=config)
        um = getattr(resp, "usage_metadata", None)
        with self.lock:
            self.usage["calls"] += 1
            if um is not None:
                self.usage["prompt"] += um.prompt_token_count or 0
                self.usage["output"] += um.candidates_token_count or 0
                self.usage["thoughts"] += getattr(um, "thoughts_token_count", 0) or 0
        return resp

    def close(self):
        if hasattr(self._inner, "close"):
            self._inner.close()


def _panel(df, users_per_q: int, seed: int) -> dict[int, list]:
    """Participants per question: raters with a cluster label, fixed seed."""
    labelled = df.dropna(subset=["cluster_kmeans"])
    rng = random.Random(seed)
    out = {}
    for qid in sorted(labelled["question_id"].unique()):
        users = sorted(labelled.loc[labelled["question_id"] == qid, "user"].unique())
        if users_per_q and len(users) > users_per_q:
            users = sorted(rng.sample(users, users_per_q))
        out[int(qid)] = users
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="OvertonBench authors' judge on our answers")
    ap.add_argument("--responses", nargs="+", required=True, help="our responses .jsonl file(s)")
    ap.add_argument("--conditions", default="", help="comma list; empty = all in the files")
    ap.add_argument("--rollouts", default="", help="comma list of rollout ids; empty = all")
    ap.add_argument("--users_per_q", type=int, default=0, help="0 = every rater (their protocol)")
    ap.add_argument("--source", default="full", choices=["full", "modelslant", "prism"],
                    help="question split; modelslant = the 15 questions of App. C.1")
    ap.add_argument("--repeats", type=int, default=1,
                    help="predictions per rating, averaged and rounded (their protocol: 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_references", action="store_true",
                    help="also re-rate the benchmark's own model responses")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--thinking_budget", type=int, default=0,
                    help="0 = thinking off; -1 = model default (their setting)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N rows (cost test)")
    ap.add_argument("--out", default="ob_gemini.csv")
    ap.add_argument("--dry_run", action="store_true", help="print one prompt; no API calls")
    args = ap.parse_args()

    sys.path.insert(0, OB_DIR)
    sys.path.insert(0, os.path.join(OB_DIR, "prompting_pipeline"))
    import helper_functions
    import prompts
    from load_dataset import get_overtonbench_data

    helper_functions.set_data_options(path=None, source_split=args.source)
    df = get_overtonbench_data(source_split=args.source)
    meta = df.drop_duplicates(["user", "question_id"]).set_index(["user", "question_id"])
    panel = _panel(df, args.users_per_q, args.seed)

    conds = {c.strip() for c in args.conditions.split(",") if c.strip()}
    rolls = {int(r) for r in args.rollouts.split(",") if r.strip()}
    answers = []                                 # (question_id, model, rollout, text)
    for path in args.responses:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if conds and r["condition"] not in conds:
                    continue
                if rolls and r.get("rollout", 0) not in rolls:
                    continue
                answers.append((int(r["question_id"]), r["condition"],
                                r.get("rollout", 0), r["response"]))

    targets = []
    for qid, cond, roll, text in answers:
        for u in panel.get(qid, []):
            m = meta.loc[(u, qid)]
            targets.append({"user": u, "question_id": qid, "question": m["question"],
                            "freeresponse": m.get("freeresponse", ""),
                            "llm_response": text, "cluster_kmeans": m["cluster_kmeans"],
                            "model": f"{cond}_r{roll}", "rollout": roll, "source": "ours"})
    if args.with_references:
        ref = df[df.apply(lambda x: x["user"] in set(panel.get(int(x["question_id"]), [])),
                          axis=1)]
        for x in ref.to_dict(orient="records"):
            targets.append({**x, "rollout": 0, "source": "reference"})
    if args.limit:
        targets = targets[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["user"], str(r["question_id"]), r["model"]))
    todo = [t for t in targets
            if (str(t["user"]), str(t["question_id"]), t["model"]) not in done]
    n_models = len({t["model"] for t in targets})
    print(f"{len(answers)} answers x panel -> {len(targets)} judge rows "
          f"({n_models} models, {sum(len(v) for v in panel.values())} participants "
          f"over {len(panel)} questions, source={args.source}); {len(todo)} to do "
          f"x {args.repeats} repeats; judge={args.model} "
          f"thinking_budget={args.thinking_budget}")

    if args.dry_run:
        if todo:
            p = helper_functions.format_prompt("freeresponse_fewshot", todo[0])
            print("\n--- system ---\n" + prompts.system_prompt)
            print("\n--- user (their fs+fr template) ---\n" + p)
            print(f"\n~{(len(prompts.system_prompt) + len(p)) // 4} tokens (chars/4)")
        return

    import llm_api
    judge = _JudgeClient(llm_api._get_gemini_client(), args.model, args.thinking_budget)
    llm_api._gemini_client = judge               # their generate() now calls through us

    def rate_once(p):
        for attempt in range(4):                 # outer backoff for 429/5xx bursts
            r = llm_api.generate(preformatted_prompt=p, sys_prompt=prompts.system_prompt,
                                 model_name="gemini", max_retries=3, temperature=0)
            if r is not None:
                return r
            time.sleep(5 * 2 ** attempt)
        return None

    def rate(t):
        p = helper_functions.format_prompt("freeresponse_fewshot", t)
        rs = [r for r in (rate_once(p) for _ in range(args.repeats)) if r is not None]
        if len(rs) < args.repeats:               # never average a partial set
            return t, None
        return t, int(sum(rs) / len(rs) + 0.5)   # rounded average, half up

    new = not os.path.exists(args.out)
    n_ok = n_fail = 0
    t0 = time.time()
    with open(args.out, "a", newline="", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(rate, t) for t in todo]), 1):
            t, r = fut.result()
            if r is None:
                n_fail += 1
                continue
            n_ok += 1
            w.writerow({**t, "representation_rating": r})
            if k % 200 == 0:
                f.flush()
                u = judge.usage
                print(f"  {k}/{len(todo)}  ok {n_ok} fail {n_fail}  "
                      f"tokens/call in {u['prompt'] / max(1, u['calls']):.0f} "
                      f"out {u['output'] / max(1, u['calls']):.0f} "
                      f"think {u['thoughts'] / max(1, u['calls']):.0f}  "
                      f"{(time.time() - t0) / 60:.1f} min")
    judge.close()
    u = judge.usage
    print(f"done: {n_ok} rated, {n_fail} unrated ({n_fail / max(1, len(todo)):.1%}) -> {args.out}")
    print(f"tokens: prompt {u['prompt']}, output {u['output']}, thinking {u['thoughts']} "
          f"over {u['calls']} calls")
    if n_fail / max(1, len(todo)) > 0.05:
        print("WARNING: >5% of rows unrated -- rerun the same command to retry them "
              "(rated rows are skipped) before scoring.")


if __name__ == "__main__":
    main()
