"""GRPO alignment trainer entrypoint. See docs/grpo_alignment.txt.

SCAFFOLD: the reward (alignment/reward.py), advantage (alignment/advantage.py),
and prompt dataset (alignment/rollout_dataset.py) are implemented and tested; the
policy-update loop here wires them into TRL's GRPOTrainer (LoRA policy, our
reward_funcs, group_size). trl/peft are imported lazily so the pure pieces stay
offline-testable.

Two entrypoints:
  --dry_run   build the real prompt dataset, generate fake rollouts, and score
              the reward + group-relative advantage on them. No GPU, no trl; use
              a hashing stub embedder (--stub_embed, default in dry_run) so it
              needs no model download. Verifies the reward pipeline end to end.
  (default)   full GRPO run: lazy trl.GRPOTrainer over the LoRA policy.

    python -m alignment.train_grpo --embeddings embeddings_opinionqa.pt \
        --text_feat feats_opinionqa.pt --dry_run
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment.grpo_config import GRPOAlignConfig
from alignment.reward import Position, RewardConfig, coverage_reward


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _hashing_embed_fn(dim: int = 256, seed: int = 0):
    """Deterministic bag-of-words embedder -- no model download. Shared vocabulary
    across calls so cosine reflects word overlap. For --dry_run / tests only."""
    rng = np.random.default_rng(seed)
    vocab: dict[str, int] = {}

    def _embed(texts):
        V = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", (t or "").lower()):
                V[i, vocab.setdefault(w, int(rng.integers(0, dim)))] += 1.0
            n = np.linalg.norm(V[i])
            if n:
                V[i] /= n
        return V
    return _embed


def _positions_of(row: dict) -> list[Position]:
    return [Position(**p) for p in row["positions"]]


def _completion_text(c) -> str:
    """TRL hands completions as a string or [{'role','content'}]; take the text,
    then keep only the <answer> span (the judge/reward never sees <think>)."""
    from retrieval.answer import extract_answer
    if isinstance(c, list):
        c = "".join(m.get("content", "") for m in c if isinstance(m, dict))
    ans, _ = extract_answer(c)
    return ans


def build_dataset(args, cfg: GRPOAlignConfig):
    """Materialize the scout-injected prompt records (needs the graph + embeddings)."""
    import torch
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import PLURALISM_ROUTE
    from retrieval.scout import ScoutConfig, load_or_compute_text_feat
    from alignment.rollout_dataset import build_prompts, load_eval_holdout_texts

    if args.dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=cfg.seed, leakage_safe=True)
    else:
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=cfg.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)
    holdout = set() if args.no_holdout else load_eval_holdout_texts()

    recs = build_prompts(graph, h_all, text_feat, manifold,
                         cfg=ScoutConfig(tau=cfg.tau, alpha=cfg.alpha),
                         instruction=PLURALISM_ROUTE,
                         eval_holdout_texts=holdout,
                         max_questions=cfg.prompts_max)
    print(f"built {len(recs)} training prompts "
          f"(eval holdout: {'off' if args.no_holdout else len(holdout)} questions)")
    return recs


def make_reward_func(embed_fn, rcfg: RewardConfig):
    """A TRL-compatible reward_func closed over the embedder + reward config.

    TRL calls reward_func(prompts=..., completions=..., **columns); the dataset's
    ``positions`` column arrives as a per-example list.
    """
    def reward_func(completions, positions=None, **_):
        rewards = []
        for c, pos in zip(completions, positions):
            plist = [Position(**p) for p in pos]
            r, _bd = coverage_reward(_completion_text(c), plist, embed_fn, rcfg)
            rewards.append(r)
        return rewards
    return reward_func


# ---------------------------------------------------------------------------
# --dry_run: exercise reward + advantage on real prompts, no GPU / no trl
# ---------------------------------------------------------------------------
def dry_run(args, cfg: GRPOAlignConfig):
    from alignment.advantage import group_relative_advantage

    recs = build_dataset(args, cfg)
    if not recs:
        print("no prompts built — check embeddings / graph / holdout"); return
    embed_fn = _hashing_embed_fn() if args.stub_embed else \
        __import__("alignment.reward", fromlist=["default_embed_fn"]).default_embed_fn(cfg.reward_embedder)
    rcfg = RewardConfig(match_thr=cfg.match_thr, l_precision=cfg.l_precision,
                        l_verbose=cfg.l_verbose,
                        min_depth_words=cfg.min_depth_words, weight=cfg.weight)

    print(f"\nscoring {min(3, len(recs))} prompts x group_size={cfg.group_size} "
          f"fake rollouts ({'stub' if args.stub_embed else cfg.reward_embedder} embedder):\n")
    for rec in recs[:3]:
        positions = _positions_of(rec)
        top = ", ".join(f"{p.option}={p.prevalence:.2f}" for p in positions[:4])
        # Fake rollouts spanning the axis the reward now scores: DEPTH per
        # position, not just how many are named. `shallow` is the `route` pattern
        # (every position named, none articulated) that scored 0.072 on
        # OvertonBench -- it must score ~0 here, or the reward is still blind.
        def _para(p, n_words):
            """n_words of text about p, built from p's OWN phrasing so the stub
            bag-of-words embedder can match it (a real embedder matches on
            meaning; the stub needs lexical overlap). Not realistic prose -- this
            is a plumbing smoke test, not a generation sample."""
            w = p.embed_text.split() or [p.option]
            return " ".join((w * (n_words // len(w) + 1))[:n_words])

        d = rcfg.min_depth_words
        shallow = " ".join(f"Some say {p.option}." for p in positions)
        deep_narrow = _para(positions[0], 2 * d)
        deep_broad = "\n".join(_para(p, 2 * d) for p in positions[:3])
        labels = ["shallow(route-like)", "deep-narrow", "deep-broad"]
        rollouts = ([shallow, deep_narrow, deep_broad] * cfg.group_size)[:cfg.group_size]
        scored = [coverage_reward(r, positions, embed_fn, rcfg) for r in rollouts]
        rewards = [s[0] for s in scored]
        adv = group_relative_advantage(rewards)
        print(f"  q{rec['question_id']}: {rec['question'][:70]}")
        print(f"    positions[{len(positions)}]: {top}")
        for lab, (rw, bd), a in zip(labels, scored[:3], adv[:3]):
            print(f"    {lab:<20} reward={rw:.3f} adv={a:+.3f}  "
                  f"mentioned={bd['n_mentioned']} covered={bd['n_expressed']} "
                  f"depth={bd['mean_depth']:.0f}w")
        print()
    print("dry run OK — reward + advantage pipeline runs on real prompts.")


# ---------------------------------------------------------------------------
# Full GRPO run (lazy trl / peft)
# ---------------------------------------------------------------------------
def train(args, cfg: GRPOAlignConfig):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from alignment.reward import default_embed_fn

    recs = build_dataset(args, cfg)
    if not recs:
        raise SystemExit("no prompts built — aborting")
    ds = Dataset.from_list([{"prompt": r["prompt"], "positions": r["positions"]}
                            for r in recs])

    embed_fn = default_embed_fn(cfg.reward_embedder)
    rcfg = RewardConfig(match_thr=cfg.match_thr, l_precision=cfg.l_precision,
                        l_verbose=cfg.l_verbose,
                        min_depth_words=cfg.min_depth_words, weight=cfg.weight)
    reward_func = make_reward_func(embed_fn, rcfg)

    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, task_type="CAUSAL_LM",
                      target_modules="all-linear")

    grpo = GRPOConfig(
        output_dir=cfg.save_dir,
        learning_rate=cfg.lr,
        beta=cfg.kl_coef,
        num_generations=cfg.group_size,
        per_device_train_batch_size=cfg.batch_prompts,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.n_epochs,
        max_steps=cfg.max_steps or -1,
        max_prompt_length=cfg.max_prompt_tokens,
        max_completion_length=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        logging_steps=cfg.log_every,
        bf16=True,
        # TODO(scale): enable use_vllm=True for fast colocated rollouts once the
        # cluster vllm build is confirmed against this trl version.
    )

    trainer = GRPOTrainer(
        model=cfg.base_model,
        reward_funcs=reward_func,
        args=grpo,
        train_dataset=ds,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(cfg.save_dir)
    print(f"saved LoRA adapter -> {cfg.save_dir}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GRPO alignment phase")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--dataset", choices=["opinionqa", "globalopinionqa"], default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--dry_run", action="store_true",
                    help="build prompts + score reward/advantage on fake rollouts; no GPU/trl")
    ap.add_argument("--stub_embed", action="store_true",
                    help="dry_run: hashing embedder (no model download)")
    ap.add_argument("--no_holdout", action="store_true",
                    help="skip the OvertonBench leakage guard (NOT for real runs)")
    # config overrides
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--group_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--kl_coef", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--prompts_max", type=int, default=None)
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    cfg = GRPOAlignConfig()
    for k in ("base_model", "group_size", "lr", "kl_coef", "max_steps",
              "prompts_max", "save_dir"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.dry_run:
        args.stub_embed = args.stub_embed or True     # default stub in dry_run
        dry_run(args, cfg)
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()
