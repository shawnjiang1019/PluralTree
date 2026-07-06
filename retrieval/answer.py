"""Inject scout forks into an LLM prompt and generate a pluralistic answer.

The scout returns ScoredForks (relevant AND divergent branch pairs); this
module turns them into prompt context and calls a generator LLM. Opinion
leaves are expanded into their actual answer distributions — that is the
evidence the LLM reasons over, not node names. See docs/overtonbench_eval.txt.

The LLM client speaks the OpenAI chat-completions protocol over stdlib urllib,
so any vLLM/OpenAI-compatible endpoint works with zero extra dependencies.

Usage:
    python -m retrieval.answer --question "..." \
        --embeddings embeddings_goqa.pt --curvature 0.5 --text_feat feats_goqa.pt \
        --condition scout --base_url http://localhost:8000/v1 --model Qwen/... \
        [--dry_run]
"""

from __future__ import annotations

import json
import os
import urllib.request

from retrieval.scout import ScoredFork, ScoutConfig, describe_node, scout

# Condition -> ScoutConfig overrides. baseline = no retrieval at all;
# div_only ablates the relevance guards (old pure-divergence scout).
CONDITIONS: dict[str, ScoutConfig | None] = {
    "baseline": None,
    "scout": ScoutConfig(tau=0.25, alpha=1.0),
    "div_only": ScoutConfig(tau=0.0, alpha=0.0),
}

BASELINE_INSTRUCTION = (
    "Answer the question thoughtfully and concisely."
)

PLURALISM_INSTRUCTION = (
    "The context below contains divergent perspectives retrieved from a "
    "knowledge graph of survey data. Different branches represent groups that "
    "genuinely disagree. When answering: (1) represent each perspective "
    "faithfully and specifically, (2) attribute claims to the group holding "
    "them, (3) do NOT average the disagreement away into a consensus, "
    "(4) where perspectives conflict, say so explicitly."
)


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible, stdlib only)
# ---------------------------------------------------------------------------
def chat(base_url: str, model: str, messages: list[dict], *,
         temperature: float = 0.0, max_tokens: int = 1024,
         timeout: float = 120.0) -> str:
    """One chat-completions call against a vLLM/OpenAI-compatible endpoint."""
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Fork -> prompt context (opinion leaves expanded to distributions)
# ---------------------------------------------------------------------------
def fork_context(fork: ScoredFork, graph, k: int = 1) -> str:
    """One fork as a perspective-contrast block."""
    lines = [f"[fork {k}] at '{describe_node(graph, fork.anchor, 60)}' "
             f"(divergence={fork.w:.2f}, relevance={fork.relevance:.2f}):",
             f"  Perspective A ({describe_node(graph, fork.branch_a, 40)}):",
             f"  Perspective B ({describe_node(graph, fork.branch_b, 40)}):"]
    for na, nb, _ in fork.top_pairs:
        lines.append(f"    A: {describe_node(graph, na, 0, show_q=True)}")
        lines.append(f"    B: {describe_node(graph, nb, 0, show_q=True)}")
    return "\n".join(lines)


def build_prompt(question: str, forks: list[ScoredFork] | None, graph) -> list[dict]:
    """Chat messages: pluralism instruction + fork blocks + question (last)."""
    if not forks:
        return [{"role": "system", "content": BASELINE_INSTRUCTION},
                {"role": "user", "content": question}]
    ctx = "\n\n".join(fork_context(f, graph, k) for k, f in enumerate(forks, 1))
    return [{"role": "system", "content": PLURALISM_INSTRUCTION},
            {"role": "user", "content": ctx + "\n\nQuestion: " + question}]


def answer(question: str, condition: str, *, graph=None, h_all=None,
           text_feat=None, manifold=None, base_url: str = "",
           model: str = "", dry_run: bool = False, q_emb=None,
           cfg: ScoutConfig | None = None) -> str:
    """Generate one answer under a condition; returns the prompt if dry_run.

    ``cfg`` overrides the condition's ScoutConfig (e.g. a recalibrated tau).
    """
    import sys

    cfg = cfg if cfg is not None else CONDITIONS[condition]
    forks = None
    if cfg is not None:
        forks = scout(question, graph, h_all, text_feat, manifold,
                      cfg=cfg, q_emb=q_emb)
        if not forks:
            print(f"warning: scout returned 0 forks (tau={cfg.tau}) — "
                  f"baseline prompt used for: {question[:60]}", file=sys.stderr)
    messages = build_prompt(question, forks, graph)
    if dry_run:
        return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
    return chat(base_url, model, messages, temperature=0.7)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Scout-injected answer generation")
    ap.add_argument("--question", required=True)
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="scout")
    ap.add_argument("--embeddings", default=None, help=".pt of h_all on the ball")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the condition's relevance gate")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the condition's relevance exponent")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the assembled prompt instead of calling the LLM")
    args = ap.parse_args()

    cfg = None
    if args.condition != "baseline" and (args.tau is not None or args.alpha is not None):
        base = CONDITIONS[args.condition]
        cfg = ScoutConfig(tau=args.tau if args.tau is not None else base.tau,
                          alpha=args.alpha if args.alpha is not None else base.alpha)

    graph = h_all = text_feat = manifold = None
    if args.condition != "baseline":
        import torch
        from pluraltree.manifolds.poincare import PoincareBall
        from data.globalopinionqa import load_globalopinionqa
        from retrieval.scout import load_or_compute_text_feat

        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
        h_all = torch.load(args.embeddings, map_location="cpu")
        if not isinstance(h_all, torch.Tensor):
            h_all = h_all["h_all"]
        manifold = PoincareBall(c=args.curvature)
        text_feat = load_or_compute_text_feat(graph, "globalopinionqa", args.text_feat)

    print(answer(args.question, args.condition, graph=graph, h_all=h_all,
                 text_feat=text_feat, manifold=manifold, base_url=args.base_url,
                 model=args.model, dry_run=args.dry_run, cfg=cfg))


if __name__ == "__main__":
    _main()
