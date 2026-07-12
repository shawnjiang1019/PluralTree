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
import re
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

# Think/answer separation: the retrieved forks may be off-topic, and an
# instruction that FORCES them into the answer derails it (measured: coverage
# 0.51 -> 0.06 on OvertonBench when GOQA forks were mandatory). The model
# triages relevance inside <think>; only the <answer> span is shown/judged,
# so irrelevant retrieval fails soft (~baseline) instead of catastrophically.
PLURALISM_INSTRUCTION = (
    "You will see context retrieved from a knowledge graph of survey data, "
    "followed by a question. The context may or may not be relevant to the "
    "question.\n"
    "First, inside <think></think> tags, BRIEFLY (a few sentences) assess "
    "which retrieved perspectives (if any) actually bear on the question, "
    "and discard the irrelevant ones.\n"
    "Then, inside <answer></answer> tags, answer the question directly and "
    "thoughtfully, covering the range of positions people genuinely hold on "
    "it. If relevant perspectives were retrieved, represent them faithfully, "
    "attribute them to the groups holding them, and do not average real "
    "disagreement into a consensus. If none are relevant, ignore the context "
    "entirely and answer as if it were not provided.\n"
    "The reader sees ONLY what is inside the <answer> tags."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN_RE = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_SPAN_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>(.*?)(?:<answer>|$)", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> tuple[str, bool]:
    """The <answer> span (what the reader/judge sees), and whether tags held.

    An unclosed <answer> means generation hit max_tokens mid-answer — keep
    everything after the opening tag (truncated but real). Final fallback:
    strip any <think> block and stray tag literals, so a reasoning dump or
    tag fragment never reaches the judge whole.
    """
    m = _ANSWER_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip(), True
    m = _ANSWER_OPEN_RE.search(text)                 # truncated before </answer>
    if m and m.group(1).strip():
        return m.group(1).strip(), True
    rest = _THINK_RE.sub("", text)
    rest = re.sub(r"</?(answer|think)>", "", rest, flags=re.IGNORECASE).strip()
    return (rest or text.strip()), False


def extract_think(text: str) -> str:
    """The <think> span — the model's relevance-triage trace ('' if none).

    An unclosed <think> (generation truncated, or the model jumped straight to
    <answer> without closing) still yields the reasoning up to <answer>/EOS, so
    traces stay observable even for malformed generations.
    """
    m = _THINK_SPAN_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _THINK_OPEN_RE.search(text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible, stdlib only)
# ---------------------------------------------------------------------------
def chat(base_url: str, model: str, messages: list[dict], *,
         temperature: float = 0.0, max_tokens: int = 1024,
         top_p: float | None = None, timeout: float = 120.0) -> str:
    """One chat-completions call against a vLLM/OpenAI-compatible endpoint."""
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if top_p is not None:
        payload["top_p"] = top_p
    body = json.dumps(payload).encode()
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


def forks_to_context(forks: list[ScoredFork], graph) -> str:
    """All fork blocks joined — exactly the context string the LLM sees."""
    return "\n\n".join(fork_context(f, graph, k) for k, f in enumerate(forks, 1))


def build_prompt(question: str, forks: list[ScoredFork] | None, graph) -> list[dict]:
    """Chat messages: pluralism instruction + fork blocks + question (last)."""
    if not forks:
        return [{"role": "system", "content": BASELINE_INSTRUCTION},
                {"role": "user", "content": question}]
    ctx = forks_to_context(forks, graph)
    return [{"role": "system", "content": PLURALISM_INSTRUCTION},
            {"role": "user", "content": ctx + "\n\nQuestion: " + question}]


def answer(question: str, condition: str, *, graph=None, h_all=None,
           text_feat=None, manifold=None, base_url: str = "",
           model: str = "", dry_run: bool = False, q_emb=None,
           cfg: ScoutConfig | None = None, with_raw: bool = False,
           with_trace: bool = False):
    """Generate one answer under a condition; returns the prompt if dry_run.

    ``cfg`` overrides the condition's ScoutConfig (e.g. a recalibrated tau).
    ``with_raw=True`` returns ``(answer, raw_generation)`` so the <think>
    triage trace is inspectable (raw == answer when there was no tagging).
    ``with_trace=True`` (supersedes with_raw) returns ``(answer, trace)`` with
    the COMPLETE reasoning record for the row::

        {"raw": full generation, "think": extracted <think> span,
         "fork_context": the injected fork blocks ('' for baseline/no-forks),
         "n_forks": how many forks were injected}

    so retrieval -> triage -> answer is observable end to end.
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
    ctx = forks_to_context(forks, graph) if forks else ""

    def _pack(ans: str, raw: str):
        if with_trace:
            return ans, {"raw": raw, "think": extract_think(raw),
                         "fork_context": ctx, "n_forks": len(forks or [])}
        return (ans, raw) if with_raw else ans

    if dry_run:
        prompt = "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
        return _pack(prompt, prompt)
    if not forks:
        text = chat(base_url, model, messages, temperature=0.7)
        return _pack(text, text)
    # Injected conditions think before answering — budget for both spans,
    # then keep only the <answer> span (the judge must not see the triage).
    # 4096: at 2048, ~58% of answers lost their closing tag to the token cap.
    text = chat(base_url, model, messages, temperature=0.7, max_tokens=4096)
    ans, tagged = extract_answer(text)
    if not tagged:
        print(f"warning: missing <answer> tags — used stripped text for: "
              f"{question[:60]}", file=sys.stderr)
    return _pack(ans, text)


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
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="globalopinionqa")
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
    ap.add_argument("--show_raw", action="store_true",
                    help="print the full generation (incl. <think> trace) "
                         "before the extracted answer")
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
        from retrieval.scout import load_or_compute_text_feat

        if args.dataset == "opinionqa":
            from data.loaders.opinionqa import load_opinionqa
            graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
        else:
            from data.loaders.globalopinionqa import load_globalopinionqa
            graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
        h_all = torch.load(args.embeddings, map_location="cpu")
        if not isinstance(h_all, torch.Tensor):
            h_all = h_all["h_all"]
        manifold = PoincareBall(c=args.curvature)
        text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)

    out = answer(args.question, args.condition, graph=graph, h_all=h_all,
                 text_feat=text_feat, manifold=manifold, base_url=args.base_url,
                 model=args.model, dry_run=args.dry_run, cfg=cfg,
                 with_raw=args.show_raw)
    if args.show_raw:
        ans, raw = out
        print("=== RAW GENERATION (with <think> trace) ===")
        print(raw)
        print("\n=== EXTRACTED ANSWER ===")
        print(ans)
    else:
        print(out)


if __name__ == "__main__":
    _main()
