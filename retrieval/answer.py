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
    python -m retrieval.answer --selftest      # merge_v2 guard, no endpoint needed
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from retrieval.scout import (ScoredFork, ScoutConfig, describe_node,
                            node_relevance, random_forks, scout)

# Condition -> ScoutConfig overrides. baseline = no retrieval at all;
# div_only ablates the relevance guards (old pure-divergence scout).
CONDITIONS: dict[str, ScoutConfig | None] = {
    "baseline": None,
    "scout": ScoutConfig(tau=0.25, alpha=1.0),
    "div_only": ScoutConfig(tau=0.0, alpha=0.0),
    "route": ScoutConfig(tau=0.25, alpha=1.0),   # same retrieval as scout;
    #                                              model self-routes (PLURALISM_ROUTE)
    "expand": ScoutConfig(tau=0.25, alpha=1.0),  # same retrieval as scout; model
    #                             enumerates distinct positions first (PLURALISM_EXPAND)
    "distributional": ScoutConfig(tau=0.25, alpha=1.0),  # same retrieval + same
    #        instruction as scout, but injects the FULL subgroup spectrum, not the
    #        two poles (content-fix; subtree_middle.py confirmed the middle is real)
    "merge": ScoutConfig(tau=0.25, alpha=1.0),   # answer twice (plain + injected)
    #        then extractively merge both drafts -- targets the union of their
    #        covered clusters, which is >= the routing oracle (MERGE_INSTRUCTION)
    "merge_v2": ScoutConfig(tau=0.25, alpha=1.0),  # same idea, made STRUCTURALLY
    #        lossless: N drafts, one paragraph per position, plus a length/position
    #        guard that falls back to concatenation when the merge compresses
    #        (MERGE_INSTRUCTION_V2 / merge_drafts)
    "merge_v2_rand": ScoutConfig(tau=0.25, alpha=1.0),  # CONTROL: merge_v2 with
    #        the retrieved fork swapped for a structurally matched fork from an
    #        UNRELATED part of the graph. Everything else identical, so the arm
    #        isolates fork CONTENT from draft VARIANCE. See
    #        docs/random_fork_control.md -- if this ties merge_v2, the graph is a
    #        randomizer and the retrieval contribution does not exist.
    "persona_merge": ScoutConfig(tau=0.25, alpha=1.0),  # merge_v2's machinery,
    #        but the drafts are diversified by SUBGROUP rather than by
    #        instruction: one draft per point on the anchor's opinion spectrum.
    #        Compute-matched to merge_v2 (same n_drafts, one merge call), so the
    #        pair is a clean A/B on the diversification SOURCE.
}

# Conditions that inject the full subgroup spectrum (fork_context_full) instead of
# the two poles. Same instruction as scout, so scout-vs-distributional is a clean
# content-only A/B: does showing the real middle beat showing the two extremes?
FULL_DIST_CONDITIONS: set[str] = {"distributional"}

BASELINE_INSTRUCTION = (
    "Answer the question thoughtfully and concisely."
)

# Think/answer separation: the retrieved forks may be off-topic, and an
# instruction that FORCES them into the answer derails it (measured: coverage
# 0.51 -> 0.06 on OvertonBench when GOQA forks were mandatory). The model
# triages relevance inside <think>; only the <answer> span is shown/judged.
#
# ADDITIVE, not restrictive (v4 lesson): "represent the retrieved perspectives
# faithfully" made the model treat the two injected poles as THE range and
# write narrower answers than baseline (kept-forks: 2 win / 9 loss; even
# discarded forks lost 27/18 — the context anchored the answer). The retrieved
# forks must only ENRICH an already-full-spectrum answer, never bound it.
PLURALISM_INSTRUCTION = (
    "You will see context retrieved from a knowledge graph of survey data, "
    "followed by a question. The context may or may not be relevant, and it "
    "is NEVER complete: it shows at most a few of the many positions people "
    "hold.\n"
    "First, inside <think></think> tags, BRIEFLY (a few sentences) assess "
    "which retrieved perspectives (if any) actually bear on the question.\n"
    "Then, inside <answer></answer> tags, answer the question directly and "
    "thoughtfully, covering the FULL range of positions people genuinely "
    "hold — including positions that do not appear in the context. Treat "
    "relevant retrieved perspectives as supplements only: use them to add "
    "attributed detail (name the groups that hold them), never as the "
    "boundaries of the debate, and do not average real disagreement into a "
    "consensus. If none are relevant, ignore the context entirely and answer "
    "as if it were not provided.\n"
    "The reader sees ONLY what is inside the <answer> tags."
)

# ROUTE (v6): the graph cannot tell us WHICH questions are contested — measured,
# corr(graph divergence, human contestedness) ~ +0.20, driver-match ~ 0. So the
# model decides. The v4/v5 failure was consensus dilution: forced full-spectrum
# enumeration turned committed baseline answers (coverage 1.0) into hedged lists
# (0.0) on questions where people broadly AGREE. This instruction gives the model
# explicit permission to COMMIT when there is consensus, and to pluralize only on
# genuine disagreement — routing per question instead of always pluralizing.
PLURALISM_ROUTE = (
    "You will see context retrieved from a survey knowledge graph, followed by "
    "a question. The context may or may not be relevant, and is never complete.\n"
    "First, inside <think></think> tags, decide in a few sentences: does this "
    "question genuinely divide people into substantially different positions, or "
    "is there broad consensus on it? Treat the retrieved perspectives as evidence "
    "ONLY if they reflect real disagreement on THIS question; ignore tangential "
    "or off-topic context.\n"
    "Then, inside <answer></answer> tags:\n"
    "- If there is broad consensus, give the direct, committed answer that states "
    "the shared view. Be concise; do NOT enumerate multiple positions or hedge — "
    "stating the consensus plainly is what represents people here.\n"
    "- If people genuinely disagree, cover the range of positions, attributing "
    "retrieved perspectives to the groups that hold them, and do not average real "
    "disagreement into a false consensus.\n"
    "The reader sees ONLY what is inside the <answer> tags."
)

# EXPAND (v6): directly attacks the binary-collapse mechanism (docs/framing_hurts.png
# — injected answers pile onto the two poles; on-pole similarity 0.334->0.389,
# corr(attraction, dcoverage) = -0.31). The two retrieved forks are the max-W
# EXTREMES, so anchoring on them caps coverage at ~2 clusters. This instruction
# makes the model enumerate the DISTINCT positions first — explicitly including
# ones beyond the two shown and ones that are NOT between them (avoiding the 1-D
# "moderate midpoint" that just averages disagreement into a false consensus).
# Attribution stays grounded in the retrieved data, not invented group->opinion
# links. Prompt-level proxy for the full-distribution content-fix.
PLURALISM_EXPAND = (
    "You will see context retrieved from a survey knowledge graph, followed by a "
    "question. The retrieved perspectives are the two most DIVERGENT positions on "
    "a related survey item — they are the extremes, not the whole range, and may "
    "not be relevant at all.\n"
    "First, inside <think></think> tags, briefly list the DISTINCT positions people "
    "genuinely hold on this question. Include positions beyond the two retrieved "
    "ones, and positions that do not fall between them — real disagreement is often "
    "several different framings, not a single axis with a middle. Attribute a "
    "position to a group only when the retrieved data supports it; do not invent "
    "group-opinion links.\n"
    "Then, inside <answer></answer> tags, answer the question directly, covering "
    "those distinct positions. Treat the retrieved perspectives as supplements that "
    "add attributed detail, never as the boundaries of the debate, and do not "
    "average real disagreement into a consensus. If none are relevant, ignore the "
    "context and answer as if it were not provided.\n"
    "The reader sees ONLY what is inside the <answer> tags."
)

# MERGE (v7): answer TWICE -- once plain, once with the forks -- then combine.
# Rationale: per-question coverage sets differ between conditions (v6: Q177
# baseline 0.00 / scout 1.00; Q7212 baseline 1.00 / scout 0.33), so the UNION of
# their covered clusters is >= the routing oracle by construction. Routing picks
# the better draft; merging keeps both. That sidesteps the routing problem which
# route_signal (no graph signal) and route (0.072) both failed to solve.
#
# The merge MUST be extractive, not abstractive. `route` shows that naming a
# position is not covering it -- the >=4 bar ("my perspective is represented")
# rewards ARTICULATION DEPTH, so a digest of two drafts can score below either
# one. Hence: preserve each position at the depth its best draft gave it, and
# delete only literal duplication. The output is expected to be LONG.
MERGE_INSTRUCTION = (
    "You will see a question and two draft answers to it, written independently. "
    "Draft A was written without any retrieved context; draft B had access to "
    "survey data about how different groups answer related questions.\n"
    "Write ONE final answer that keeps EVERY distinct position appearing in "
    "either draft.\n"
    "Rules:\n"
    "- Preserve each position at the SAME level of detail as the draft that "
    "explained it best. Do not summarize, compress, or shorten explanations — a "
    "position mentioned in passing does not represent the people who hold it.\n"
    "- Remove only literal duplication: where both drafts make the same point, "
    "keep the fuller version once.\n"
    "- Do not average or reconcile disagreeing positions into a middle view, and "
    "do not add editorial framing about the drafts themselves.\n"
    "- Keep any group attributions (which groups hold which view) from draft B.\n"
    "The final answer will be longer than either draft; that is expected.\n"
    "Put the final answer inside <answer></answer> tags. The reader sees ONLY "
    "what is inside those tags."
)

# MERGE_V2 (v8): `merge` lost to baseline (0.4967), so the merge itself is the
# leak -- the merger SUMMARIZES. The instruction below is the same intent made
# STRUCTURAL: one paragraph per position (a position that shares a paragraph with
# another gets compressed into it), an explicit length floor, and dedup narrowed
# to *identical* positions only. Measured basis: union 0.6730 > oracle 0.6344 >
# baseline 0.4967 -- combining beats picking, but only losslessly; and `route`
# (0.072) proves a NAMED position is not a covered one, so every clause that
# invites brevity is a coverage leak.
MERGE_INSTRUCTION_V2 = (
    "You will see a question and several draft answers to it, written "
    "independently by different processes. Some drafts had access to survey data "
    "about how different groups answer related questions; some did not.\n"
    "Your job is ASSEMBLY, not writing. Produce one final answer that contains "
    "EVERY distinct position that appears in ANY draft, each stated in full.\n"
    "Rules:\n"
    "- One paragraph per distinct position. Never fold two positions into one "
    "paragraph, and never reduce a position to a clause inside another "
    "position's paragraph.\n"
    "- Carry each position over at the FULL length and detail of the draft that "
    "explained it best — copy that draft's wording where you can. You may not "
    "shorten, summarize, condense, abridge, or drop anything. Omitting a "
    "position is the single worst outcome; redundancy is harmless.\n"
    "- Merge two positions into one paragraph ONLY if they are the same position "
    "stated twice. Positions that are related, adjacent, compatible, or "
    "differently-framed versions of a similar idea are DISTINCT — keep them "
    "separate, each with its own paragraph.\n"
    "- Do not average or reconcile disagreeing positions into a middle view, do "
    "not rank them, and do not add framing about the drafts or the merge itself.\n"
    "- Keep every group attribution (which groups hold which view) verbatim.\n"
    "The final answer MUST be at least as long as the longest draft, and will "
    "normally be considerably longer. Length is not a defect here.\n"
    "Put the final answer inside <answer></answer> tags. The reader sees ONLY "
    "what is inside those tags."
)

# Instruction per condition (injected conditions only; baseline uses its own).
INSTRUCTION_BY_CONDITION: dict[str, str] = {
    "route": PLURALISM_ROUTE,
    "expand": PLURALISM_EXPAND,
}

# Conditions that make MULTIPLE generation calls (draft A, draft B, then merge)
# instead of one. Handled by _merge_answer*(), not the single-call path.
MULTI_PASS_CONDITIONS: set[str] = {"merge", "merge_v2", "persona_merge",
                                   "merge_v2_rand"}

# Conditions whose fork is REPLACED by a matched irrelevant one after retrieval.
RANDOM_FORK_CONDITIONS: set[str] = {"merge_v2_rand"}


@dataclass(frozen=True)
class MergeConfig:
    """Knobs for merge_v2. Every default is a measured quantity, not a guess.

    n_drafts        3, because the union over three conditions covers 0.6730 of
                    clusters vs 0.6458 over two -- the third draft is +0.027,
                    exactly the noise floor, so it is worth its extra call only
                    if the merge is lossless. Capped at len(DRAFT_SPECS).
    min_len_ratio   merged words / longest draft's words. 1.0 = the task floor
                    ("at least as long as the longer draft"); a merge that comes
                    back shorter has provably deleted text.
    min_pos_ratio   merged positions / max positions in any single draft. 1.0:
                    the merge may not express fewer positions than its best
                    input already did. This is the floor, not the target (the
                    target is the union, which we cannot count without a judge).
    min_depth_words words a paragraph needs before it counts as a position at
                    all. 20, matching alignment.reward.RewardConfig.min_depth_words
                    -- the bar that separates `route`-style mentions (0.072) from
                    articulation. Counting shallow paragraphs would let a merge
                    pass the guard by listing position names.
    fallback        False disables concatenation (returns the lossy merge anyway);
                    only for measuring how often the guard would have fired.
    """
    n_drafts: int = 3
    min_len_ratio: float = 1.0
    min_pos_ratio: float = 1.0
    min_depth_words: int = 20
    fallback: bool = True
    merge_max_tokens: int = 4096
    #   UPPER bound; the effective value is computed per call (see _merge_budget).
    #   vLLM rejects prompt_tokens + max_tokens > max_model_len with a bare HTTP
    #   400 -- job 69559 died exactly here at 8192, since three drafts plus the
    #   instruction are ~2k prompt tokens before the merge even starts.
    max_model_len: int = 8192     # must match `vllm serve --max-model-len`
    token_slack: int = 256        # margin for the chat template + estimator error


# Draft recipes, in priority order; merge_v2 takes the first ``n_drafts``. Chosen
# so the drafts DISAGREE about what to cover -- v6 measured per-question coverage
# sets that differ between conditions (Q177 baseline 0.00/scout 1.00; Q7212
# baseline 1.00/scout 0.33). Two drafts from the same recipe would union to
# nothing new. Order = measured standalone coverage (baseline 0.4967 > scout
# 0.3927 ~ distributional 0.3941), so a 2-draft run keeps the strongest pair.
#   (label, needs_forks, instruction, full_dist)
DRAFT_SPECS: list[tuple[str, bool, str, bool]] = [
    ("plain", False, BASELINE_INSTRUCTION, False),
    ("scout", True, PLURALISM_INSTRUCTION, False),
    ("distributional", True, PLURALISM_INSTRUCTION, True),
    ("expand", True, PLURALISM_EXPAND, False),
]

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # vLLM puts the REASON in the body ("maximum context length is 8192,
        # however you requested ..."), and a bare HTTPError discards it -- job
        # 69559 cost a 4-minute run and a guess. Surface it.
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        raise RuntimeError(
            f"chat {e.code} {e.reason} (model={model}, max_tokens={max_tokens}, "
            f"n_messages={len(messages)}): {detail}") from e
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


def _leaf_vec(graph, nid):
    d = getattr(graph, "opinion_dist", {}).get(nid)
    return d if d else None


def fork_context_full(fork: ScoredFork, graph, k: int = 1,
                      max_subgroups: int = 8) -> str:
    """The fork as the FULL subgroup spectrum, not just the two poles.

    The 2-pole render (fork_context) collapses the answer onto the extremes
    (docs/framing_hurts.png). subtree_middle.py showed 68% of an axis's non-pole
    subgroups genuinely lie BETWEEN the poles (only ~11% of axes are bimodal), so
    injecting every subgroup gives the model the real middle instead of a binary.
    Renders the anchor's opinion-leaf children ordered as a spectrum (along the
    pole-to-pole axis); falls back to the 2-pole block if the anchor has too few.
    """
    anchor = fork.anchor
    leaves = spectrum_leaves(fork, graph, max_subgroups)
    if not leaves:
        return fork_context(fork, graph, k)          # not enough spread → poles

    lines = [f"[fork {k}] on '{describe_node(graph, anchor, 60)}' — full subgroup "
             f"spectrum (divergence={fork.w:.2f}, relevance={fork.relevance:.2f}):"]
    for c in leaves:
        lines.append(f"    {describe_node(graph, c, 0, show_q=False)}")
    return "\n".join(lines)


def spectrum_leaves(fork, graph, max_subgroups: int = 8) -> list[int]:
    """The anchor's opinion leaves ordered along the pole-to-pole axis.

    Returns [] when the anchor has too little spread to be a spectrum (<3 leaves,
    or a pole with no distribution) -- callers fall back to the 2-pole render.

    Extracted from fork_context_full so persona selection picks its subgroups
    from the SAME ordering the distributional render shows, rather than
    re-deriving a second, silently-different axis.
    """
    leaves = list(dict.fromkeys(
        c for c in graph.children_indices[fork.anchor]
        if _leaf_vec(graph, c) is not None))
    A = fork.branch_a if _leaf_vec(graph, fork.branch_a) is not None else None
    B = fork.branch_b if _leaf_vec(graph, fork.branch_b) is not None else None
    if len(leaves) < 3 or A is None or B is None:
        return []

    pa, pb = _leaf_vec(graph, A), _leaf_vec(graph, B)
    if len(pa) == len(pb) and all(len(_leaf_vec(graph, c)) == len(pa) for c in leaves):
        av = [b - a for a, b in zip(pa, pb)]
        den = sum(x * x for x in av) or 1.0
        def _t(c):
            p = _leaf_vec(graph, c)
            return sum((pc - a) * v for pc, a, v in zip(p, pa, av)) / den
        leaves.sort(key=_t)
    if len(leaves) > max_subgroups:                  # poles + evenly-spaced middle
        keep = {0, len(leaves) - 1}
        step = (len(leaves) - 1) / (max_subgroups - 1)
        keep.update(round(i * step) for i in range(max_subgroups))
        leaves = [leaves[i] for i in sorted(keep)][:max_subgroups]
    return leaves


def pick_personas(fork, graph, n: int) -> list[int]:
    """``n`` subgroups spread along the spectrum: both poles first, then middle.

    Poles first because they sit furthest apart and so are most likely to make
    the drafts DISAGREE -- disagreement between drafts is the only thing that
    raises the union merge_v2 harvests. The middle comes next: subtree_middle.py
    found 68% of an axis's non-pole subgroups genuinely lie BETWEEN the poles, so
    a third persona is a real third position, not a blend of the first two.
    """
    leaves = spectrum_leaves(fork, graph)
    if n < 1:
        return []
    if len(leaves) < 2:
        return leaves[:n]
    picks = [0, len(leaves) - 1]                     # both poles
    while len(picks) < min(n, len(leaves)):          # then split the widest gap
        gaps = sorted(zip(picks, picks[1:]), key=lambda ab: ab[1] - ab[0])
        if not gaps:
            break
        lo, hi = gaps[-1]
        mid = (lo + hi) // 2
        if mid in picks:
            break
        picks = sorted(picks + [mid])
    return [leaves[i] for i in picks[:n]]


def persona_context(graph, leaf: int, k: int = 1) -> str:
    """One subgroup's actual survey distribution, as the draft's vantage point."""
    return f"[group {k}] {describe_node(graph, leaf, 0, show_q=True)}"


def forks_to_context(forks: list[ScoredFork], graph, full_dist: bool = False) -> str:
    """All fork blocks joined — exactly the context string the LLM sees.

    ``full_dist=True`` renders every subgroup of each anchor (the content-fix)
    instead of the two poles.
    """
    render = fork_context_full if full_dist else fork_context
    return "\n\n".join(render(f, graph, k) for k, f in enumerate(forks, 1))



def matched_random_fork(real, graph, h_all, text_feat, manifold,
                        cfg: ScoutConfig, q_emb, full_dist: bool, seed: int = 0):
    """Swap the retrieved forks for matched IRRELEVANT ones; returns (forks, stats).

    The control for docs/random_fork_control.md. `random_forks` supplies
    candidates at the same depth with a comparable branching factor; this picks
    among them on RENDERED LENGTH, because that is what actually reaches the
    prompt.

    MATCHES THE WHOLE BLOCK, NOT ONE FORK. The first version returned a single
    random fork and compared its rendering against the top real fork -- so the
    stats reported "length ratio 0.97" while the arms actually injected 4256 vs
    859 characters (4.85 forks vs 1). That made the control vary VOLUME 5x as
    well as relevance, and any difference between the arms was uninterpretable.
    One random fork is now drawn per real fork, greedily nearest on rendered
    length, without reuse.

    Returns ``([], stats)`` when the graph offers no comparable unrelated anchor.
    The caller must SKIP such a question rather than fall through to an
    uninjected prompt -- a silent baseline row inside the control arm would bias
    it toward merge_v2 by exactly the questions where matching is hardest.
    """
    stats = {"randomized": False, "n_candidates": 0, "n_real": 0, "n_rand": 0,
             "len_real": 0, "len_rand": 0,
             "rel_real": float("nan"), "rel_rand": float("nan")}
    if not real:
        return [], stats
    rel = node_relevance(q_emb, text_feat)
    # Candidates are drawn once, matched to the TOP real fork's shape, then
    # reused across slots -- n_candidates only has to exceed len(real).
    cands = random_forks(real[0], graph, h_all, rel, manifold, cfg,
                         n_candidates=max(12, 3 * len(real)), seed=seed)
    stats["n_candidates"] = len(cands)
    if not cands:
        return [], stats

    chosen, pool = [], list(cands)
    for rf in real:                       # one random fork per real fork
        if not pool:
            break
        target = len(forks_to_context([rf], graph, full_dist))
        best_i, best_gap = 0, None
        for i, c in enumerate(pool):
            gap = abs(len(forks_to_context([c], graph, full_dist)) - target)
            if best_gap is None or gap < best_gap:
                best_i, best_gap = i, gap
        chosen.append(pool.pop(best_i))   # no reuse: a repeated block is not a fork set
    if not chosen:
        return [], stats

    stats.update(randomized=True, n_real=len(real), n_rand=len(chosen),
                 len_real=len(forks_to_context(real, graph, full_dist)),
                 len_rand=len(forks_to_context(chosen, graph, full_dist)),
                 rel_real=sum(f.relevance for f in real) / len(real),
                 rel_rand=sum(f.relevance for f in chosen) / len(chosen))
    return chosen, stats


def build_prompt(question: str, forks: list[ScoredFork] | None, graph,
                 instruction: str = PLURALISM_INSTRUCTION,
                 full_dist: bool = False) -> list[dict]:
    """Chat messages: instruction + fork blocks + question (last).

    ``instruction`` is the injected-condition system prompt (default = additive
    pluralism; ``route`` passes PLURALISM_ROUTE for self-routing). ``full_dist``
    injects each anchor's full subgroup spectrum instead of the two poles.
    """
    if not forks:
        return [{"role": "system", "content": BASELINE_INSTRUCTION},
                {"role": "user", "content": question}]
    ctx = forks_to_context(forks, graph, full_dist)
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": ctx + "\n\nQuestion: " + question}]


def _merge_answer(question: str, forks, graph, base_url: str, model: str,
                  full_dist: bool = False) -> tuple[str, dict]:
    """Draft plain -> draft injected -> extractive merge. Returns (answer, parts).

    Three calls. The point is the UNION of what the two drafts cover: v6 showed
    the covered-cluster sets differ per question, and union >= oracle >= best
    single by construction. Whether the merge REALIZES that union is the open
    question -- compression is how it would fail (see MERGE_INSTRUCTION).
    """
    draft_a = chat(base_url, model,
                   [{"role": "system", "content": BASELINE_INSTRUCTION},
                    {"role": "user", "content": question}],
                   temperature=0.7, max_tokens=2048)

    if forks:
        raw_b = chat(base_url, model,
                     build_prompt(question, forks, graph, PLURALISM_INSTRUCTION,
                                  full_dist),
                     temperature=0.7, max_tokens=4096)
        draft_b, _ = extract_answer(raw_b)
    else:                                   # no forks retrieved -> nothing to merge
        raw_b, draft_b = "", ""
        return draft_a, {"draft_a": draft_a, "draft_b": "", "raw_merge": draft_a}

    raw_merge = chat(base_url, model,
                     [{"role": "system", "content": MERGE_INSTRUCTION},
                      {"role": "user", "content":
                       f"Question: {question}\n\n"
                       f"--- DRAFT A (no retrieved context) ---\n{draft_a}\n\n"
                       f"--- DRAFT B (with survey context) ---\n{draft_b}"}],
                     temperature=0.7, max_tokens=6144)   # holds both drafts' content
    merged, tagged = extract_answer(raw_merge)
    if not tagged:
        import sys
        print(f"warning: merge missing <answer> tags for: {question[:60]}",
              file=sys.stderr)
    return merged, {"draft_a": draft_a, "draft_b": draft_b, "raw_merge": raw_merge}


# ---------------------------------------------------------------------------
# merge_v2: structurally lossless merge (guard + concatenation fallback)
# ---------------------------------------------------------------------------
def _n_words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def split_paragraphs(text: str) -> list[str]:
    """Text -> candidate position units: bullet lines, else blank-line blocks.

    MERGE_INSTRUCTION_V2 asks for one paragraph per position, so paragraphs are
    the countable unit. Bullets are split individually because a bulleted block
    is one paragraph typographically but N positions semantically.
    """
    units: list[str] = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if sum(1 for ln in lines if _BULLET_RE.match(ln)) >= 2:
            units.extend(_BULLET_RE.sub("", ln).strip() for ln in lines)
        elif lines:
            units.append(" ".join(ln.strip() for ln in lines))
    return [u for u in units if u]


def count_positions(text: str, min_depth_words: int = 20) -> int:
    """Paragraphs deep enough to count as an expressed position.

    Depth-gated on purpose: `route` named many positions in one clause each and
    scored 0.072 vs baseline's 0.507, so a shallow paragraph is not a covered
    cluster and must not let a lossy merge past the guard.
    """
    return sum(1 for u in split_paragraphs(text) if _n_words(u) >= min_depth_words)


def merge_guard(merged: str, drafts: list[str],
                cfg: MergeConfig = MergeConfig()) -> tuple[bool, str, dict]:
    """(ok, reason, stats) — did the merge preserve its inputs?

    Two structural checks, both one-sided (they can only detect LOSS):
      length    a merge shorter than its longest input deleted text;
      positions a merge expressing fewer deep paragraphs than its best single
                input dropped positions.
    Neither proves losslessness -- a merge can keep the word count and still
    swap a position for filler -- but `merge`'s failure mode is compression, and
    compression is exactly what these catch.
    """
    m_words = _n_words(merged)
    d_words = max((_n_words(d) for d in drafts), default=0)
    m_pos = count_positions(merged, cfg.min_depth_words)
    d_pos = max((count_positions(d, cfg.min_depth_words) for d in drafts), default=0)
    stats = {"merged_words": m_words, "max_draft_words": d_words,
             "merged_positions": m_pos, "max_draft_positions": d_pos}
    if not merged.strip():
        return False, "empty", stats
    if m_words < cfg.min_len_ratio * d_words:
        return False, f"short ({m_words} < {cfg.min_len_ratio:g}x{d_words})", stats
    if m_pos < cfg.min_pos_ratio * d_pos:
        return False, f"positions ({m_pos} < {cfg.min_pos_ratio:g}x{d_pos})", stats
    return True, "", stats


def concat_drafts(drafts: list[str], labels: list[str] | None = None) -> str:
    """The lossless fallback: every draft kept whole, under neutral headings.

    Lossless by construction, so its coverage is >= the best single draft's --
    i.e. >= oracle-per-question (0.6344) modulo judge noise, where a compressing
    merge measured BELOW baseline (0.4967). Headings are content-neutral
    ("Perspective k"): naming the drafts would be editorial framing about the
    generation process, which the judge reads as part of the answer.
    """
    kept = [d.strip() for d in drafts if d and d.strip()]
    if len(kept) <= 1:
        return kept[0] if kept else ""
    del labels                                   # deliberately not surfaced
    return "\n\n".join(f"## Perspective {i}\n\n{d}" for i, d in enumerate(kept, 1))


def _merge_budget(prompt_text: str, cfg: MergeConfig) -> int:
    """Completion tokens left in the context window after the merge prompt.

    A fixed max_tokens is wrong in both directions: too high and vLLM rejects the
    request outright (HTTP 400, prompt + max_tokens > max_model_len -- how job
    69559 died), too low and the merge is forced to compress, manufacturing the
    very loss merge_guard exists to detect. So size it from the actual prompt.

    ~4 chars/token is the usual English estimate; it over-counts on prose, which
    errs toward a SMALLER request -- the safe direction, since the guard catches a
    short merge but nothing catches a rejected one. Never returns less than 512:
    below that the merge is doomed anyway and should fail visibly, not silently.
    """
    est_prompt = len(prompt_text) // 4
    room = cfg.max_model_len - est_prompt - cfg.token_slack
    return max(512, min(cfg.merge_max_tokens, room))


def merge_drafts(question: str, drafts: list[str], base_url: str, model: str, *,
                 cfg: MergeConfig = MergeConfig(), chat_fn=None,
                 labels: list[str] | None = None) -> tuple[str, dict]:
    """One merge call + guard + fallback. Pure w.r.t. retrieval — drafts in, answer out.

    ``chat_fn`` is injected (defaults to ``chat``) so the merge logic is testable
    without an endpoint; see ``--selftest``.
    """
    chat_fn = chat_fn or chat
    kept = [d.strip() for d in drafts if d and d.strip()]
    info: dict = {"drafts": kept, "labels": list(labels or []),
                  "merge_fallback": False, "merge_fail": "", "raw_merge": ""}
    if len(kept) <= 1:                      # nothing to merge (e.g. 0 forks)
        out = kept[0] if kept else ""
        info["raw_merge"] = out
        return out, info

    blocks = "\n\n".join(
        f"--- DRAFT {chr(65 + i)} ---\n{d}" for i, d in enumerate(kept))
    # The merge must hold every draft in full -- a token cap that truncates IS
    # compression, i.e. it manufactures the failure the guard exists to catch.
    # So take everything the context window has left rather than a fixed number.
    user_msg = f"Question: {question}\n\n{blocks}"
    budget = _merge_budget(MERGE_INSTRUCTION_V2 + user_msg, cfg)

    # If the drafts alone overflow the window, no max_tokens makes the request
    # legal -- it would 400 whatever we ask for. Concatenating is exactly what
    # the guard would have done anyway, so do it now instead of losing the call.
    est = len(MERGE_INSTRUCTION_V2 + user_msg) // 4
    if est + budget > cfg.max_model_len:
        info["merge_fallback"] = True
        info["merge_fail"] = f"prompt_overflow ({est} tok > {cfg.max_model_len})"
        print(f"warning: merge_v2 drafts exceed the context window "
              f"({est} tok) -- concatenating for: {question[:60]}", file=sys.stderr)
        return concat_drafts(kept), info

    raw_merge = chat_fn(base_url, model,
                        [{"role": "system", "content": MERGE_INSTRUCTION_V2},
                         {"role": "user", "content": user_msg}],
                        temperature=0.7, max_tokens=budget)
    merged, tagged = extract_answer(raw_merge)
    info["raw_merge"] = raw_merge
    if not tagged:
        print(f"warning: merge_v2 missing <answer> tags for: {question[:60]}",
              file=sys.stderr)

    ok, reason, stats = merge_guard(merged, kept, cfg)
    info["merge_stats"] = stats
    if ok:
        return merged, info
    info["merge_fail"] = reason
    # The fallback RATE is the finding: it measures how often the model cannot
    # merge without compressing, which is the hypothesis for `merge` < baseline.
    print(f"warning: merge_v2 LOSSY [{reason}] "
          f"({stats['merged_words']}w/{stats['merged_positions']}pos vs draft "
          f"{stats['max_draft_words']}w/{stats['max_draft_positions']}pos) -- "
          f"{'concatenating' if cfg.fallback else 'kept anyway'} for: "
          f"{question[:60]}", file=sys.stderr)
    if not cfg.fallback:
        return merged, info
    info["merge_fallback"] = True
    return concat_drafts(kept, labels), info


PERSONA_INSTRUCTION = (
    "You are shown how one specific group of people actually answered a survey "
    "question, with their real response percentages. Write the most thoughtful "
    "answer to the question that reflects how that group sees it, and the "
    "reasons they would give.\n"
    "Commit to that view. Do not hedge, do not present other groups' positions, "
    "and do not mention the group, the survey, or these instructions -- write "
    "the answer itself, as someone holding that view would argue it."
)


def _persona_merge_answer(question: str, forks, graph, base_url: str, model: str,
                          cfg: MergeConfig = MergeConfig(),
                          chat_fn=None) -> tuple[str, dict]:
    """merge_v2's machinery, drafts diversified by SUBGROUP instead of instruction.

    merge_v2 varies the INSTRUCTION across drafts (plain / scout / distributional)
    and its whole gain comes from the union over drafts that disagree. But those
    three drafts see the same fork block, so they disagree only as much as three
    phrasings of one context can. Conditioning each draft on a DIFFERENT point of
    the anchor's opinion spectrum makes them disagree about content instead.

    Compute-matched to merge_v2 on purpose: n_drafts drafts + one merge call, and
    the first draft is the same plain baseline. Only the source of draft-to-draft
    variation differs, so persona_merge vs merge_v2 isolates exactly that.

    Falls back to plain-only when the anchor has no usable spectrum (<3 opinion
    leaves) -- which is also the signal that this question had no subgroup
    disagreement to exploit.
    """
    chat_fn = chat_fn or chat
    n = max(1, min(cfg.n_drafts, len(DRAFT_SPECS)))
    personas = pick_personas(forks[0], graph, n - 1) if forks else []

    drafts, labels = [], []
    # Draft 1 is the plain baseline -- the strongest single condition measured
    # (0.4967 vs scout 0.3927), so dropping it to buy a third persona would
    # trade the best draft for a marginal one.
    raw = chat_fn(base_url, model,
                  build_prompt(question, None, graph, BASELINE_INSTRUCTION),
                  temperature=0.7, max_tokens=2048)
    text, _ = extract_answer(raw)
    if text.strip():
        drafts.append(text.strip()); labels.append("plain")

    for k, leaf in enumerate(personas, 1):
        msgs = [{"role": "system", "content": PERSONA_INSTRUCTION},
                {"role": "user", "content": persona_context(graph, leaf, k)
                 + "\n\nQuestion: " + question}]
        raw = chat_fn(base_url, model, msgs, temperature=0.7, max_tokens=2048)
        text, _ = extract_answer(raw)
        if text.strip():
            drafts.append(text.strip()); labels.append(f"persona{k}")

    merged, info = merge_drafts(question, drafts, base_url, model, cfg=cfg,
                                chat_fn=chat_fn, labels=labels)
    info["n_personas"] = len(personas)
    parts = {"draft_a": drafts[0] if drafts else "",
             "draft_b": drafts[1] if len(drafts) > 1 else "", **info}
    return merged, parts


def _merge_answer_v2(question: str, forks, graph, base_url: str, model: str,
                     full_dist: bool = False, cfg: MergeConfig = MergeConfig(),
                     chat_fn=None) -> tuple[str, dict]:
    """N drafts -> lossless merge. Returns (answer, parts). 1 + n_drafts calls.

    ``full_dist`` forces the fork-injected drafts to render the full subgroup
    spectrum; the 3-draft plan already includes one of each rendering, so it only
    matters at n_drafts=2.
    """
    chat_fn = chat_fn or chat
    specs = DRAFT_SPECS[:max(1, min(cfg.n_drafts, len(DRAFT_SPECS)))]
    if not forks:                       # nothing retrieved -> only the plain draft
        specs = [s for s in specs if not s[1]]

    drafts, labels = [], []
    for label, needs_forks, instruction, spec_full in specs:
        fd = spec_full or (full_dist and needs_forks)
        msgs = build_prompt(question, forks if needs_forks else None, graph,
                            instruction, fd)
        # 2048 plain / 4096 injected: injected drafts also spend tokens on the
        # <think> triage span (at 2048, ~58% lost their closing tag).
        raw = chat_fn(base_url, model, msgs, temperature=0.7,
                      max_tokens=4096 if needs_forks else 2048)
        # extract unconditionally: the plain draft has no tags to strip, but a
        # model that emits them anyway must not leak tag literals into the merge
        text, _ = extract_answer(raw)
        if text.strip():
            drafts.append(text.strip())
            labels.append(label)

    merged, info = merge_drafts(question, drafts, base_url, model, cfg=cfg,
                                chat_fn=chat_fn, labels=labels)
    # draft_a/draft_b keep the v1 trace schema so eval/analysis reads both alike
    parts = {"draft_a": drafts[0] if drafts else "",
             "draft_b": drafts[1] if len(drafts) > 1 else "", **info}
    return merged, parts


def _selftest_random_fork() -> None:
    """random_forks must pick UNRELATED anchors at MATCHED depth.

    Synthetic tree, so this checks the sampler's contract rather than the graph:
    root -> {1,2,3}, each with three leaves, and only anchor 1 made relevant. The
    relevance assertion is the manipulation check -- if the sampled forks were as
    relevant as the real one, the control would not be a control.
    """
    import types

    from retrieval.scout import _node_depths

    kids = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] + [[] for _ in range(9)]
    g = types.SimpleNamespace(children_indices=kids)
    torch_mod = __import__("torch")
    torch_mod.manual_seed(0)
    h = torch_mod.randn(13, 8)
    rel = torch_mod.zeros(13)
    for i in (1, 4, 5, 6):
        rel[i] = 0.9
    cfg = ScoutConfig()
    real = ScoredFork(anchor=1, branch_a=4, branch_b=5, w=1.0, relevance=0.9,
                      score=0.9, nodes_a=[4], nodes_b=[5])

    cands = random_forks(real, g, h, rel, None, cfg, seed=0)
    assert cands, "should find unrelated anchors at the same depth"
    assert all(c.anchor in (2, 3) for c in cands), "must exclude the real anchor"
    assert all(_node_depths(kids)[c.anchor] == 1 for c in cands), "depth must match"
    assert all(c.relevance < real.relevance for c in cands),         "sampled forks must be LESS relevant -- the manipulation check"
    assert all(c.w == c.w and c.w > 0 for c in cands), "W recomputed, not copied"

    # A graph with no comparable anchor must return [], so answer() skips the
    # question rather than silently generating an uninjected (baseline) row.
    kids2 = [[1], [2, 3], [], []]
    g2 = types.SimpleNamespace(children_indices=kids2)
    real2 = ScoredFork(anchor=1, branch_a=2, branch_b=3, w=1.0, relevance=0.9,
                       score=0.9, nodes_a=[2], nodes_b=[3])
    assert random_forks(real2, g2, torch_mod.randn(4, 8), torch_mod.ones(4),
                        None, cfg) == [], "no alternative anchor -> skip"
    print("  rand fork : unrelated, depth-matched, less relevant, W recomputed")


def _pack_skip(with_trace: bool, with_raw: bool):
    """Sentinel for a question the control arm cannot match. Empty answer, plus a
    trace saying why -- so the row is visibly skipped, not silently baseline."""
    if with_trace:
        return "", {"raw": "", "think": "", "fork_context": "", "n_forks": 0,
                    "skipped": "no_matched_random_fork"}
    return ("", "") if with_raw else ""


def answer(question: str, condition: str, *, graph=None, h_all=None,
           text_feat=None, manifold=None, base_url: str = "",
           model: str = "", dry_run: bool = False, q_emb=None,
           cfg: ScoutConfig | None = None, with_raw: bool = False,
           with_trace: bool = False, merge_cfg: MergeConfig | None = None,
           chat_fn=None):
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
    instruction = INSTRUCTION_BY_CONDITION.get(condition, PLURALISM_INSTRUCTION)
    full_dist = condition in FULL_DIST_CONDITIONS
    rand_stats = None
    if condition in RANDOM_FORK_CONDITIONS and forks:
        if q_emb is None:
            from retrieval.scout import embed_question
            q_emb = embed_question(question)
        forks, rand_stats = matched_random_fork(
            forks, graph, h_all, text_feat, manifold, cfg, q_emb, full_dist)
        if not forks:
            # No comparable unrelated anchor. Falling through would inject
            # nothing and silently score a baseline answer inside the control.
            print(f"warning: {condition} found no matched random fork -- "
                  f"SKIPPING: {question[:60]}", file=sys.stderr)
            return _pack_skip(with_trace, with_raw)
    messages = build_prompt(question, forks, graph, instruction, full_dist)
    ctx = forks_to_context(forks, graph, full_dist) if forks else ""

    def _pack(ans: str, raw: str):
        if with_trace:
            return ans, {"raw": raw, "think": extract_think(raw),
                         "fork_context": ctx, "n_forks": len(forks or [])}
        return (ans, raw) if with_raw else ans

    if dry_run:
        prompt = "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
        return _pack(prompt, prompt)
    if condition in MULTI_PASS_CONDITIONS:
        if condition == "persona_merge":
            merged, parts = _persona_merge_answer(
                question, forks, graph, base_url, model,
                merge_cfg or MergeConfig(), chat_fn)
        elif condition == "merge_v2":
            merged, parts = _merge_answer_v2(
                question, forks, graph, base_url, model, full_dist,
                merge_cfg or MergeConfig(), chat_fn)
        else:
            merged, parts = _merge_answer(question, forks, graph, base_url,
                                          model, full_dist)
        if with_trace:
            trace = {"raw": parts["raw_merge"],
                     "think": extract_think(parts["raw_merge"]),
                     "fork_context": ctx, "n_forks": len(forks or []),
                     "draft_a": parts["draft_a"],
                     "draft_b": parts["draft_b"]}
            # merge_v2 only: whether the guard fired (the rate is the finding).
            # n_personas is here because persona_merge falls back to plain-only
            # when the anchor has <3 opinion leaves, and such a row is a BASELINE
            # row wearing the condition's name -- it drags both the score and the
            # union gain toward zero. Without the field the v11 run could not say
            # how many rows actually tested the condition.
            for k in ("merge_fallback", "merge_fail", "merge_stats", "labels",
                      "n_personas"):
                if k in parts:
                    trace[k] = parts[k]
            if rand_stats is not None:
                trace["random_fork"] = rand_stats
            return merged, trace
        return (merged, parts["raw_merge"]) if with_raw else merged
    chat_fn = chat_fn or chat
    if not forks:
        text = chat_fn(base_url, model, messages, temperature=0.7)
        return _pack(text, text)
    # Injected conditions think before answering — budget for both spans,
    # then keep only the <answer> span (the judge must not see the triage).
    # 4096: at 2048, ~58% of answers lost their closing tag to the token cap.
    text = chat_fn(base_url, model, messages, temperature=0.7, max_tokens=4096)
    ans, tagged = extract_answer(text)
    if not tagged:
        print(f"warning: missing <answer> tags — used stripped text for: "
              f"{question[:60]}", file=sys.stderr)
    return _pack(ans, text)


# ---------------------------------------------------------------------------
# Self-test (no endpoint required: chat_fn is stubbed)
# ---------------------------------------------------------------------------
def _persona_selftest() -> None:
    """pick_personas spreads along the spectrum; the draft path stays compute-matched.

    Two things that would silently break the A/B with merge_v2 and are invisible
    at runtime: picking personas that are NOT spread (so the drafts agree and the
    union gains nothing), and spending a different number of calls (so any win is
    just extra compute). Both are asserted here.
    """
    class _G:                                    # 5 subgroups on one axis
        children_indices = {0: [1, 2, 3, 4, 5]}
        id_to_entity = {i: f"op:q1:PARTY:g{i}" for i in range(6)}
        opinion_texts = {i: ["Agree", "Disagree"] for i in range(1, 6)}
        # a clean gradient from all-agree to all-disagree
        opinion_dist = {i: [1.0 - (i - 1) / 4.0, (i - 1) / 4.0] for i in range(1, 6)}
        entity_text = {}

    class _F:
        anchor, branch_a, branch_b = 0, 1, 5
        w, relevance, top_pairs = 3.0, 0.9, []

    g, fork = _G(), _F()
    leaves = spectrum_leaves(fork, g)
    assert leaves == [1, 2, 3, 4, 5], leaves          # ordered along A->B

    assert pick_personas(fork, g, 2) == [1, 5]        # both poles first
    p3 = pick_personas(fork, g, 3)
    assert p3 == [1, 3, 5], p3                        # then the true middle
    assert len(pick_personas(fork, g, 9)) == 5        # never more than exist

    # An anchor with no spectrum yields no personas -- the caller must degrade to
    # the plain draft rather than inventing a vantage point.
    class _Flat(_G):
        children_indices = {0: [1]}
    assert pick_personas(_F(), _Flat(), 2) in ([], [1]), "flat anchor"

    # Compute parity with merge_v2: n_drafts draft calls + exactly one merge.
    seen = []

    def _chat(base_url, model, messages, **kw):
        seen.append(messages[0]["content"])
        if messages[0]["content"] == MERGE_INSTRUCTION_V2:
            return "<answer>" + ("word " * 400) + "</answer>"
        return "<answer>" + ("word " * 120) + "</answer>"

    cfg = MergeConfig(n_drafts=3)
    _out, parts = _persona_merge_answer("Q?", [fork], g, "", "", cfg, _chat)
    n_merge = sum(1 for c in seen if c == MERGE_INSTRUCTION_V2)
    n_persona = sum(1 for c in seen if c == PERSONA_INSTRUCTION)
    n_plain = sum(1 for c in seen if c == BASELINE_INSTRUCTION)
    assert n_merge == 1, seen
    assert n_plain == 1 and n_persona == 2, (n_plain, n_persona)
    assert len(seen) == cfg.n_drafts + 1 == 4, len(seen)
    assert parts["n_personas"] == 2, parts["n_personas"]
    print("  persona   : leaves ordered, poles-then-middle, "
          f"{n_plain} plain + {n_persona} persona + {n_merge} merge = "
          f"{len(seen)} calls (merge_v2 parity)")


def _selftest() -> None:
    """Exercise the merge_v2 guard against canned merger outputs.

    The claim under test: a merger that SUMMARIZES must be caught and replaced by
    concatenation, while a genuinely lossless merge must pass untouched. That is
    the only thing separating merge_v2 from `merge` (which lost to baseline at
    0.4967), so it is what the offline test has to pin down.
    """
    def para(topic: str, n: int) -> str:
        """One position articulated at ~n words."""
        return f"Some people think {topic} " + " ".join([topic] * (n - 4)) + "."

    draft_a = "\n\n".join(para(t, 40) for t in ("cost", "safety", "freedom"))
    draft_b = "\n\n".join(para(t, 50) for t in ("cost", "tradition", "equity"))
    draft_c = "\n\n".join(para(t, 45) for t in ("religion", "science"))
    drafts = [draft_a, draft_b, draft_c]

    good = "\n\n".join(para(t, 50) for t in
                       ("cost", "safety", "freedom", "tradition", "equity"))
    short = "\n\n".join(para(t, 12) for t in ("cost", "safety"))   # a digest
    wall = para("everything", 400)          # long enough, but ONE position

    def stub(reply: str):
        """chat() double: drafts echo canned text, the merge returns ``reply``."""
        calls = []

        def _chat(base_url, model, messages, **kw):
            calls.append(messages[0]["content"])
            if messages[0]["content"] == MERGE_INSTRUCTION_V2:
                return f"<answer>{reply}</answer>"
            return "<answer>" + draft_a + "</answer>"
        return _chat, calls

    q = "Should X be allowed?"
    cfg = MergeConfig(n_drafts=3)

    # 1. lossless merge -> returned as-is, guard silent
    fn, _ = stub(good)
    out, info = merge_drafts(q, drafts, "", "", cfg=cfg, chat_fn=fn)
    assert not info["merge_fallback"], info
    assert out == good and info["merge_fail"] == "", info["merge_fail"]
    s_good = info["merge_stats"]

    # 2. summarizing merge -> guard fires on LENGTH, falls back to concatenation
    fn, _ = stub(short)
    out, info = merge_drafts(q, drafts, "", "", cfg=cfg, chat_fn=fn)
    assert info["merge_fallback"] and info["merge_fail"].startswith("short"), info
    for d in drafts:                        # concatenation is lossless: all text kept
        for p in split_paragraphs(d):
            assert p in out, p[:40]
    assert out.count("## Perspective") == 3, out[:200]
    s_short = info["merge_stats"]

    # 3. long but collapsed (positions folded into one paragraph) -> also caught.
    #    This is the case a length-only guard would MISS, and it is the `route`
    #    failure mode (0.072): text present, positions not separately articulated.
    fn, _ = stub(wall)
    out, info = merge_drafts(q, drafts, "", "", cfg=cfg, chat_fn=fn)
    assert info["merge_fallback"] and info["merge_fail"].startswith("positions"), info
    s_wall = info["merge_stats"]

    # 4. fallback=False keeps the lossy merge but still records the failure,
    #    so the fallback RATE can be measured without changing the answers.
    fn, _ = stub(short)
    out, info = merge_drafts(q, drafts, "", "", cfg=MergeConfig(fallback=False),
                             chat_fn=fn)
    assert out == short and not info["merge_fallback"] and info["merge_fail"], info

    # 5. thresholds are configurable, not hardcoded: a permissive ratio accepts
    #    the same digest the default rejects.
    fn, _ = stub(short)
    out, info = merge_drafts(q, drafts, "", "", chat_fn=fn,
                             cfg=MergeConfig(min_len_ratio=0.1, min_pos_ratio=0.0))
    assert out == short and not info["merge_fallback"], info

    # 6. N drafts: 3 in, one merge call, and n_drafts drives the draft plan.
    fn, calls = stub(good)
    merge_drafts(q, drafts, "", "", cfg=cfg, chat_fn=fn)
    assert len(calls) == 1, calls
    assert len(DRAFT_SPECS[:cfg.n_drafts]) == 3

    # 7. no forks retrieved -> only the plain draft exists; nothing to merge and
    #    no spurious merge call (the guard must not fire on a single draft).
    fn, calls = stub(good)
    out, parts = _merge_answer_v2(q, None, None, "", "", cfg=cfg, chat_fn=fn)
    assert len(calls) == 1 and out == draft_a and not parts["merge_fallback"], parts

    _persona_selftest()
    _selftest_random_fork()
    print("merge_v2 self-test OK")
    print(f"  lossless  : {s_good['merged_words']}w/{s_good['merged_positions']}pos "
          f"vs longest draft {s_good['max_draft_words']}w/"
          f"{s_good['max_draft_positions']}pos -> kept")
    print(f"  summarized: {s_short['merged_words']}w/"
          f"{s_short['merged_positions']}pos -> FALLBACK (short)")
    print(f"  collapsed : {s_wall['merged_words']}w/{s_wall['merged_positions']}pos "
          f"-> FALLBACK (positions; a length-only guard passes this)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Scout-injected answer generation")
    ap.add_argument("--question", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the merge_v2 guard with a stub endpoint")
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
    ap.add_argument("--n_drafts", type=int, default=MergeConfig.n_drafts,
                    help=f"merge_v2: drafts to merge (<= {len(DRAFT_SPECS)})")
    ap.add_argument("--merge_min_len", type=float, default=MergeConfig.min_len_ratio,
                    help="merge_v2 guard: merged words / longest draft's words")
    ap.add_argument("--merge_min_pos", type=float, default=MergeConfig.min_pos_ratio,
                    help="merge_v2 guard: merged positions / best draft's positions")
    ap.add_argument("--no_merge_fallback", action="store_true",
                    help="merge_v2: keep a lossy merge instead of concatenating "
                         "(measures the guard's fire rate without changing answers)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.question:
        ap.error("--question is required (or pass --selftest)")

    merge_cfg = MergeConfig(n_drafts=args.n_drafts,
                            min_len_ratio=args.merge_min_len,
                            min_pos_ratio=args.merge_min_pos,
                            fallback=not args.no_merge_fallback)

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
                 with_raw=args.show_raw, merge_cfg=merge_cfg)
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
