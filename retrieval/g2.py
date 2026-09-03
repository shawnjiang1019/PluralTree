"""G2: guided generation for output diversity (Ruan et al., EMNLP 2025).

Replication of arXiv:2511.00432, plus a graph-guided variant for PluralTree.

WHY THIS PAPER. Its NoveltyBench table independently reproduces our own failure:
the "Diverse Prompt" baseline -- i.e. ASKING the model to be diverse, exactly what
our `route` and `expand` conditions do -- raises Distinct 4.02 -> 6.58 but craters
Quality 7.97 -> 4.73. Our route (0.072) and expand (0.377) are that same collapse
measured on OvertonBench. G2's claim is that diversity has to be applied at the
LOGIT level, not the prompt level: it gets Distinct 5.80 at Quality 7.79 (-0.18).

THE METHOD. Three forward passes of the SAME model under different prompts,
combined per token:

    logits = z + alpha_t * (z+  -  z-)

    z     base generator   -- sees only the query, NEVER the prior answers
    z+    Diversity Guide  -- sees prior answers, prompted to DIVERGE from them
    z-    Dedupe Guide     -- sees prior answers, prompted to REPLICATE them

    alpha_t = theta  if H(softmax(z)) >= beta   (uncertain -> intervene)
            = 0      otherwise                  (confident -> leave it alone)

The entropy gate is what preserves quality: never perturb a token the base model
is already sure about. Answer 1 has no priors, so it is plain sampling.
Their ablation: the DEDUPE guide is more load-bearing than the diversity guide.

GRAPH-GUIDED VARIANT (ours). G2's diversity guide is generic ("be different").
PluralTree has a graph that says WHICH viewpoints exist and which are still
uncovered, so `graph_guided_messages` points the contrastive vector at a NAMED
missing subgroup position instead of at generic novelty:
    z+  "argue the position held by <uncovered subgroup>"
    z-  "repeat the positions already stated"

Needs local HF weights (three forward passes per token; not expressible over the
OpenAI-compatible endpoint). Sequential across answers by construction: answer i
conditions on answers < i.

    python -m retrieval.g2 --selftest
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from retrieval.cad import _sample                    # shared nucleus sampler


@dataclass
class G2Config:
    theta: float = 0.3            # intervention strength (paper sweeps .15/.3/.5/.7)
    beta: float = 0.1             # entropy gate, in nats (paper: fixed 0.1)
    k_repr: int = 3               # representative priors conditioned on
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 1024


# Base: no prior answers, no diversity instruction (the reference distribution).
G2_BASE = "Answer the question thoughtfully and concisely."

G2_DIVERSITY_GUIDE = (
    "You have already given the answers below to this question. Produce a "
    "genuinely DIFFERENT answer: take a different position, use different "
    "reasoning, and do not restate what has already been said. Stay accurate and "
    "responsive to the question."
)

# Deliberately extreme: the stronger this stream's repetition preference, the
# cleaner the (z+ - z-) contrast. Their ablation says this guide matters most.
G2_DEDUPE_GUIDE = (
    "EXACT REPLICATION MODE. You have already given the answers below. Reproduce "
    "them as closely as possible -- same positions, same reasoning, same wording. "
    "Do not introduce anything new."
)


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------
def g2_logits(z: torch.Tensor, z_plus: torch.Tensor, z_minus: torch.Tensor,
              alpha: float) -> torch.Tensor:
    """z + alpha*(z+ - z-).  alpha=0 returns the base logits unchanged."""
    if alpha == 0.0:
        return z
    return z + alpha * (z_plus - z_minus)


def token_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy (nats) of softmax(logits) for a (1, vocab) row."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(-(logp.exp() * logp).sum())


def alpha_gate(logits: torch.Tensor, cfg: G2Config) -> float:
    """theta where the base model is UNCERTAIN, else 0 (leave confident tokens).

    This is the quality-preserving half of G2: a fixed alpha applied everywhere
    intervenes on ~45% more tokens for the same diversity and measurably worse
    quality (their Fig. 3 ablation).
    """
    return cfg.theta if token_entropy(logits) >= cfg.beta else 0.0


def select_representatives(responses: Sequence[str], k: int,
                           embed_fn: Callable[[Sequence[str]], np.ndarray]
                           ) -> list[str]:
    """Center Selection: k maximally-dissimilar prior answers.

    Conditioning the guides on ALL prior answers bloats the context and is
    redundant; the paper greedily picks a spread-out subset instead (beats random
    selection at equal quality). Farthest-point traversal seeded with the most
    recent answer -- the one we most want to diverge from.
    """
    texts = [r for r in responses if r and r.strip()]
    if len(texts) <= k:
        return texts
    E = np.asarray(embed_fn(texts), dtype=float)
    chosen = [len(texts) - 1]                     # seed: most recent
    while len(chosen) < k:
        # pick the response farthest (in cosine distance) from everything chosen
        sims = (E @ E[chosen].T).max(axis=1)      # similarity to nearest chosen
        sims[chosen] = np.inf
        chosen.append(int(np.argmin(sims)))
    return [texts[i] for i in sorted(chosen)]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _prior_block(priors: Sequence[str]) -> str:
    return "\n\n".join(f"--- PREVIOUS ANSWER {i} ---\n{p}"
                       for i, p in enumerate(priors, 1))


def g2_messages(question: str, priors: Sequence[str]
                ) -> tuple[list[dict], list[dict], list[dict]]:
    """(base, diversity-guide, dedupe-guide) message lists for one generation."""
    base = [{"role": "system", "content": G2_BASE},
            {"role": "user", "content": question}]
    if not priors:
        return base, base, base                   # answer 1: no contrast to draw
    block = _prior_block(priors)
    plus = [{"role": "system", "content": G2_DIVERSITY_GUIDE},
            {"role": "user", "content": f"{block}\n\nQuestion: {question}"}]
    minus = [{"role": "system", "content": G2_DEDUPE_GUIDE},
             {"role": "user", "content": f"{block}\n\nQuestion: {question}"}]
    return base, plus, minus


def graph_guided_messages(question: str, priors: Sequence[str],
                          target_position: str | None
                          ) -> tuple[list[dict], list[dict], list[dict]]:
    """PluralTree variant: aim the contrast at a NAMED uncovered viewpoint.

    G2 pushes toward generic novelty. We know which positions exist (the graph's
    subgroup distributions) and which the prior answers have not expressed, so
    the diversity guide names one instead of saying "be different". The
    contrastive vector then points at a specific missing viewpoint.
    """
    if not target_position:
        return g2_messages(question, priors)
    base = [{"role": "system", "content": G2_BASE},
            {"role": "user", "content": question}]
    block = _prior_block(priors) if priors else "(none yet)"
    plus_sys = (
        "Answer the question by articulating this specific position, in the terms "
        "someone who genuinely holds it would use, so that they would feel fairly "
        f"represented:\n\n    {target_position}\n\n"
        "Do not restate the previous answers below and do not hedge between "
        "positions; make this one's case."
    )
    plus = [{"role": "system", "content": plus_sys},
            {"role": "user", "content": f"{block}\n\nQuestion: {question}"}]
    minus = [{"role": "system", "content": G2_DEDUPE_GUIDE},
             {"role": "user", "content": f"{block}\n\nQuestion: {question}"}]
    return base, plus, minus


# ---------------------------------------------------------------------------
# Generation: three KV caches stepped in lockstep
# ---------------------------------------------------------------------------
@torch.no_grad()
def g2_generate_ids(model, ids_base: torch.Tensor, ids_plus: torch.Tensor,
                    ids_minus: torch.Tensor, cfg: G2Config = G2Config(),
                    eos_token_id: int | None = None,
                    return_stats: bool = False):
    """One answer under G2 decoding. Streams are advanced with the SAME token, so
    the per-step contrast is exactly "what do the guides disagree about here?".

    No contrast, hence a single forward per token, when EITHER
      - ids_plus/ids_minus are the base ids (answer 1 has no priors), or
      - theta == 0 (the `baseline` condition).

    The theta==0 case matters for cost, not correctness: alpha is 0 at every
    step, so z is untouched and the guide passes cannot affect the sampled token
    -- but they still cost two forwards each. Skipping them leaves the output
    distribution bit-identical (the guides consume no RNG) while making baseline
    3x cheaper, which is most of this job's compute.
    """
    dev = next(model.parameters()).device
    contrast = (cfg.theta != 0.0
                and not (ids_plus is ids_base and ids_minus is ids_base))
    cur_b, cur_p, cur_m = ids_base.to(dev), ids_plus.to(dev), ids_minus.to(dev)
    past_b = past_p = past_m = None
    out: list[int] = []
    n_gated = 0

    for _ in range(cfg.max_new_tokens):
        o_b = model(input_ids=cur_b, past_key_values=past_b, use_cache=True)
        past_b = o_b.past_key_values
        z = o_b.logits[:, -1, :].float()

        alpha = 0.0
        if contrast:
            alpha = alpha_gate(z, cfg)
            if alpha != 0.0:
                o_p = model(input_ids=cur_p, past_key_values=past_p, use_cache=True)
                o_m = model(input_ids=cur_m, past_key_values=past_m, use_cache=True)
                past_p, past_m = o_p.past_key_values, o_m.past_key_values
                z = g2_logits(z, o_p.logits[:, -1, :].float(),
                              o_m.logits[:, -1, :].float(), alpha)
            else:
                # gate closed: still advance the guide caches so they stay aligned
                o_p = model(input_ids=cur_p, past_key_values=past_p, use_cache=True)
                o_m = model(input_ids=cur_m, past_key_values=past_m, use_cache=True)
                past_p, past_m = o_p.past_key_values, o_m.past_key_values
                n_gated += 1

        nxt = _sample(z, cfg.temperature, cfg.top_p)
        tok = int(nxt.item())
        if eos_token_id is not None and tok == eos_token_id:
            break
        out.append(tok)
        cur_b = cur_p = cur_m = nxt.view(1, 1)

    if return_stats:
        n = max(1, len(out))
        return out, {"n_tokens": len(out), "frac_gated_off": n_gated / n}
    return out


def g2_generate(model, tokenizer, question: str, priors: Sequence[str],
                cfg: G2Config = G2Config(),
                embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
                target_position: str | None = None) -> str:
    """Text answer under G2, conditioned on previously generated answers."""
    sel = list(priors)
    if embed_fn is not None and len(sel) > cfg.k_repr:
        sel = select_representatives(sel, cfg.k_repr, embed_fn)
    elif len(sel) > cfg.k_repr:
        sel = sel[-cfg.k_repr:]                   # fall back to most recent
    m_b, m_p, m_m = (graph_guided_messages(question, sel, target_position)
                     if target_position else g2_messages(question, sel))

    def _ids(msgs):
        # apply_chat_template returns a bare Tensor on older transformers and a
        # BatchEncoding on newer ones; model(input_ids=...) needs the Tensor.
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt")
        return enc["input_ids"] if hasattr(enc, "input_ids") else enc
    ids_b = _ids(m_b)
    # Vanilla G2 has nothing to contrast on answer 1 (no priors), so the guides
    # collapse to base. The GRAPH-GUIDED variant does: the target names a
    # position independently of what came before, and answer 1 seeds the priors
    # for the rest of the pool, so dropping the contrast there weakens every
    # later answer too.
    ids_p, ids_m = ((ids_b, ids_b) if not (sel or target_position)
                    else (_ids(m_p), _ids(m_m)))
    out = g2_generate_ids(model, ids_b, ids_p, ids_m, cfg,
                          eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out, skip_special_tokens=True)


def g2_generate_pool(model, tokenizer, question: str, n: int,
                     cfg: G2Config = G2Config(),
                     embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
                     targets: Sequence[str] | None = None) -> list[str]:
    """N answers, each conditioned on the ones before it (sequential by design).

    ``targets`` optionally supplies a per-answer graph position to aim at
    (the graph-guided variant); pass e.g. the top-N positions from
    ``alignment.reward.positions_from_subtree``.
    """
    pool: list[str] = []
    for i in range(n):
        tgt = targets[i] if targets and i < len(targets) else None
        pool.append(g2_generate(model, tokenizer, question, pool, cfg,
                                embed_fn, tgt))
    return pool


# ---------------------------------------------------------------------------
def _selftest() -> None:
    import re

    # --- logit algebra ---
    z = torch.tensor([[1.0, 2.0, 3.0]])
    zp = torch.tensor([[0.0, 0.0, 4.0]])
    zm = torch.tensor([[4.0, 0.0, 0.0]])
    assert torch.allclose(g2_logits(z, zp, zm, 0.0), z)          # gate closed
    g = g2_logits(z, zp, zm, 0.5)
    # z+ favours idx2, z- favours idx0 -> contrast must push mass 0 -> 2
    assert g[0, 2] > z[0, 2] and g[0, 0] < z[0, 0]

    # --- entropy gate ---
    peaked = torch.tensor([[50.0, 0.0, 0.0]])                    # near-deterministic
    flat = torch.tensor([[1.0, 1.0, 1.0]])
    cfg = G2Config(theta=0.3, beta=0.1)
    assert token_entropy(peaked) < 0.1 < token_entropy(flat)
    assert alpha_gate(peaked, cfg) == 0.0 and alpha_gate(flat, cfg) == 0.3

    # --- center selection: must span the clusters, not pick near-duplicates ---
    rng = np.random.default_rng(0)
    vocab: dict[str, int] = {}

    def fake_embed(texts):
        V = np.zeros((len(texts), 32))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                V[i, vocab.setdefault(w, int(rng.integers(0, 32)))] += 1.0
            nrm = np.linalg.norm(V[i])
            if nrm:
                V[i] /= nrm
        return V

    priors = ["alpha alpha alpha", "alpha alpha alpha", "beta beta beta",
              "gamma gamma gamma"]
    sel = select_representatives(priors, 3, fake_embed)
    assert len(sel) == 3 and len(set(sel)) == 3, sel      # no duplicate picked

    # --- end-to-end on a tiny randomly-initialised model (no download) ---
    from transformers import GPT2Config, GPT2LMHeadModel
    m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64,
                                   n_positions=128)).eval()
    ids_b = torch.tensor([[1, 2, 3]])
    ids_p = torch.tensor([[1, 2, 3, 4, 5, 6]])
    ids_m = torch.tensor([[1, 2, 3, 7, 8, 9]])
    c = G2Config(theta=0.5, beta=0.1, temperature=0.0, max_new_tokens=8)
    out, stats = g2_generate_ids(m, ids_b, ids_p, ids_m, c, return_stats=True)
    assert len(out) == 8 and all(0 <= t < 64 for t in out)

    # theta=0 must reproduce plain base decoding (no contrast at all)
    base_only = g2_generate_ids(m, ids_b, ids_b, ids_b,
                                G2Config(temperature=0.0, max_new_tokens=8))
    off = g2_generate_ids(m, ids_b, ids_p, ids_m,
                          G2Config(theta=0.0, temperature=0.0, max_new_tokens=8))
    assert off == base_only, (off, base_only)

    # ...and a large enough theta must actually steer the output away from base.
    # (On this RANDOM model the guides are near-noise, so a realistic theta=0.3
    # need not flip an argmax; a real instruction-following model produces a far
    # more targeted (z+ - z-). Assert the mechanism, not a specific theta.)
    steered = g2_generate_ids(m, ids_b, ids_p, ids_m,
                              G2Config(theta=8.0, temperature=0.0, max_new_tokens=8))
    assert steered != base_only, (steered, base_only)
    print(f"g2 self-test OK: base {base_only} -> steered {steered}; "
          f"theta=0 reproduces base; gate closed on "
          f"{stats['frac_gated_off']:.0%} of tokens")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="G2 guided generation (2511.00432)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.error("library module; run --selftest or import g2_generate_pool")
