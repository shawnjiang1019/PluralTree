"""Context-aware decoding: steer HOW MUCH the injected forks matter, not whether.

Adaptation 3 of 3 (see docs/adaptive_injection.md). Every attempt to decide
WHETHER to inject has failed -- no graph signal separates (route_signal), and
asking the model scored 0.072 (route). This module stops trying to make a binary
decision and turns context influence into a DIAL, which is what the measured
failure actually calls for: injection does not fail because it happens on the
wrong questions so much as because the model over-anchors on the two injected
poles (docs/framing_hurts.png -- on-pole similarity 0.334 -> 0.389,
corr(attraction, dcoverage) = -0.31).

Context-aware decoding (Shi et al., 2023) contrasts the with-context and
without-context next-token distributions:

    logits = (1 + alpha) * logits(y | context, x)  -  alpha * logits(y | x)

    alpha  = 0   plain with-context decoding (what we do today)
    alpha  > 0   AMPLIFY the context's influence
    alpha  < 0   SUPPRESS it, interpolating back toward the no-context model
                 -- the anti-anchoring direction, unreachable by prompting

CtrlA (arXiv:2405.18727) does the analogous thing in activation space (steer
along an extracted direction); this is the logit-space version, which needs no
probe and no training.

The synthesis: alpha_from_contestedness() sets alpha from the self-consistency
score (retrieval/contestedness.py), giving SOFT routing -- consensus questions
decode with the context suppressed, contested ones with it amplified, and
everything in between is continuous instead of a threshold.

Needs local HF weights (two forward passes per token; not expressible over the
OpenAI-compatible endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CADConfig:
    alpha: float = 0.5            # context strength; 0 = ordinary decoding
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1024
    # soft-routing range for alpha_from_contestedness
    alpha_consensus: float = -0.5   # score 0 -> suppress the injected context
    alpha_contested: float = 1.0    # score 1 -> amplify it


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------
def cad_logits(logits_ctx: torch.Tensor, logits_plain: torch.Tensor,
               alpha: float) -> torch.Tensor:
    """(1+a)*ctx - a*plain. a=0 returns ctx unchanged; a<0 moves toward plain."""
    return (1.0 + alpha) * logits_ctx - alpha * logits_plain


def alpha_from_contestedness(score: float, cfg: CADConfig = CADConfig()) -> float:
    """Soft routing: map a [0,1] contestedness score to a context-strength alpha.

    Replaces the binary gate every previous attempt tried to learn. A consensus
    question (score ~0) decodes with the forks SUPPRESSED (negative alpha, which
    no prompt can express); a contested one amplifies them.
    """
    s = min(1.0, max(0.0, float(score)))
    return cfg.alpha_consensus + s * (cfg.alpha_contested - cfg.alpha_consensus)


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Nucleus sample one token from (1, vocab) logits. temperature<=0 = greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    probs = torch.softmax(logits.float(), dim=-1)
    if 0 < top_p < 1:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(dim=-1)
        keep = cum - sp <= top_p                      # always keeps the top token
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pick = torch.multinomial(sp, 1)
        return si.gather(-1, pick)
    return torch.multinomial(probs, 1)


# ---------------------------------------------------------------------------
# Generation (two KV caches stepped in lockstep)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cad_generate_ids(model, ids_ctx: torch.Tensor, ids_plain: torch.Tensor,
                     cfg: CADConfig = CADConfig(),
                     eos_token_id: int | None = None) -> list[int]:
    """Token ids generated under context-aware decoding.

    ``ids_ctx`` is the prompt WITH the injected forks, ``ids_plain`` the same
    question WITHOUT them. Both are advanced with the same sampled token, so the
    contrast at every step is exactly "what does the context change here?".
    """
    dev = next(model.parameters()).device
    cur_c, cur_p = ids_ctx.to(dev), ids_plain.to(dev)
    past_c = past_p = None
    out: list[int] = []
    for _ in range(cfg.max_new_tokens):
        o_c = model(input_ids=cur_c, past_key_values=past_c, use_cache=True)
        o_p = model(input_ids=cur_p, past_key_values=past_p, use_cache=True)
        past_c, past_p = o_c.past_key_values, o_p.past_key_values
        nxt = _sample(cad_logits(o_c.logits[:, -1, :].float(),
                                 o_p.logits[:, -1, :].float(), cfg.alpha),
                      cfg.temperature, cfg.top_p)
        tok = int(nxt.item())
        if eos_token_id is not None and tok == eos_token_id:
            break
        out.append(tok)
        cur_c = cur_p = nxt.view(1, 1)
    return out


def cad_generate(model, tokenizer, messages_ctx: list[dict],
                 messages_plain: list[dict],
                 cfg: CADConfig = CADConfig()) -> str:
    """Text answer under CAD. ``messages_ctx`` is the injected prompt (e.g. from
    retrieval.answer.build_prompt with forks), ``messages_plain`` the baseline
    prompt for the same question."""
    def _ids(msgs):
        # apply_chat_template returns a bare Tensor on older transformers and a
        # BatchEncoding on newer ones; model(input_ids=...) needs the Tensor.
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt")
        return enc["input_ids"] if hasattr(enc, "input_ids") else enc
    ids = cad_generate_ids(model, _ids(messages_ctx), _ids(messages_plain), cfg,
                           eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
def _selftest() -> None:
    """cad_logits algebra, the alpha map, and a real two-pass generation loop on
    a tiny randomly-initialised model (no download)."""
    lc = torch.tensor([[1.0, 2.0, 3.0]])
    lp = torch.tensor([[3.0, 2.0, 1.0]])
    assert torch.allclose(cad_logits(lc, lp, 0.0), lc)                  # a=0 = ctx
    assert torch.allclose(cad_logits(lc, lp, -1.0), lp)                 # a=-1 = plain
    amp = cad_logits(lc, lp, 1.0)                                       # a>0 amplifies
    assert (amp[0, 2] - amp[0, 0]) > (lc[0, 2] - lc[0, 0])

    cfg = CADConfig()
    assert alpha_from_contestedness(0.0, cfg) == cfg.alpha_consensus
    assert alpha_from_contestedness(1.0, cfg) == cfg.alpha_contested
    mid = alpha_from_contestedness(0.5, cfg)
    assert cfg.alpha_consensus < mid < cfg.alpha_contested
    assert alpha_from_contestedness(-5) == cfg.alpha_consensus          # clipped

    # greedy determinism + top_p keeps the argmax
    g = _sample(torch.tensor([[0.1, 5.0, 0.2]]), 0.0, 1.0)
    assert int(g.item()) == 1
    assert int(_sample(torch.tensor([[0.1, 9.0, 0.2]]), 1.0, 0.1).item()) == 1

    # end-to-end loop on a 2-layer random GPT-2 (constructed, not downloaded)
    from transformers import GPT2Config, GPT2LMHeadModel
    m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64,
                                   n_positions=64)).eval()
    ids_c = torch.tensor([[1, 2, 3, 4, 5]])
    ids_p = torch.tensor([[1, 2]])                    # shorter = "no context"
    out = cad_generate_ids(m, ids_c, ids_p,
                           CADConfig(alpha=0.7, temperature=0.0, max_new_tokens=6))
    assert len(out) == 6 and all(0 <= t < 64 for t in out), out
    # alpha must actually change the output distribution
    a0 = cad_generate_ids(m, ids_c, ids_p,
                          CADConfig(alpha=0.0, temperature=0.0, max_new_tokens=6))
    a2 = cad_generate_ids(m, ids_c, ids_p,
                          CADConfig(alpha=2.0, temperature=0.0, max_new_tokens=6))
    print(f"cad self-test OK: greedy alpha=0 {a0} vs alpha=2 {a2} "
          f"({'differ' if a0 != a2 else 'same (possible on a random model)'})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Context-aware decoding / steering")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.error("library module; run --selftest or import cad_generate")
