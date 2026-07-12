"""Evaluate a trained Map Reader: per-fact latent-reading via causal ablation.

Validation protocol from pluraltree/sft/map_reader.md. Runs on **held-out
(val-split) nodes only** so a correct answer cannot come from a memorized
node->caption mapping. For each structural fact it reports accuracy under five
conditions and the ablation deltas with bootstrap CIs:

    true            node's own latent           (the model)
    zero            zeroed latent               (does the latent matter at all?)
    shuffle         another val node's latent   (does the *specific* latent matter?)
    text-only       no latent                   (lower bound: prompt/prior)
    facts-in-prompt caption in the prompt        (upper bound: LM format ceiling)

    Delta_zero(f)    = acc_f(true) - acc_f(zero)
    Delta_shuffle(f) = acc_f(true) - acc_f(shuffle)

A large positive delta on a fact means the latent genuinely carries it. A
near-zero delta on a non-norm fact (e.g. domain) is the failure signal: the
model is memorizing, not reading.

Usage:
    python scripts/eval_map_reader.py --run runs/map_reader \
        --embeddings runs/h_all.pt --data data/map_reader_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.sft.latent_bridge import LatentBridge


def _norm(s: str) -> str:
    return s.strip().lower().rstrip(".")


def _correct(fact: str, pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if fact == "rho":                       # numeric, allow small tolerance
        try:
            return abs(float(p.split()[0]) - float(g)) <= 0.05
        except (ValueError, IndexError):
            return False
    return p.startswith(g) or g in p


def _derangement(n: int, seed: int) -> np.ndarray:
    """A permutation with no fixed points (so shuffle never returns own latent)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if n < 2 or not np.any(perm == idx):
            return perm
    perm = np.roll(idx, 1)                   # fallback: guaranteed derangement
    return perm


@torch.no_grad()
def generate(model, embed_layer, tok, prompts, soft, device, max_new=16, bs=16):
    """Greedy-decode answers. ``soft`` is (B, n_soft, d) or None (text-only)."""
    eos = tok.eos_token_id
    n_soft = 0 if soft is None else soft.shape[1]
    out_text = []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s : s + bs]
        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
        max_len = max(len(x) for x in ids)
        d = embed_layer.weight.shape[1]
        embeds, attn = [], []
        for j, x in enumerate(ids):
            pad = max_len - len(x)                          # left-pad for generate
            row = [torch.zeros(pad, d, device=device, dtype=embed_layer.weight.dtype)]
            if n_soft:
                row.append(soft[s + j].to(embed_layer.weight.dtype))
            row.append(embed_layer(torch.tensor(x, device=device)))
            embeds.append(torch.cat(row, dim=0))
            attn.append(torch.tensor([0] * pad + [1] * (n_soft + len(x)), device=device))
        inputs_embeds = torch.stack(embeds)
        attn = torch.stack(attn)
        gen = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                             max_new_tokens=max_new, do_sample=False, pad_token_id=eos)
        out_text.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out_text


def boot_ci(a: np.ndarray, b: np.ndarray, n=2000, seed=0):
    """Bootstrap CI for mean(a) - mean(b) (paired over examples)."""
    rng = np.random.default_rng(seed)
    n_ex = len(a)
    if n_ex == 0:
        return float("nan"), (float("nan"), float("nan"))
    diffs = [(a[idx].mean() - b[idx].mean())
             for idx in (rng.integers(0, n_ex, n_ex) for _ in range(n))]
    return float(a.mean() - b.mean()), (float(np.percentile(diffs, 2.5)),
                                        float(np.percentile(diffs, 97.5)))


def main():
    ap = argparse.ArgumentParser(description="Per-fact ablation eval for the Map Reader")
    ap.add_argument("--run", required=True, help="dir from train_map_reader.py")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--data", required=True, help="map_reader_sft.jsonl (with split tags)")
    ap.add_argument("--max_eval", type=int, default=400, help="cap val QA examples per fact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write per-fact results to this JSON "
                    "(for compare_probe_mapreader.py)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(args.run, "bridge_config.json")))

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": device})
    model = PeftModel.from_pretrained(base, args.run).eval()
    embed_layer = model.get_input_embeddings()

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    h_all = h_all.to(device=device, dtype=torch.bfloat16)
    manifold = PoincareBall(c=cfg["curvature"])
    bridge = LatentBridge(manifold, d_latent=cfg["d_latent"], d_model=cfg["d_model"],
                          n_tokens=cfg["n_tokens"]).to(device=device, dtype=torch.bfloat16)
    bridge.load_state_dict(torch.load(os.path.join(args.run, "latent_bridge.pt")))
    bridge.eval()

    # --- val QA records + node->caption map (for facts-in-prompt) -----------
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    cap_of = {r["node_id"]: r["target"] for r in rows
              if r["kind"] == "decode" and r.get("split") == "val"}
    by_fact = defaultdict(list)
    for r in rows:
        if r["kind"] == "qa" and r.get("split") == "val":
            by_fact[r["fact"]].append(r)

    print(f"{'fact':<10} {'true':>6} {'zero':>6} {'shuf':>6} {'text':>6} {'fip':>6} "
          f"{'d_zero[CI]':>22} {'d_shuf[CI]':>22}")
    results: dict[str, dict] = {}
    for fact, recs in sorted(by_fact.items()):
        recs = recs[: args.max_eval]
        nids = torch.tensor([r["node_id"] for r in recs], device=device)
        soft_true = bridge(h_all[nids])
        perm = _derangement(len(recs), args.seed)
        soft_shuf = bridge(h_all[nids[torch.tensor(perm, device=device)]])
        soft_zero = torch.zeros_like(soft_true)

        q = [f"{r['prompt']}\nAnswer:" for r in recs]
        q_fip = [f"Context: {cap_of.get(r['node_id'], '')}\n{r['prompt']}\nAnswer:" for r in recs]
        gold = [r["target"] for r in recs]

        conds = {
            "true": generate(model, embed_layer, tok, q, soft_true, device),
            "zero": generate(model, embed_layer, tok, q, soft_zero, device),
            "shuffle": generate(model, embed_layer, tok, q, soft_shuf, device),
            "text": generate(model, embed_layer, tok, q, None, device),
            "fip": generate(model, embed_layer, tok, q_fip, None, device),
        }
        acc = {k: np.array([_correct(fact, p, g) for p, g in zip(v, gold)], dtype=float)
               for k, v in conds.items()}
        dz, ciz = boot_ci(acc["true"], acc["zero"], seed=args.seed)
        ds, cis = boot_ci(acc["true"], acc["shuffle"], seed=args.seed)
        print(f"{fact:<10} {acc['true'].mean():>6.2f} {acc['zero'].mean():>6.2f} "
              f"{acc['shuffle'].mean():>6.2f} {acc['text'].mean():>6.2f} "
              f"{acc['fip'].mean():>6.2f} "
              f"{dz:>+6.2f}[{ciz[0]:+.2f},{ciz[1]:+.2f}] "
              f"{ds:>+6.2f}[{cis[0]:+.2f},{cis[1]:+.2f}]")
        results[fact] = {
            "true": float(acc["true"].mean()), "zero": float(acc["zero"].mean()),
            "shuffle": float(acc["shuffle"].mean()), "text": float(acc["text"].mean()),
            "fip": float(acc["fip"].mean()),
            "d_zero": dz, "d_zero_ci": list(ciz),
            "d_shuf": ds, "d_shuf_ci": list(cis), "n": len(recs),
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote Map Reader results to {args.json}")


if __name__ == "__main__":
    main()
