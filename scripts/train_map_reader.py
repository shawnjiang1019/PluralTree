"""Train the Map Reader: teach a 7-8B LM to read injected Poincaré latents.

Phase 1 of pluraltree/sft/plan.md. The LM is frozen + 4-bit (QLoRA); only the
LoRA adapter and the LatentBridge are trained. Each example injects one node's
latent as ``n_tokens`` soft tokens prepended to the text, and the LM is trained
(next-token CE on the target span only) to decode the structural fact.

Why this is the Map Reader and not the Reasoner: input is a single latent + a
structural prompt; target is the geometry (depth / role / rho / branching). It
makes the model *able* to read a latent before Phase 2 forces it to *use* one.

Requires (not in the base repo's deps):
    pip install transformers peft bitsandbytes accelerate

Usage (single GPU, ~24GB with a 7-8B model in 4-bit):
    python scripts/train_map_reader.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --embeddings runs/h_all.pt \
        --data data/map_reader_sft.jsonl \
        --out runs/map_reader --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.sft.latent_bridge import LatentBridge


class MapReaderDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def build_collate(tokenizer, n_soft, device, dtype):
    """Collate -> dict of (inputs_embeds, attention_mask, labels, node_ids).

    Layout per example: [n_soft placeholder] + prompt_ids + target_ids + eos.
    Soft-token embeddings are spliced in during the forward pass; here we only
    reserve their positions (mask=1, label=-100).
    """
    eos = tokenizer.eos_token_id

    def collate(batch):
        prompt_ids, target_ids, node_ids = [], [], []
        for r in batch:
            p = tokenizer(f"{r['prompt']}\nAnswer:", add_special_tokens=False)["input_ids"]
            t = tokenizer(f" {r['target']}", add_special_tokens=False)["input_ids"] + [eos]
            prompt_ids.append(p)
            target_ids.append(t)
            node_ids.append(r["node_id"])

        lengths = [n_soft + len(p) + len(t) for p, t in zip(prompt_ids, target_ids)]
        max_len = max(lengths)

        text_ids = torch.full((len(batch), max_len - n_soft), eos, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, (p, t) in enumerate(zip(prompt_ids, target_ids)):
            txt = p + t
            text_ids[i, : len(txt)] = torch.tensor(txt, dtype=torch.long)
            attn[i, : n_soft + len(txt)] = 1
            # labels: -100 over soft tokens + prompt, real ids over target
            labels[i, n_soft + len(p) : n_soft + len(p) + len(t)] = torch.tensor(t)

        return {
            "text_ids": text_ids.to(device),
            "attention_mask": attn.to(device),
            "labels": labels.to(device),
            "node_ids": torch.tensor(node_ids, dtype=torch.long, device=device),
        }

    return collate


def main():
    ap = argparse.ArgumentParser(description="Train the Map Reader (QLoRA + LatentBridge)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--embeddings", required=True, help=".pt of trained h_all on the ball")
    ap.add_argument("--data", required=True, help="map_reader_sft.jsonl")
    ap.add_argument("--out", required=True, help="output dir for adapter + bridge")
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--n_soft", type=int, default=4, help="soft tokens per latent")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # --- frozen 4-bit LM ---------------------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    embed_layer = model.get_input_embeddings()
    d_model = model.config.hidden_size

    # --- latent bridge -----------------------------------------------------
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    h_all = h_all.to(device=device, dtype=torch.bfloat16)
    manifold = PoincareBall(c=args.curvature)
    bridge = LatentBridge(manifold, d_latent=h_all.shape[1], d_model=d_model,
                          n_tokens=args.n_soft).to(device=device, dtype=torch.bfloat16)

    # --- data --------------------------------------------------------------
    ds = MapReaderDataset(args.data)
    collate = build_collate(tok, args.n_soft, device, torch.bfloat16)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    params = list(bridge.parameters()) + [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    model.train()
    bridge.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            # soft-token embeds from the node latents
            h = h_all[batch["node_ids"]]                       # (B, d_latent)
            soft = bridge(h).to(torch.bfloat16)                # (B, n_soft, d_model)
            text_emb = embed_layer(batch["text_ids"])          # (B, T, d_model)
            inputs_embeds = torch.cat([soft, text_emb], dim=1)

            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {out.loss.item():.4f}")
            step += 1

    # --- save --------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)                            # LoRA adapter
    torch.save(bridge.state_dict(), os.path.join(args.out, "latent_bridge.pt"))
    with open(os.path.join(args.out, "bridge_config.json"), "w") as f:
        json.dump({"d_latent": h_all.shape[1], "d_model": d_model,
                   "n_tokens": args.n_soft, "curvature": args.curvature,
                   "base": args.base}, f, indent=2)
    print(f"Saved adapter + bridge to {args.out}")


if __name__ == "__main__":
    main()
