"""Linear/MLP probe — how much structure is *in* the embedding (the ceiling).

For each structural fact the Map Reader is trained to decode, fit a simple probe
directly on ``log_map_zero(h)`` (the tangent-space view the LatentBridge sees) and
report its held-out accuracy next to the majority-class prior. This is the
information ceiling: the most a direct decoder can extract from the vector.

Decision rule (see the project notes):
    probe ~= prior            -> EMBEDDING problem (info isn't in the vector).
    probe >> prior, MR low     -> INTERPRETATION problem (LM isn't reading it).
    probe >> prior, MR ~= probe-> at ceiling; improve the embedding to gain more.

The probe is a near-best-case decoder with *direct* access to the raw vector, so
a low probe is a hard ceiling the (more indirect) Map Reader cannot exceed.

Run on the SAME held-out split as the Map Reader for an apples-to-apples ceiling:
    python scripts/probe_embeddings.py --embeddings embeddings_up.pt \
        --dataset wn18rr --data_dir data/wn18rr --split runs/holdout.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from pluraltree.manifolds.poincare import PoincareBall
from evaluation.intrinsic.structure_metrics import (
    _parents_from_children,
    _depths,
    _level_k_ancestors,
)


# ---------------------------------------------------------------------------
# Fact extraction (identical definitions to generate_map_reader_sft.py)
# ---------------------------------------------------------------------------
def _role(nid, ci, parents) -> str:
    if not parents[nid]:
        return "root"
    if not ci[nid]:
        return "leaf"
    return "junction"


def _branching(nid, ci, depth, levels: int = 2):
    counts = [0] * levels
    queue, seen = [(nid, 0)], 0
    while queue and seen < 256:
        v, d = queue.pop(0)
        seen += 1
        for c in ci[v]:
            rd = depth[c] - depth[nid]
            if 1 <= rd <= levels:
                counts[rd - 1] += 1
            if rd < levels:
                queue.append((c, d + 1))
    return tuple(counts)


def build_facts(graph, h_all, manifold):
    """Return {fact: (labels, kind)} where kind is 'cls' or 'reg'."""
    ci = graph.children_indices
    parents = _parents_from_children(ci)
    depth = _depths(ci, parents)
    lvl1 = _level_k_ancestors(ci, parents, depth, 1)
    n = len(ci)

    rho = (manifold.c.sqrt() * h_all.norm(dim=1)).tolist()
    facts = {
        "depth":     ([depth[i] for i in range(n)], "cls"),
        "role":      ([_role(i, ci, parents) for i in range(n)], "cls"),
        "children":  ([len(ci[i]) for i in range(n)], "cls"),
        "branching": ([_branching(i, ci, depth) for i in range(n)], "cls"),
        "domain":    ([(sorted(lvl1[i])[0] if lvl1[i] else i) for i in range(n)], "cls"),
        "rho":       (rho, "reg"),
    }
    return facts


# ---------------------------------------------------------------------------
# Probes (torch-only; no sklearn dependency)
# ---------------------------------------------------------------------------
def _standardize(X, idx_train):
    mu = X[idx_train].mean(0, keepdim=True)
    sd = X[idx_train].std(0, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd


def _make_probe(d_in, d_out, hidden):
    if hidden and hidden > 0:
        return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))
    return nn.Linear(d_in, d_out)


def train_classifier(X, y, idx_tr, idx_va, *, hidden=0, epochs=80, lr=1e-2,
                     batch_size=4096, device="cpu"):
    """Fit a probe to predict class y; return (val_acc, prior_acc)."""
    classes = sorted(set(y[i] for i in idx_tr))
    cmap = {c: k for k, c in enumerate(classes)}
    yt = torch.tensor([cmap.get(y[i], -1) for i in range(len(y))], dtype=torch.long)
    Xtr, ytr = X[idx_tr].to(device), yt[idx_tr].to(device)
    Xva = X[idx_va].to(device)
    yva = yt[idx_va]

    model = _make_probe(X.shape[1], len(classes), hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ntr = Xtr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(ntr, device=device)
        for s in range(0, ntr, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(1).cpu()
    val_acc = float((pred == yva).float().mean())

    # prior = predict the train-majority class on val
    maj = max(classes, key=lambda c: sum(1 for i in idx_tr if y[i] == c))
    prior = float(sum(1 for i in idx_va if y[i] == maj) / max(1, len(idx_va)))
    return val_acc, prior


def train_regressor(X, y, idx_tr, idx_va, *, hidden=0, epochs=80, lr=1e-2,
                    tol=0.05, batch_size=4096, device="cpu"):
    """Fit a probe to predict float y; return (val_mae, prior_mae, thr_acc, prior_thr)."""
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    Xtr, ytr = X[idx_tr].to(device), yt[idx_tr].to(device)
    Xva = X[idx_va].to(device)
    yva = yt[idx_va]

    model = _make_probe(X.shape[1], 1, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    ntr = Xtr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(ntr, device=device)
        for s in range(0, ntr, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xva).cpu()
    mae = float((pred - yva).abs().mean())
    thr = float(((pred - yva).abs() <= tol).float().mean())
    mean_tr = ytr.mean().item()
    prior_mae = float((yva - mean_tr).abs().mean())
    prior_thr = float(((yva - mean_tr).abs() <= tol).float().mean())
    return mae, prior_mae, thr, prior_thr


def main():
    ap = argparse.ArgumentParser(description="Probe an embedding for structural facts")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--dataset", choices=["wn18rr", "culturalbench", "globalopinionqa", "grailqa"],
                    default="wn18rr")
    ap.add_argument("--data_dir", default="data/wn18rr")
    ap.add_argument("--split", default=None,
                    help="train/val node split JSON (reuse the Map Reader holdout). "
                         "If absent, a fresh split is made with --val_frac.")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=0,
                    help="hidden width for the MLP probe (0 = linear only). When >0, "
                         "BOTH linear and MLP are reported.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--max_train", type=int, default=20000,
                    help="cap #train nodes used to fit the probe (0 = all). A linear "
                         "ceiling estimate is stable with a few thousand; this is the "
                         "main speed knob on large graphs like WN18RR.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto",
                    help="'auto' uses cuda if available, else cpu.")
    ap.add_argument("--json", default=None, help="also write per-fact results to this JSON "
                    "(for compare_probe_mapreader.py)")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.dataset == "wn18rr":
        from data.loaders.wordnet import load_wn18rr
        graph = load_wn18rr(data_dir=args.data_dir, split_seed=args.seed, leakage_safe=True)
    elif args.dataset == "globalopinionqa":
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
    elif args.dataset == "grailqa":
        from data.loaders.grailqa import load_grailqa
        graph = load_grailqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.loaders.culturalbench import load_culturalbench
        graph = load_culturalbench(split_seed=args.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    X = manifold.log_map_zero(h_all).detach().float()    # tangent view (bridge input)

    n = len(graph.children_indices)
    if args.split and os.path.exists(args.split):
        with open(args.split, encoding="utf-8") as f:
            val_ids = set(json.load(f)["val"])
        print(f"  loaded split: {len(val_ids)} val nodes from {args.split}")
    else:
        order = list(range(n))
        random.Random(args.seed).shuffle(order)
        val_ids = set(order[: int(n * args.val_frac)])
        print(f"  fresh split: {len(val_ids)} val nodes")
    idx_va = [i for i in range(n) if i in val_ids]
    idx_tr = [i for i in range(n) if i not in val_ids]

    X = _standardize(X, idx_tr)
    facts = build_facts(graph, h_all, manifold)

    # Subsample train nodes for speed (the probe ceiling is stable with a few k).
    n_tr_full = len(idx_tr)
    if args.max_train and n_tr_full > args.max_train:
        idx_tr = random.Random(args.seed).sample(idx_tr, args.max_train)

    print(f"\nProbe ceiling on {args.dataset}  ({len(idx_tr)}/{n_tr_full} train, "
          f"{len(idx_va)} val nodes; device={args.device})")
    hdr = f"{'fact':<10}{'prior':>8}{'linear':>9}"
    if args.hidden:
        hdr += f"{'mlp':>9}"
    hdr += "   verdict"
    print(hdr)
    print("-" * len(hdr))

    results: dict[str, dict] = {}
    for fact, (labels, kind) in facts.items():
        if kind == "reg":
            mae, prior_mae, thr, prior_thr = train_regressor(
                X, labels, idx_tr, idx_va, epochs=args.epochs,
                batch_size=args.batch_size, device=args.device)
            line = (f"{fact:<10}{prior_thr:>8.2f}{thr:>9.2f}")
            rec = {"kind": "reg", "prior": prior_thr, "linear": thr,
                   "mae": mae, "prior_mae": prior_mae}
            if args.hidden:
                mae_h, _, thr_h, _ = train_regressor(
                    X, labels, idx_tr, idx_va, hidden=args.hidden,
                    epochs=args.epochs, batch_size=args.batch_size, device=args.device)
                line += f"{thr_h:>9.2f}"
                rec["mlp"] = thr_h
            line += f"   (MAE {mae:.3f} vs prior {prior_mae:.3f})"
            print(line)
            rec["best"] = max(rec["linear"], rec.get("mlp", rec["linear"]))
            results[fact] = rec
            continue

        lin, prior = train_classifier(X, labels, idx_tr, idx_va, epochs=args.epochs,
                                      batch_size=args.batch_size, device=args.device)
        best = lin
        line = f"{fact:<10}{prior:>8.2f}{lin:>9.2f}"
        rec = {"kind": "cls", "prior": prior, "linear": lin}
        if args.hidden:
            mlp, _ = train_classifier(X, labels, idx_tr, idx_va, hidden=args.hidden,
                                      epochs=args.epochs, batch_size=args.batch_size,
                                      device=args.device)
            best = max(lin, mlp)
            line += f"{mlp:>9.2f}"
            rec["mlp"] = mlp
        gap = best - prior
        verdict = ("EMBEDDING (info absent)" if gap < 0.05
                   else "info present -> compare to Map Reader")
        print(line + f"   {verdict}")
        rec["best"] = best
        results[fact] = rec

    print("\nRead: probe~prior => embedding problem; probe>>prior => info is there, "
          "so a low Map Reader score is an interpretation problem.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote probe results to {args.json}")


if __name__ == "__main__":
    main()
