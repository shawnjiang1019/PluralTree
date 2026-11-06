"""Train the simple subgroup encoder and save h_all for retrieval.

Inputs per subgroup leaf: q (item text), g ("attribute: group"), and, for
--inputs qgp, p (the answer barycenter). --inputs qg drops the answers, so the
model must PREDICT disagreement from text: that is the variant to evaluate on
held-out items (--split_by item). Loss: the disagreement rank loss plus the
hierarchy margin. Selection: val Spearman(geodesic, D).

Log lines match train.py, so plot_train_log.py reads these runs too.

    python scripts/train/train_subgroup.py --inputs qgp --text_feat feats_opinionqa.pt --save_embeddings embeddings_opinionqa_simple_qgp.pt --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple subgroup encoder")
    ap.add_argument("--dataset", default="opinionqa")
    ap.add_argument("--inputs", choices=["qgp", "qg"], default="qgp")
    ap.add_argument("--split_by", choices=["axis", "item"], default="axis")
    ap.add_argument("--text_feat", required=True)
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--d_out", type=int, default=64)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--tangent_clip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dis_kappa", type=float, default=1.0)
    ap.add_argument("--dis_lam", type=float, default=0.5)
    ap.add_argument("--lambda_struct", type=float, default=0.1)
    ap.add_argument("--struct_margin", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42, help="graph split seed: must match retrieval")
    ap.add_argument("--save_embeddings", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from data.loaders.graphs import load_graph
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.scout import load_or_compute_text_feat
    from training.losses import disagreement_rank_loss
    from training.subgroup_encoder import SubgroupEncoder, full_positions, tree_levels
    from training.trainer import disagreement_pairs, spearman_t

    print(f"EMB={args.save_embeddings} inputs={args.inputs} split_by={args.split_by} "
          f"tclip={args.tangent_clip}")
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    n = len(graph.id_to_entity)
    feats = load_or_compute_text_feat(graph, args.dataset, args.text_feat,
                                      model_name=args.embed_model).float()

    # Leaf inputs. Names are op:{qkey}:{attr}:{group}.
    leaf_ids = sorted(v for v in graph.opinion_dist
                      if graph.id_to_entity[v].startswith("op:"))
    q_ids, g_text = [], []
    for v in leaf_ids:
        qkey, attr, group = graph.id_to_entity[v][3:].rsplit(":", 2)
        q_ids.append(graph.entity_vocab[f"q:{qkey}"])
        g_text.append(f"{attr}: {group}")
    from sentence_transformers import SentenceTransformer
    uniq = sorted(set(g_text))
    enc_g = SentenceTransformer(args.embed_model).encode(uniq, convert_to_tensor=True,
                                                         show_progress_bar=False).cpu().float()
    g_of = {t: i for i, t in enumerate(uniq)}
    parts = [feats[q_ids], enc_g[[g_of[t] for t in g_text]]]
    if args.inputs == "qgp":
        parts.append(feats[leaf_ids])
    x = torch.cat(parts, dim=-1).to(dev)
    leaf_t = torch.tensor(leaf_ids, dtype=torch.long, device=dev)
    print(f"  {len(leaf_ids)} subgroups, input dim {x.shape[1]}, {len(uniq)} group labels")

    man = PoincareBall(c=args.curvature).to(dev)
    man.tangent_cap = args.tangent_clip
    c = float(man.c)
    model = SubgroupEncoder(x.shape[1], args.d_out, man, args.d_hidden).to(dev)
    levels = [(a.to(dev), b.to(dev)) for a, b in
              tree_levels(graph.children_indices, graph.entity_types)]

    dis = {s: tuple(t.to(dev) for t in v) for s, v in disagreement_pairs(
        graph, args.dis_lam, args.embed_model, by_item=args.split_by == "item").items()}
    print(f"  disagreement pairs (by {args.split_by}): " +
          " | ".join(f"{s} {len(v[2])}" for s, v in dis.items()) + f" | lam={args.dis_lam}")
    ch = torch.cat([a for a, _ in levels])
    par = torch.cat([b for _, b in levels])

    def encode():
        return full_positions(model(x), leaf_t, levels, n, c)

    def rho(split: str, h=None) -> float:
        i, j, d = dis[split]
        with torch.no_grad():
            h = encode() if h is None else h
            return spearman_t(man.distance(h[i], h[j]).squeeze(-1), d)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ti, tj, td = dis["train"]
    best, t0 = float("-inf"), time.time()
    for step in range(1, args.steps + 1):
        h = encode()
        a = torch.randint(len(td), (args.batch_size,), device=dev)
        b = torch.randint(len(td), (args.batch_size,), device=dev)
        l_dis = disagreement_rank_loss(h, ti[a], tj[a], td[a], ti[b], tj[b], td[b],
                                       man, args.dis_kappa)
        e = torch.randint(len(ch), (args.batch_size,), device=dev)
        neg = torch.randint(n, (args.batch_size,), device=dev)
        d_pos = man.distance(h[ch[e]], h[par[e]]).squeeze(-1)
        d_neg = man.distance(h[ch[e]], h[neg]).squeeze(-1)
        l_str = F.relu(args.struct_margin + d_pos - d_neg).mean()
        loss = l_dis + args.lambda_struct * l_str
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | lp 0.0000 | "
                  f"struct {l_str.item():.4f} | div 0.0000 | dec 0.0000 | "
                  f"dis {l_dis.item():.4f} | gate_bias 0.00")
        if step % args.eval_every == 0:
            with torch.no_grad():
                h_e = encode()
                r = man.c.sqrt() * h_e.norm(dim=-1)
                g = torch.Generator(device="cpu").manual_seed(0)
                i = torch.randint(0, n, (512,), generator=g).to(dev)
                j = torch.randint(0, n, (512,), generator=g).to(dev)
                sd = man.distance(h_e[i], h_e[j]).squeeze(-1).std()
            print(f"  [eval] step {step} | MRR 0.0000 | H@1 0.0000 | H@10 0.0000")
            print(f"  [ball]  rho mean {r.mean():.4f} | p50 {r.median():.4f} | "
                  f"frac>0.9 {(r > 0.9).float().mean():.3f} | pair-dist sd {sd:.3f}")
            v = rho("val", h_e)
            print(f"  [dis]   val Spearman(geodesic, D) {v:+.4f} (held-out {args.split_by}s)")
            if v > best:
                best = v
                torch.save(h_e.detach().cpu(), args.save_embeddings)

    h_best = torch.load(args.save_embeddings, map_location=dev)
    test = rho("test", h_best)
    print(f"\nDIS | inputs={args.inputs} split={args.split_by} | best_val_dis={best:+.4f} "
          f"test_dis={test:+.4f}")
    print(f"Done. Wrote {args.save_embeddings} {tuple(h_best.shape)} "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
