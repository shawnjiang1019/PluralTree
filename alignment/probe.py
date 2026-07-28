"""CtrlA-style contestedness probe on hidden states.

Adaptation 2 of 3 (see docs/adaptive_injection.md). CtrlA (arXiv:2405.18727)
extracts "honesty" and "confidence" DIRECTIONS from an LLM's internal
representations and triggers retrieval off the confidence monitor -- explicitly
because statistical output-level signals and self-reports are weaker. We want
the same trick for a different quantity: a CONTESTEDNESS direction.

Why this and not the alternatives we already burned:
  graph features  route_signal: nothing separates (corr ~+0.20 with human
                  contestedness -- the graph encodes demographic splits, not
                  disagreement).
  self-report     route: 0.072. The model SAYING "this is contested" is
                  hallucination-prone (the 2026 adaptive-RAG consensus).
  hidden states   read what the model internally represents, before it is
                  collapsed into a token. That is the part neither of the above
                  gets at.

Labels: do NOT train on graph divergence -- you would inherit its +0.20 ceiling.
Use the self-consistency score from retrieval/contestedness.py as a weak label
(it measures the model's own stance spread), or supply your own.

Needs local HF weights (hidden states are not exposed by the OpenAI-compatible
endpoint). The probe itself is a plain torch logistic regression -- no sklearn.

    python -m alignment.probe --labels contestedness.json \
        --model Qwen/Qwen2.5-7B-Instruct --layer -1 --out probe.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ---------------------------------------------------------------------------
# Probe (pure torch; offline-testable)
# ---------------------------------------------------------------------------
@dataclass
class ProbeConfig:
    layer: int = -1           # which hidden layer to read (-1 = last)
    lr: float = 1e-2
    epochs: int = 400
    l2: float = 1e-3          # weight decay -- probes overfit fast on small n
    standardize: bool = True


class LinearProbe(torch.nn.Module):
    """Logistic regression on mean-pooled hidden states = one direction in
    activation space, exactly CtrlA's construction (a linear readout), fit here
    by gradient descent rather than contrastive-pair differencing."""

    def __init__(self, d: int, cfg: ProbeConfig = ProbeConfig()):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)
        self.cfg = cfg
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def _norm(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu) / self.sd if self.cfg.standardize else X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self._norm(X)).squeeze(-1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self(X))

    @property
    def direction(self) -> torch.Tensor:
        """The contestedness direction in activation space (for steering)."""
        w = self.linear.weight.detach().squeeze(0)
        return w / w.norm().clamp_min(1e-12)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "LinearProbe":
        X, y = X.float(), y.float()
        if self.cfg.standardize:
            self.mu = X.mean(0)
            self.sd = X.std(0).clamp_min(1e-6)
        opt = torch.optim.Adam(self.linear.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.l2)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            loss = lossf(self(X), y)
            loss.backward()
            opt.step()
        return self


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC via the rank identity (ties averaged)."""
    s, y = scores.flatten().float(), labels.flatten().float()
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    # average ranks within ties
    uniq = torch.unique(s)
    if len(uniq) < len(s):
        for v in uniq:
            m = s == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cross_val_auc(X: torch.Tensor, y: torch.Tensor, folds: int = 5,
                  cfg: ProbeConfig = ProbeConfig(), seed: int = 0) -> dict:
    """K-fold CV AUC. With a few dozen questions a probe WILL fit the training
    set perfectly -- only the held-out number means anything."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    aucs, tr_aucs = [], []
    for f in range(folds):
        te = idx[f::folds]
        tr = torch.tensor([i for i in idx.tolist() if i not in set(te.tolist())])
        if len(tr) < 4 or len(te) < 2 or len(torch.unique(y[te])) < 2:
            continue
        p = LinearProbe(X.shape[1], cfg).fit(X[tr], y[tr])
        aucs.append(auc(p.predict_proba(X[te]), y[te]))
        tr_aucs.append(auc(p.predict_proba(X[tr]), y[tr]))
    aucs = [a for a in aucs if a == a]
    return {"cv_auc": sum(aucs) / len(aucs) if aucs else float("nan"),
            "train_auc": sum(tr_aucs) / len(tr_aucs) if tr_aucs else float("nan"),
            "folds": len(aucs)}


# ---------------------------------------------------------------------------
# Hidden-state extraction (needs local HF weights)
# ---------------------------------------------------------------------------
class HiddenExtractor:
    """Mean-pooled hidden states at one layer -- the probe's input features."""

    def __init__(self, model_name: str, layer: int = -1, device: str | None = None,
                 dtype: str = "bfloat16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype),
            output_hidden_states=True).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size: int = 8,
                 max_length: int = 512) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(self.device)
            hs = self.model(**enc).hidden_states[self.layer]      # (b, t, d)
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            outs.append(pooled.float().cpu())
        return torch.cat(outs)


# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Separable features -> high CV AUC; pure noise -> ~0.5 (and train >> cv,
    the overfit signature this CV is here to expose)."""
    torch.manual_seed(0)
    n, d = 120, 32
    y = (torch.arange(n) % 2).float()
    X = torch.randn(n, d) + y.unsqueeze(1) * torch.cat(
        [torch.ones(4), torch.zeros(d - 4)]) * 2.0
    good = cross_val_auc(X, y, folds=5)
    noise = cross_val_auc(torch.randn(n, d), y, folds=5)
    assert good["cv_auc"] > 0.85, good
    assert noise["cv_auc"] < 0.70, noise
    assert abs(auc(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 1.0])) - 1.0) < 1e-6
    p = LinearProbe(d).fit(X, y)
    assert abs(float(p.direction.norm()) - 1.0) < 1e-5
    print(f"probe self-test OK: separable cv_auc={good['cv_auc']:.3f} "
          f"(train {good['train_auc']:.3f}) | noise cv_auc={noise['cv_auc']:.3f} "
          f"(train {noise['train_auc']:.3f} <- overfit gap is the point)")


def main():
    ap = argparse.ArgumentParser(description="Contestedness probe on hidden states")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default=None,
                    help="contestedness.json from retrieval.contestedness")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--threshold", type=float, default=None,
                    help="binarize the weak label at this score (default: median)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="contestedness_probe.pt")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.labels:
        ap.error("pass --labels contestedness.json (or --selftest)")

    rows = json.load(open(args.labels, encoding="utf-8"))
    questions = [r["question"] for r in rows]
    scores = torch.tensor([r["score"] for r in rows]).float()
    thr = args.threshold if args.threshold is not None else float(scores.median())
    y = (scores > thr).float()
    print(f"{len(rows)} questions, label threshold {thr:.3f} -> "
          f"{int(y.sum())} contested / {int((1 - y).sum())} consensus")

    ex = HiddenExtractor(args.model, args.layer)
    X = ex(questions)
    print(f"hidden states: {tuple(X.shape)} from layer {args.layer} of {args.model}")

    stats = cross_val_auc(X, y, folds=args.folds)
    print(f"\ncross-validated AUC = {stats['cv_auc']:.3f}  "
          f"(train {stats['train_auc']:.3f}, {stats['folds']} folds)")
    print("  0.5 = the direction does not exist / not linearly readable here;\n"
          "  a large train-vs-cv gap means the probe memorized -- get more questions.")

    probe = LinearProbe(X.shape[1]).fit(X, y)
    torch.save({"state_dict": probe.state_dict(), "d": X.shape[1],
                "layer": args.layer, "model": args.model, "threshold": thr,
                "cv_auc": stats["cv_auc"]}, args.out)
    print(f"saved probe -> {args.out}")


if __name__ == "__main__":
    main()
