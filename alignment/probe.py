"""CtrlA-style contestedness probe on hidden states.

Adaptation 2 of 3 (see docs/adaptive_injection.md). CtrlA (arXiv:2405.18727)
extracts "honesty" and "confidence" DIRECTIONS from an LLM's internal
representations and triggers retrieval off the confidence monitor -- explicitly
because statistical output-level signals and self-reports are weaker. We want
the same trick for a different quantity: a CONTESTEDNESS direction.

Why this and not the alternatives we already burned (measured, v5/v6):
  graph features  route_signal: best graph signal w_raw corr(delta)=+0.19 with
                  the per-question help-delta; `relevance` is ANTI-correlated
                  (-0.26). Root cause: the scout SELECTS max-Wasserstein forks,
                  so W is high on consensus questions too -- no variance left.
                  Graph divergence = demographic subgroup splits, which tracks
                  human contestedness only ~+0.20.
  self-report     route (v6): asking the model in <think> scored 0.072 vs
                  baseline 0.479. Hallucination-prone (2026 adaptive-RAG
                  consensus).
  hidden states   read what the model internally represents, BEFORE it is
                  collapsed into a token -- which is exactly where `route` lost
                  the signal. That is the part neither of the above gets at.
Prize: always-baseline 0.497, always-scout 0.443, ORACLE 0.622.

Labels: do NOT train on graph divergence -- you would inherit its +0.19 ceiling
by construction. Use the self-consistency score from retrieval/contestedness.py
(the model's own stance spread over K committed samples), produced by
scripts/generate_contestedness_labels.py.

  That makes the probe learn "will I collapse to one stance on this question" --
  a property of the MODEL, not of the data. Deliberate: it is the complement of
  the graph-side signal, and their DISAGREEMENT (graph says the population is
  divided, the model does not act divided) is the routing signal we want.

Holdout is by TOPIC, not by question: adjacent OvertonBench questions are near
paraphrases, and a probe that memorizes a topic cluster passes random k-fold and
fails in deployment. `grouped_cv_auc` is the only number that means anything;
with ~60 questions the train AUC is ~1.0 regardless.

Cost model: inference is ONE forward pass, no generation. The sampling cost
(K generations x 60 questions) is paid once, at label time.

Needs local HF weights (hidden states are not exposed by the OpenAI-compatible
endpoint). The probe itself is a plain torch logistic regression -- no sklearn.

    python -m alignment.probe --selftest
    python -m alignment.probe --labels contestedness_labels.json \
        --model /path/to/Qwen2.5-7B-Instruct --layer -1 \
        --scores overton_scores_v5.csv --out contestedness_probe.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ---------------------------------------------------------------------------
# Probe (pure torch; offline-testable)
# ---------------------------------------------------------------------------
@dataclass
class ProbeConfig:
    lr: float = 1e-2
    epochs: int = 400
    l2: float = 1e-3          # weight decay -- probes overfit fast on small n
    standardize: bool = True


class LinearProbe(torch.nn.Module):
    """Logistic regression on mean-pooled hidden states = one direction in
    activation space, exactly CtrlA's construction (a linear readout), fit here
    by gradient descent rather than contrastive-pair differencing."""

    def __init__(self, d: int, cfg: ProbeConfig | None = None):
        super().__init__()
        # FIX: was `cfg: ProbeConfig = ProbeConfig()` -- one dataclass instance
        # shared by every probe constructed in the process.
        self.cfg = cfg or ProbeConfig()
        self.linear = torch.nn.Linear(d, 1)
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def _norm(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu) / self.sd if self.cfg.standardize else X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self._norm(X.float())).squeeze(-1)

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
            # Fit on the TRAIN split only; grouped_cv_auc constructs a fresh
            # probe per fold precisely so mu/sd cannot leak the held-out topic.
            self.mu = X.mean(0)
            self.sd = X.std(0).clamp_min(1e-6)
        # FIX: weight decay on the weight only. Penalising the bias shrinks the
        # intercept toward 0, i.e. toward a 50/50 base rate -- which biases the
        # probe on the class-imbalanced splits a median threshold can produce.
        opt = torch.optim.Adam(
            [{"params": [self.linear.weight], "weight_decay": self.cfg.l2},
             {"params": [self.linear.bias], "weight_decay": 0.0}],
            lr=self.cfg.lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            lossf(self(X), y).backward()
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
    uniq, counts = torch.unique(s, return_counts=True)
    for v in uniq[counts > 1]:                      # average ranks within ties
        m = s == v
        ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fold_indices(n: int, groups: Sequence | None, folds: int,
                  seed: int) -> list[torch.Tensor]:
    """Test-index list. With `groups` this is leave-one-GROUP-out (one fold per
    topic); without it, random k-fold -- kept only for the noise controls."""
    if groups is None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)
        return [idx[f::folds] for f in range(folds)]
    return [torch.tensor([i for i, gg in enumerate(groups) if gg == u],
                         dtype=torch.long) for u in sorted(set(groups))]


def grouped_cv_auc(X: torch.Tensor, y: torch.Tensor,
                   groups: Sequence | None = None, cfg: ProbeConfig | None = None,
                   folds: int = 5, seed: int = 0) -> dict:
    """Leave-one-topic-out CV AUC + out-of-fold scores.

    With a few dozen questions a probe WILL fit the training set perfectly --
    only the held-out number means anything, and only if the held-out unit is a
    TOPIC (adjacent OvertonBench questions are near-paraphrases).

    Returns {cv_auc, oof_auc, train_auc, folds, oof}:
      cv_auc    mean of per-fold AUCs, skipping single-class folds. Topics are
                small, so this silently drops folds and is noisy.
      oof_auc   AUC over the POOLED out-of-fold scores -- the number to quote.
                It pools probes with different intercepts, but AUC is
                rank-based and this is the only estimate using every question.
      oof       (n,) held-out probability per question, nan where unscored.
                Feed THIS to evaluate_gate; in-sample scores are meaningless.
    """
    cfg = cfg or ProbeConfig()
    n = len(X)
    oof = torch.full((n,), float("nan"))
    aucs, tr_aucs = [], []
    for te in _fold_indices(n, groups, folds, seed):
        if len(te) < 1:
            continue
        mask = torch.ones(n, dtype=torch.bool)
        mask[te] = False
        tr = mask.nonzero().squeeze(1)
        # FIX: the original required a 2-class TEST fold before fitting, so a
        # single-class topic produced no out-of-fold score at all and silently
        # vanished from the gate evaluation. Fit whenever TRAIN has both
        # classes; only the per-fold AUC needs a 2-class TEST fold.
        if len(tr) < 4 or len(torch.unique(y[tr])) < 2:
            continue
        p = LinearProbe(X.shape[1], cfg).fit(X[tr], y[tr])
        oof[te] = p.predict_proba(X[te])
        tr_aucs.append(auc(p.predict_proba(X[tr]), y[tr]))
        if len(torch.unique(y[te])) >= 2:
            aucs.append(auc(oof[te], y[te]))
    aucs = [a for a in aucs if a == a]
    ok = ~torch.isnan(oof)
    oof_auc = auc(oof[ok], y[ok]) if int(ok.sum()) > 1 else float("nan")
    return {"cv_auc": sum(aucs) / len(aucs) if aucs else float("nan"),
            "oof_auc": oof_auc,
            "train_auc": sum(tr_aucs) / len(tr_aucs) if tr_aucs else float("nan"),
            "folds": len(aucs), "oof": oof}


def cross_val_auc(X: torch.Tensor, y: torch.Tensor, folds: int = 5,
                  cfg: ProbeConfig | None = None, seed: int = 0) -> dict:
    """Random k-fold. Kept for the noise controls ONLY -- real reporting uses
    grouped_cv_auc(groups=topics); random folds leak topic-mates."""
    return grouped_cv_auc(X, y, None, cfg, folds, seed)


# ---------------------------------------------------------------------------
# Hidden-state extraction (needs local HF weights)
# ---------------------------------------------------------------------------
# The probe must read the state the model is in when the label was MEASURED:
# the label is stance spread under the commit-forcing prompt, so the features
# are the states under that same prompt. Raw-question states are a different
# quantity -- --prompt raw exists to ablate that, not as the default.
_FALLBACK_INSTRUCTION = (
    "Answer the question directly and take a clear position. State the single "
    "view you think is most defensible in two or three sentences. Do not "
    "enumerate multiple positions, do not hedge, and do not describe the debate "
    "-- commit to one answer.")


def commit_instruction() -> str:
    """The exact prompt the labels were sampled under (single source of truth)."""
    try:
        from retrieval.contestedness import PROBE_INSTRUCTION
        return PROBE_INSTRUCTION
    except Exception:
        return _FALLBACK_INSTRUCTION


class HiddenExtractor:
    """Mean-pooled hidden states at one layer -- the probe's input features."""

    def __init__(self, model_name: str, layer: int = -1, device: str | None = None,
                 dtype: str = "bfloat16", prompt: str = "chat"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.layer, self.prompt = layer, prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # FIX: models with no pad token (Llama family) crashed on padding=True.
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)).to(self.device).eval()

    def build_prompts(self, questions: Sequence[str]) -> list[str]:
        if self.prompt != "chat" or not hasattr(self.tok, "apply_chat_template"):
            return list(questions)
        sysmsg = commit_instruction()
        return [self.tok.apply_chat_template(
            [{"role": "system", "content": sysmsg},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True) for q in questions]

    @torch.no_grad()
    def __call__(self, texts: Sequence[str], batch_size: int = 8,
                 max_length: int = 512) -> torch.Tensor:
        texts = self.build_prompts(texts)
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(self.device)
            # FIX: request hidden states per call. Passing output_hidden_states
            # to from_pretrained() only sets it on the config and is dropped by
            # some model classes -- .hidden_states would then be None.
            out = self.model(**enc, output_hidden_states=True)
            # FIX: pool in fp32. bf16 has 8 mantissa bits; summing 512 token
            # vectors in bf16 throws away ~2 decimal digits before the probe
            # ever sees the feature.
            hs = out.hidden_states[self.layer].float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            outs.append(pooled.cpu())
        return torch.cat(outs)


# ---------------------------------------------------------------------------
# Persistence + the hook route_signal.py can call
# ---------------------------------------------------------------------------
def save_probe(probe: LinearProbe, path: str, **meta) -> None:
    torch.save({"state_dict": probe.state_dict(), "d": probe.linear.in_features,
                "cfg": asdict(probe.cfg), **meta}, path)


def load_probe(path: str) -> tuple[LinearProbe, dict]:
    """FIX: the old save wrote no cfg, so a reloaded probe silently reverted to
    default standardize -- and mu/sd are in state_dict, so it would normalize
    with fitted statistics under a config that might not use them."""
    ck = torch.load(path, map_location="cpu")
    probe = LinearProbe(ck["d"], ProbeConfig(**ck.get("cfg", {})))
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    return probe, ck


def probe_route_signal(questions, probe_path: str, *, model: str | None = None,
                       batch_size: int = 8, max_length: int = 512,
                       device: str | None = None) -> dict[int, float]:
    """qid -> p(contested). The hook that lets route_signal.py score the probe
    as just another routing signal, through its own _best_gate.

    `questions` is what evaluation.overton.eval_overtonbench.load_questions()
    returns: [(qid, question), ...]; a bare list of strings is enumerated.
    Layer, prompt style and model default to what was recorded at training
    time, so the features match the ones the probe was fit on.

    Wiring in evaluation/overton/route_signal.py (that file is NOT edited here):

        from alignment.probe import probe_route_signal
        _p = probe_route_signal(load_questions(), args.probe)   # after `rows`
        for r in rows: r["probe"] = _p[r["qid"]]
        # then add "probe" to the sig_name tuple in the reporting loop
    """
    qs = list(questions)
    pairs = list(enumerate(qs)) if qs and isinstance(qs[0], str) else \
        [(int(a), b) for a, b in qs]
    probe, meta = load_probe(probe_path)
    ex = HiddenExtractor(model or meta["model"], meta.get("layer", -1),
                         device=device, dtype=meta.get("dtype", "bfloat16"),
                         prompt=meta.get("prompt", "chat"))
    X = ex([q for _, q in pairs], batch_size=batch_size, max_length=max_length)
    p = probe.predict_proba(X)
    return {qid: float(v) for (qid, _), v in zip(pairs, p)}


# ---------------------------------------------------------------------------
# Post-hoc gate evaluation -- scored EXACTLY like the graph signals
# ---------------------------------------------------------------------------
def _route_signal_scorers():
    """Reuse route_signal.py's own scorers so the probe is scored identically,
    including its best-single-threshold gate. The fallback keeps --selftest and
    offline runs working if that module moves; it is the same arithmetic."""
    try:
        from evaluation.overton.route_signal import _best_gate, _corr
        return _best_gate, _corr, True
    except Exception:
        import statistics as st

        def _corr(xs, ys):
            if len(xs) < 2:
                return float("nan")
            mx, my = st.mean(xs), st.mean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = (sum((a - mx) ** 2 for a in xs) *
                   sum((b - my) ** 2 for b in ys)) ** 0.5
            return num / den if den else float("nan")

        def _best_gate(sig, delta, base_cov, scout_cov):
            n = len(sig)
            base_mean = sum(base_cov) / n
            best_score, best_thr = base_mean, None
            for t in [-1e9] + sorted(set(sig)):
                s = sum(scout_cov[i] if sig[i] > t else base_cov[i]
                        for i in range(n)) / n
                if s > best_score:
                    best_score, best_thr = s, t
            return best_score, best_thr, base_mean
        return _best_gate, _corr, False


def evaluate_gate(qids: Sequence[int], sig: Sequence[float], scores_csv: str,
                  inject_cond: str = "scout", base_cond: str = "baseline",
                  name: str = "probe_oof", n_perm: int = 2000,
                  seed: int = 0) -> dict:
    """Does the probe predict the per-question help-delta better than the graph?

    Post-hoc against a finished run -- no generation, same design that killed
    the graph signals. Reference to beat, route_signal on these 60 questions:
    w_raw corr +0.19, relevance -0.26, no graph gate clearing always-baseline.
    Pass OUT-OF-FOLD probe scores; in-sample scores make this meaningless.

    Also runs a PERMUTATION NULL, because n=60 is small and this evaluation's
    noise floor is high: on overton_scores_v5, 15% of purely random signals
    reach |corr| >= 0.19 and 5% reach best_gate >= 0.532. Reading +0.19 as "a
    weak signal" rather than "nothing" is the mistake this null prevents; a
    probe result has to clear the null, not clear the graph.
    """
    import csv
    import random
    import statistics as st
    from collections import defaultdict

    _best_gate, _corr, native = _route_signal_scorers()
    cov: dict[int, dict] = defaultdict(dict)
    for r in csv.DictReader(open(scores_csv, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])

    rows = [(q, s) for q, s in zip(qids, sig)
            if s == s and base_cond in cov.get(q, {})
            and inject_cond in cov.get(q, {})]
    if len(rows) < 3:
        print(f"\nno usable overlap with {scores_csv} -- skipping gate evaluation")
        return {}
    n = len(rows)
    s_ = [s for _, s in rows]
    base = [cov[q][base_cond] for q, _ in rows]
    inj = [cov[q][inject_cond] for q, _ in rows]
    delta = [i - b for b, i in zip(base, inj)]
    best, thr, base_mean = _best_gate(s_, delta, base, inj)
    helped = [s for s, d in zip(s_, delta) if d > 0.01]
    hurt = [s for s, d in zip(s_, delta) if d < -0.01]
    res = {"n": n, "corr": _corr(s_, delta), "best_gate": best, "thr": thr,
           "always_base": base_mean, "always_inject": sum(inj) / n,
           "oracle": sum(max(b, i) for b, i in zip(base, inj)) / n,
           "helped_mean": st.mean(helped) if helped else float("nan"),
           "hurt_mean": st.mean(hurt) if hurt else float("nan")}

    # Permutation null: same signal values, question order destroyed.
    if n_perm > 0:
        rng = random.Random(seed)
        perm = list(s_)
        c_hits = g_hits = 0
        for _ in range(n_perm):
            rng.shuffle(perm)
            if abs(_corr(perm, delta)) >= abs(res["corr"]):
                c_hits += 1
            if _best_gate(perm, delta, base, inj)[0] >= res["best_gate"]:
                g_hits += 1
        res["p_corr"] = (c_hits + 1) / (n_perm + 1)
        res["p_gate"] = (g_hits + 1) / (n_perm + 1)

    print(f"\n=== gate evaluation vs {scores_csv} (n={n}, "
          f"scorers={'route_signal' if native else 'local fallback'}) ===")
    print(f"  always-{base_cond:<9} {res['always_base']:.4f}")
    print(f"  always-{inject_cond:<9} {res['always_inject']:.4f}")
    print(f"  oracle             {res['oracle']:.4f}")
    print(f"\n{'signal':<12}{'corr(delta)':>13}{'best_gate':>11}{'thr':>10}"
          f"{'helped_mean':>13}{'hurt_mean':>11}")
    print(f"{name:<12}{res['corr']:>+13.3f}{res['best_gate']:>11.4f}"
          f"{(f'{thr:.3f}' if thr is not None else 'none'):>10}"
          f"{res['helped_mean']:>13.3f}{res['hurt_mean']:>11.3f}")
    print(f"{'w_raw (v5)':<12}{'+0.190':>13}{'--':>11}{'--':>10}{'--':>13}"
          f"{'--':>11}   <- graph ceiling to beat")
    if "p_corr" in res:
        print(f"\npermutation null ({n_perm} shuffles): p(corr) = "
              f"{res['p_corr']:.3f}, p(best_gate) = {res['p_gate']:.3f}")
        print("  n=60 is small: a random signal clears |corr|>=0.19 15% of the "
              "time here.\n  If p is not small, the probe has NOT beaten the "
              "graph -- neither of them did anything.")
    print("\n(best_gate is a single-split overfit ceiling; the probe is only "
          "useful if helped_mean and hurt_mean separate -- the graph signals "
          "never did.)")
    return res


# ---------------------------------------------------------------------------
def _synth(n: int = 240, d: int = 64, sep: float = 2.0, n_topics: int = 6,
           topic_scale: float = 1.5, seed: int = 0):
    """Synthetic hidden states with ONE planted contestedness direction u.

    Topic offsets live in the subspace ORTHOGONAL to u and are large relative
    to sep: a probe that latches onto topic instead of contestedness scores
    well under random k-fold and collapses under leave-one-topic-out. That is
    the failure mode topic-holdout exists to catch."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    u = torch.zeros(d)
    u[:8] = torch.randn(8, generator=g)
    u = u / u.norm()
    y = (torch.rand(n, generator=g) < 0.5).float()
    groups = [i % n_topics for i in range(n)]
    T = torch.randn(n_topics, d, generator=g)
    T = T - (T @ u).unsqueeze(1) * u.unsqueeze(0)        # orthogonal nuisance
    X = X + torch.stack([T[t] for t in groups]) * topic_scale
    X = X + (y - 0.5).unsqueeze(1) * u.unsqueeze(0) * (2 * sep)
    return X, y, groups, u


def _selftest() -> None:
    """Planted direction -> the probe recovers it under TOPIC holdout.
    Shuffled labels -> it does not (and train >> held-out, the overfit
    signature). Without the shuffled control, a probe reading noise looks
    exactly like a result."""
    torch.manual_seed(0)
    X, y, groups, u = _synth()
    cfg = ProbeConfig()

    good = grouped_cv_auc(X, y, groups, cfg)
    p = LinearProbe(X.shape[1], cfg).fit(X, y)
    cos_good = float(torch.dot(p.direction, u))
    assert good["oof_auc"] > 0.85, good
    assert cos_good > 0.60, cos_good
    assert abs(float(p.direction.norm()) - 1.0) < 1e-5

    # CONTROL: identical features, labels shuffled. Anything the probe "finds"
    # here is noise it memorized.
    g = torch.Generator().manual_seed(123)
    y_sh = y[torch.randperm(len(y), generator=g)]
    ctrl = grouped_cv_auc(X, y_sh, groups, cfg)
    p_sh = LinearProbe(X.shape[1], cfg).fit(X, y_sh)
    cos_ctrl = abs(float(torch.dot(p_sh.direction, u)))
    assert 0.30 < ctrl["oof_auc"] < 0.70, ctrl
    assert cos_ctrl < 0.30, cos_ctrl
    assert ctrl["train_auc"] - ctrl["oof_auc"] > 0.15, ctrl   # memorization

    # CONTROL: a label that IS the topic. Random k-fold calls it a win; the
    # topic holdout refuses to. This is why holdout is by topic.
    y_topic = torch.tensor([float(t < 3) for t in groups])
    rand_cv = cross_val_auc(X, y_topic, folds=5, cfg=cfg)
    grp_cv = grouped_cv_auc(X, y_topic, groups, cfg)
    assert rand_cv["oof_auc"] - grp_cv["oof_auc"] > 0.15, (rand_cv, grp_cv)

    assert abs(auc(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 1.0])) - 1.0) < 1e-6
    assert abs(auc(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])) - 0.5) < 1e-6

    print("probe self-test OK")
    print(f"  planted   oof_auc={good['oof_auc']:.3f} cv_auc={good['cv_auc']:.3f} "
          f"train={good['train_auc']:.3f}  cos(w,u)=+{cos_good:.3f}")
    print(f"  shuffled  oof_auc={ctrl['oof_auc']:.3f} cv_auc={ctrl['cv_auc']:.3f} "
          f"train={ctrl['train_auc']:.3f}  |cos(w,u)|={cos_ctrl:.3f}  "
          f"<- train-vs-oof gap {ctrl['train_auc'] - ctrl['oof_auc']:+.3f} is the point")
    print(f"  confound  label=topic: random-kfold {rand_cv['oof_auc']:.3f} vs "
          f"leave-one-topic-out {grp_cv['oof_auc']:.3f}  <- why holdout is by topic")


def main():
    ap = argparse.ArgumentParser(description="Contestedness probe on hidden states")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", default=None,
                    help="JSON from scripts/generate_contestedness_labels.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="LOCAL directory preferred (hub ids need a populated "
                         "HF_HOME; hub resolution is broken on the cluster)")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--prompt", choices=["chat", "raw"], default="chat",
                    help="chat = the same commit-forcing prompt the labels were "
                         "measured under; raw = ablation")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=None,
                    help="binarize the weak label at this score (default: median)")
    ap.add_argument("--group_by", choices=["topic", "none"], default="topic")
    ap.add_argument("--folds", type=int, default=5, help="only if --group_by none")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--scores", default=None,
                    help="overton_scores_vN.csv -- score the probe post-hoc "
                         "against baseline/inject/oracle (no generation)")
    ap.add_argument("--inject_cond", default="scout")
    ap.add_argument("--features_out", default=None, help="cache X to .pt")
    ap.add_argument("--out", default="contestedness_probe.pt")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.labels:
        ap.error("pass --labels (or --selftest)")

    rows = json.load(open(args.labels, encoding="utf-8"))
    questions = [r["question"] for r in rows]
    qids = [int(r.get("question_id", i)) for i, r in enumerate(rows)]
    scores = torch.tensor([float(r["score"]) for r in rows]).float()
    groups = [r.get("topic", 0) for r in rows] if args.group_by == "topic" else None
    thr = args.threshold if args.threshold is not None else float(scores.median())
    y = (scores > thr).float()
    if len(torch.unique(y)) < 2:
        # A degenerate score column median-splits into one class and every AUC
        # below is nan. Catch it here, not 20 GPU-minutes later.
        raise SystemExit(f"labels are degenerate at threshold {thr:.3f}: "
                         f"score range [{float(scores.min()):.3f}, "
                         f"{float(scores.max()):.3f}] -- K too small, or the "
                         "commit prompt was not applied at sampling time")
    print(f"{len(rows)} questions, label threshold {thr:.3f} -> "
          f"{int(y.sum())} contested / {int((1 - y).sum())} consensus")
    if groups is not None:
        print(f"holdout: leave-one-topic-out over {len(set(groups))} topics "
              "(random k-fold leaks near-paraphrase topic-mates)")

    ex = HiddenExtractor(args.model, args.layer, dtype=args.dtype,
                         prompt=args.prompt)
    X = ex(questions, batch_size=args.batch_size, max_length=args.max_length)
    print(f"hidden states: {tuple(X.shape)} from layer {args.layer} of "
          f"{args.model} (prompt={args.prompt})")
    if args.features_out:
        torch.save({"X": X, "qids": qids}, args.features_out)

    cfg = ProbeConfig(lr=args.lr, epochs=args.epochs, l2=args.l2)
    stats = grouped_cv_auc(X, y, groups, cfg, folds=args.folds)
    print(f"\nheld-out AUC = {stats['oof_auc']:.3f} (pooled out-of-fold)   "
          f"per-fold mean {stats['cv_auc']:.3f} over {stats['folds']} scorable folds")
    print(f"train AUC    = {stats['train_auc']:.3f}")
    print("  0.5 = the direction does not exist / is not linearly readable here;\n"
          "  a large train-vs-held-out gap means the probe memorized "
          f"(gap {stats['train_auc'] - stats['oof_auc']:+.3f}) -- get more questions.")

    if args.scores:
        evaluate_gate(qids, stats["oof"].tolist(), args.scores,
                      inject_cond=args.inject_cond)

    probe = LinearProbe(X.shape[1], cfg).fit(X, y)
    save_probe(probe, args.out, layer=args.layer, model=args.model,
               prompt=args.prompt, threshold=thr, dtype=args.dtype,
               oof_auc=stats["oof_auc"], train_auc=stats["train_auc"],
               n=len(rows), n_topics=len(set(groups)) if groups else 0)
    print(f"saved probe -> {args.out}")


if __name__ == "__main__":
    main()
