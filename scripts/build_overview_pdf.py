"""Generate PluralTree_Overview.pdf — a concrete, specific explanation of the
codebase, architecture, and the purpose of the experiments.

Offline-friendly: uses only matplotlib (PdfPages) for rendering, so it needs no
LaTeX/pandoc/reportlab. Run:  python scripts/build_overview_pdf.py
"""

from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Page geometry (US Letter), all in inches
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = 8.5, 11.0
L_MARGIN, R_MARGIN, T_MARGIN, B_MARGIN = 0.9, 0.9, 0.95, 0.9
USABLE_W = PAGE_W - L_MARGIN - R_MARGIN

INK = "#1a1a1a"
ACCENT = "#1f4e79"
MUTED = "#555555"

# style: (size_pt, weight, family, color, space_before_in, wrap_chars, indent_in)
STYLES = {
    "h1":     (17, "bold",   "DejaVu Sans", ACCENT, 0.28, 48, 0.0),
    "h2":     (13, "bold",   "DejaVu Sans", ACCENT, 0.22, 58, 0.0),
    "h3":     (11, "bold",   "DejaVu Sans", INK,    0.16, 70, 0.0),
    "body":   (9.7, "normal","DejaVu Sans", INK,    0.06, 104, 0.0),
    "bullet": (9.7, "normal","DejaVu Sans", INK,    0.03, 98, 0.18),
    "code":   (8.2, "normal","DejaVu Sans Mono", "#202020", 0.06, 97, 0.12),
    "caption":(8.5, "italic","DejaVu Sans", MUTED,  0.04, 110, 0.0),
}


class PDFBuilder:
    def __init__(self, path: str):
        self.pdf = PdfPages(path)
        self.fig = None
        self.y = 0.0

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = PAGE_H - T_MARGIN

    def _draw(self, text, x_in, size, weight, family, color, style="normal"):
        self.fig.text(
            x_in / PAGE_W, self.y / PAGE_H, text,
            fontsize=size, fontweight=weight, fontfamily=family,
            color=color, va="top", ha="left", style=style,
        )

    def block(self, style_name, text):
        size, weight, family, color, before, wrapw, indent = STYLES[style_name]
        line_h = size * 1.42 / 72.0
        fontstyle = "italic" if style_name == "caption" or weight == "italic" else "normal"
        weight_use = "normal" if weight == "italic" else weight

        if self.fig is None:
            self._new_page()

        self.y -= before
        if self.y - line_h < B_MARGIN:
            self._new_page()

        if style_name == "code":
            lines = text.split("\n")
        elif style_name == "bullet":
            lines = textwrap.wrap(text, width=wrapw) or [""]
        else:
            lines = []
            for para_line in text.split("\n"):
                lines.extend(textwrap.wrap(para_line, width=wrapw) or [""])

        for i, ln in enumerate(lines):
            if self.y - line_h < B_MARGIN:
                self._new_page()
            x = L_MARGIN + indent
            if style_name == "bullet" and i == 0:
                self.fig.text((L_MARGIN + 0.02) / PAGE_W, self.y / PAGE_H, "•",
                              fontsize=size, color=ACCENT, va="top", ha="left")
            self._draw(ln, x, size, weight_use, family, color, fontstyle)
            self.y -= line_h

    def hrule(self):
        if self.fig is None:
            self._new_page()
        self.y -= 0.04
        if self.y - 0.05 < B_MARGIN:
            self._new_page()
        self.fig.add_artist(plt.Line2D(
            [L_MARGIN / PAGE_W, (PAGE_W - R_MARGIN) / PAGE_W],
            [self.y / PAGE_H, self.y / PAGE_H],
            color="#cccccc", linewidth=0.8, transform=self.fig.transFigure,
        ))
        self.y -= 0.10

    def title_page(self, title, subtitle, meta_lines):
        self._new_page()
        self.y = PAGE_H * 0.62
        self.fig.text(0.5, self.y / PAGE_H, title, fontsize=30, fontweight="bold",
                      color=ACCENT, va="center", ha="center", fontfamily="DejaVu Sans")
        self.y -= 0.55
        for sline in textwrap.wrap(subtitle, width=58):
            self.fig.text(0.5, self.y / PAGE_H, sline, fontsize=13, color=INK,
                          va="center", ha="center", fontfamily="DejaVu Sans")
            self.y -= 0.30
        self.y -= 0.25
        for m in meta_lines:
            self.fig.text(0.5, self.y / PAGE_H, m, fontsize=10, color=MUTED,
                          va="center", ha="center", fontfamily="DejaVu Sans")
            self.y -= 0.24
        self.fig.add_artist(plt.Line2D([0.35, 0.65], [0.40, 0.40], color=ACCENT,
                                       linewidth=1.4, transform=self.fig.transFigure))

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.pdf.close()


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------
def build(doc: PDFBuilder):
    doc.title_page(
        "PluralTree",
        "Hyperbolic Tree-GRU with Gated Knowledge Injection: "
        "Codebase, Architecture, and Experiment Guide",
        ["A concrete walkthrough of the model, math, and experiments",
         "Generated 2026-05-31"],
    )

    # 1 -------------------------------------------------------------------
    doc.block("h1", "1.  Overview")
    doc.hrule()
    doc.block("body",
        "PluralTree learns hierarchical knowledge-graph embeddings in the Poincare "
        "ball. A Hyperbolic Tree-GRU encodes a rooted tree bottom-up on the manifold; "
        "Gated Knowledge Injection (GKI) blends external knowledge into each node. The "
        "model is trained by link prediction on CulturalBench and evaluated with "
        "filtered MRR / Hits@k. This document is deliberately concrete: it gives the "
        "actual operations, tensor shapes, and hyperparameters used in the code.")
    doc.block("h3", "Concrete dimensions for the default CulturalBench run")
    doc.block("code",
        "N (entities)        = 1284     (1 world + 11 regions + 43 countries + ~1229 practices)\n"
        "num_relations       = 3        (practiced_in, located_in, part_of)\n"
        "d_text (input)      = 768      (all-mpnet-base-v2; 384 for all-MiniLM-L6-v2)\n"
        "d_hidden            = 128      (hidden size = relation-embedding size)\n"
        "curvature c         = 1.0      (ball radius 1/sqrt(c) = 1)\n"
        "train / val / test  = 1037 / 178 / 180 triples\n"
        "n_negative          = 10       margin = 1.0   batch_size = 128")

    # 2 -------------------------------------------------------------------
    doc.block("h1", "2.  Why Hyperbolic Geometry")
    doc.hrule()
    doc.block("body",
        "A tree with branching factor b has ~b^d nodes at depth d. Euclidean volume "
        "grows only polynomially with radius, so embedding a tree there forces "
        "distortion. Hyperbolic volume grows exponentially with radius, matching the "
        "tree, so trees embed with low distortion in few dimensions. In the Poincare "
        "ball, the origin acts as the abstract root and the boundary as specific "
        "leaves, so a node's radius rho = sqrt(c) * ||h|| encodes its depth. This is "
        "the geometric fact the whole architecture exploits.")

    # 3 -------------------------------------------------------------------
    doc.block("h1", "3.  Architecture in Detail")
    doc.hrule()

    doc.block("h2", "3.1  Manifold operations (pluraltree/manifolds/poincare.py)")
    doc.block("body",
        "Every state lives on the ball; all of the following are the exact operations "
        "used. <x,y> is the dot product, ||x|| the L2 norm.")
    doc.block("code",
        "Mobius addition (translation on the ball):\n"
        "  x (+) y =  ( (1 + 2c<x,y> + c||y||^2) x  +  (1 - c||x||^2) y )\n"
        "            -----------------------------------------------------\n"
        "                   1 + 2c<x,y> + c^2 ||x||^2 ||y||^2\n"
        "\n"
        "Exp map at origin (tangent -> ball):   exp0(v) = tanh(sqrt(c)||v||)/(sqrt(c)||v||) * v\n"
        "Log map at origin (ball -> tangent):   log0(x) = artanh(sqrt(c)||x||)/(sqrt(c)||x||) * x\n"
        "\n"
        "Geodesic distance:   d(x,y) = (2/sqrt(c)) * artanh( sqrt(c) || (-x) (+) y || )\n"
        "Conformal factor:    lambda(x) = 2 / (1 - c||x||^2)")
    doc.block("body",
        "Numerical safety (math_utils.py): norms are floored (MIN_NORM), artanh is "
        "clamped below 1, and every result is projected just inside the boundary so "
        "points never reach radius 1/sqrt(c).")

    doc.block("h2", "3.2  Child aggregation (tree_gru/aggregation.py)")
    doc.block("body",
        "Children are combined with the Einstein / Mobius weighted midpoint, which is "
        "the proper 'average' of points on the ball. With K padded children "
        "h_children of shape (K, B, d) and weights w of shape (K, B, 1):")
    doc.block("code",
        "gamma =  sum_k [ w_k * lambda(x_k) * x_k ]   /   sum_k [ w_k * (lambda(x_k) - 1) ]\n"
        "result = project(gamma)                       # midpoint on the ball, shape (B, d)")
    doc.block("body",
        "Weights come from attention: score_k = W_a( log0(h_k) ) with W_a a (d -> 1) "
        "linear; padded children are masked to -inf, then softmax over k. (A uniform-"
        "weight mode exists too.) Masked/padded children get weight 0 and contribute "
        "nothing, which is what makes the level-batched encoder exact.")

    doc.block("h2", "3.3  The Hyperbolic Tree-GRU cell (tree_gru/cell.py)")
    doc.block("body",
        "Each node's input feature x is the frozen text embedding passed through "
        "input_proj (a 768 -> 128 linear); x lives in tangent space. The children "
        "summary h_agg lives on the ball. The cell maps h_agg to tangent space, runs "
        "a standard GRU there, and maps back. Shapes shown for a batch of B nodes, "
        "d_hidden = 128.")
    doc.block("code",
        "h_agg   = mobius_midpoint(children)             # (B,128) on ball\n"
        "h_t     = log0(h_agg)                           # (B,128) tangent space\n"
        "\n"
        "z       = sigmoid( W_z [h_t ; x] )              # update gate   (B,128)\n"
        "r       = sigmoid( W_r [h_t ; x] )              # reset gate    (B,128)\n"
        "n       = tanh(    W_n [r*h_t ; x] )            # candidate     (B,128)\n"
        "h_t'    = (1 - z) * h_t  +  z * n               # GRU mix       (B,128)\n"
        "\n"
        "h_v     = exp0(h_t')                            # (B,128) back on the ball\n"
        "\n"
        "W_z, W_r, W_n : Linear(256 -> 128)   ([h_t ; x] concatenates 128+128)")
    doc.block("body",
        "Leaves (practices) have no children, so h_agg is the origin (h_t = 0) and "
        "h_v = exp0( z * tanh(W_n[0 ; x]) ) depends only on the leaf's own text. The "
        "same three weight matrices are reused at every node — there is no per-node "
        "embedding table. Running the gates in tangent space avoids defining sigmoid/"
        "tanh on the curved manifold.")

    doc.block("h2", "3.4  Gated Knowledge Injection and the depth-aware gate")
    doc.block("body",
        "GKI mixes the structural state h with a knowledge vector k. k is the node's "
        "text embedding sent through a ProjectionLayer (Linear + LayerNorm + exp0), so "
        "k is a point on the ball, shape (B,128). The depth-aware gate "
        "(combined/depth_aware_gate.py) appends the radius rho:")
    doc.block("code",
        "rho = sqrt(c) * ||h||                                   # (B,1) depth signal\n"
        "g   = sigmoid( W_g [ log0(h) ; log0(k) ; rho ] + b )    # (B,128) gate, W_g: 257->128\n"
        "h'  = mobius_midpoint( h, k ; weights = [1 - g, g] )    # (B,128) blended, on ball")
    doc.block("body",
        "g near 0 keeps h (trust structure); g near 1 adopts k (trust knowledge). A "
        "second head beta = sigmoid(w_beta * rho + b_beta) blends a broad vs. precise "
        "knowledge source by depth (active once a distinct second source exists). "
        "Injection point is configurable: PRE_AGGREGATION, POST_AGGREGATION (default), "
        "POST_GRU, or DUAL — swept in the A1 ablation.")

    doc.block("h2", "3.5  Curriculum: the gate-bias schedule (combined/knowledge_schedule.py)")
    doc.block("body",
        "The bias b in the gate is pinned by a 3-phase schedule on the optimizer step:")
    doc.block("code",
        "step < warmup1            : b = -2.0            sigmoid(-2.0) ~ 0.12  (nearly closed)\n"
        "warmup1 <= step < warmup2 : b ramps -2.0 -> 0  linearly\n"
        "step >= warmup2           : b = 0.0            unbiased (neutral 0.5)")
    doc.block("body",
        "Defaults used: warmup1 = 400, warmup2 = 1600. With ~9 steps/epoch, gates "
        "become unbiased near epoch 178, so 300 epochs leaves a long stretch of "
        "training with knowledge available. Starting closed forces the model to learn "
        "structure before it can lean on knowledge.")

    doc.block("h2", "3.6  Scoring head (training/scoring.py)")
    doc.block("body",
        "TransE-style translation on the ball. Each relation is a learned point r on "
        "the ball (geoopt ManifoldParameter, shape (3,128), initialised uniform in "
        "[-0.001, 0.001]).")
    doc.block("code",
        "translated = h_s (+) r_relation          # Mobius addition, (B,128)\n"
        "score      = - d( translated, h_o )      # negative geodesic distance, (B,)")
    doc.block("body",
        "Higher score = more plausible triple. After translating by the relation, "
        "prediction is exactly nearest-neighbour search in hyperbolic distance.")

    doc.block("h2", "3.7  Negatives and the training loss")
    doc.block("body",
        "Negatives corrupt the OBJECT with a same-type entity (data/negative_sampler.py). "
        "Type constraints: practiced_in -> the 43 countries, located_in -> the 11 "
        "regions, part_of -> {World}. n_negative = 10 per positive; 'filtered' sampling "
        "skips known true triples. The loss is a margin ranking term over all B*K "
        "negatives plus a small gate-sparsity penalty:")
    doc.block("code",
        "L_lp     = mean( relu( margin + score_neg - score_pos ) )     # margin = 1.0\n"
        "L_sparse = mean( | W_g.weight | )                             # proxy for gate usage\n"
        "L        = L_lp + 0.01 * L_sparse")
    doc.block("caption",
        "Note: the sparsity term penalises gate-weight magnitude as a proxy; a true "
        "L0/hard-concrete gate (see docs/GATING.md) would penalise activations directly.")

    doc.block("h2", "3.8  Evaluation (evaluation/link_prediction.py)")
    doc.block("body",
        "For each test triple (s, r, o): score s against every type-valid candidate "
        "object (e.g. all 43 countries for practiced_in), remove other known positives "
        "(filtered setting), and rank the true o. Aggregate over triples:")
    doc.block("code",
        "rank   = 1 + #{ candidate c : score(s,r,c) > score(s,r,o), c not a known positive }\n"
        "MRR    = mean( 1 / rank )\n"
        "Hits@k = fraction of triples with rank <= k        for k in {1, 3, 10}")

    doc.block("h2", "3.9  Optimizer (utils/riemannian_optim.py)")
    doc.block("body",
        "Parameters are split by type. Relation embeddings (ManifoldParameter) use "
        "geoopt.RiemannianAdam at lr_manifold = 1e-2 for curvature-aware updates that "
        "stay on the ball; all Euclidean weights (input_proj, aggregator, GRU, gate) "
        "use ordinary Adam at lr = 1e-3. Gradients are norm-clipped. The encoder is "
        "re-run every batch (h is recomputed, never cached), so gradients are exact "
        "for the current weights.")

    doc.block("h2", "3.10  The level-batched encoder (combined/gki_tree_encoder.py)")
    doc.block("body",
        "The reference path walks nodes one at a time: 1284 tiny cell calls per "
        "encode, x ~2700 encodes per run. Cost is dominated by kernel-launch overhead, "
        "not arithmetic. The fast path groups nodes by height (leaf = 0, "
        "height(v) = 1 + max child height). Every child has strictly smaller height, "
        "so a whole level is mutually independent and runs as ONE padded, masked, "
        "batched call:")
    doc.block("code",
        "height 0:  ~1229 practices   ->  1 batched leaf step\n"
        "height 1:    43 countries    ->  1 batched cell call (children padded to max_k)\n"
        "height 2:    11 regions      ->  1 batched cell call\n"
        "height 3:     1 world        ->  1 batched cell call\n"
        "\n"
        "per encode:  1284 cell calls  ->  4 cell calls")
    doc.block("body",
        "The padding plan is built once and cached (the tree is static). Verified "
        "identical to the sequential path: float64 difference ~1e-13 (machine "
        "epsilon). Measured speedup: 55x on CPU, larger on GPU.")

    # 4 -------------------------------------------------------------------
    doc.block("h1", "4.  The Dataset: CulturalBench")
    doc.hrule()
    doc.block("body",
        "kellycyy/CulturalBench (Easy split): multiple-choice cultural questions. "
        "PluralTree keeps only the geography and the question text. The graph is a "
        "single rooted tree:")
    doc.block("code",
        "World (1)\n"
        "  +- Region (11):   East_Asia, Western_Europe, Africa, ...\n"
        "       +- Country (43):  China, Japan, France, Brazil, ...\n"
        "            +- Practice (~1229):  one node per unique question")
    doc.block("body",
        "Relations: practiced_in (practice->country), located_in (country->region), "
        "part_of (region->world). Structural triples are always visible; practice "
        "triples are split 80/10/10. Each entity's description is embedded with a "
        "frozen sentence-transformer used as BOTH the node feature x and the knowledge "
        "vector k — a current limitation: because they share a source, the gate has no "
        "genuinely new information to inject yet (roadmap B2 adds a distinct source).")
    doc.block("caption",
        "Leakage guard: the graph is built from the questions, so any LLM evaluation "
        "must exclude a test question's own node and all val/test practices.")

    # 5 -------------------------------------------------------------------
    doc.block("h1", "5.  Project Layout (key modules)")
    doc.hrule()
    for path, desc in [
        ("pluraltree/manifolds/poincare.py", "Mobius ops, exp/log maps, distance, midpoint"),
        ("pluraltree/tree_gru/", "HyperbolicTreeGRUCell + ChildAggregator"),
        ("pluraltree/gki/", "EuclideanGate / HyperbolicGate + GKIInjector"),
        ("pluraltree/combined/", "depth-aware gate, GKI cell, encoder, schedule"),
        ("data/culturalbench.py", "build CulturalGraph + frozen text embeddings"),
        ("data/negative_sampler.py", "type-constrained corruption + filtered ranking"),
        ("training/", "trainer, scoring head, losses"),
        ("evaluation/link_prediction.py", "filtered MRR / Hits@{1,3,10}"),
        ("scripts/train.py", "CLI entry point (banner + RESULT summary line)"),
        ("jobs/", "SLURM scripts incl. the A1 ablation set"),
    ]:
        doc.block("bullet", f"{path}  -  {desc}")

    # 6 -------------------------------------------------------------------
    doc.block("h1", "6.  Experiments and Their Purpose")
    doc.hrule()
    doc.block("body",
        "Principle: validate each component before building on it. An early run showed "
        "opening the gates REDUCED MRR, so GKI is not yet proven to help — making the "
        "ablations urgent.")
    doc.block("h2", "6.1  A1 ablations (currently running)")
    doc.block("bullet", "--no_gki: pure Tree-GRU. Baseline vs. this is the central test — does injection help at all?")
    doc.block("bullet", "--gate_type plain|depth_aware: does conditioning the gate on radius rho matter?")
    doc.block("bullet", "--injection pre_agg|post_agg|post_gru|dual: where should knowledge enter?")
    doc.block("bullet", "Euclidean vs. hyperbolic (planned): does the ball beat flat space here? (needs a Euclidean manifold.)")
    doc.block("caption", "Each run prints one grep-able line: RESULT | config | best_val_mrr | test_mrr h@1 h@3 h@10.")
    doc.block("h2", "6.2  Roadmap A-E")
    doc.block("bullet", "A Validation: ablations, rho-vs-depth, baselines (TransE/RotatE, frozen-encoder NN).")
    doc.block("bullet", "B Architecture: stronger encoders, a distinct 2nd knowledge source, learnable c, top-down pass.")
    doc.block("bullet", "C Scale/generalisation: larger/deeper KGs, inductive eval on held-out subtrees, DAG structure.")
    doc.block("bullet", "D LLM integration: hierarchical retrieval, traversal-as-tool, soft-prompt injection, uncertainty prompting.")
    doc.block("bullet", "E Plurality/distributions: Wrapped-Normal embeddings, Pluralistic Leaf Existence, and their synthesis.")

    # 7 -------------------------------------------------------------------
    doc.block("h1", "7.  End Goal: KG-Augmented LLM Reasoning")
    doc.hrule()
    doc.block("body",
        "Target: integrate the KG with a fine-tunable open 7B (LoRA) to aid "
        "CulturalBench QA, climbing a coupling ladder so each step justifies the next:")
    doc.block("bullet", "1. Hierarchical retrieval (walk World->...->Practice) injected as text, vs. flat RAG.")
    doc.block("bullet", "2. Embedding-guided traversal exposed as an LM tool (hyperbolic NN + tree paths).")
    doc.block("bullet", "3. Soft-prompt injection: project h_v into the LM token space, train with LoRA.")
    doc.block("bullet", "4. Cross-attention fusion (GreaseLM-style) once the KG signal is proven.")
    doc.block("body",
        "Honest metric: KG-grounded LM vs. LoRA-only LM with no KG. Guardrail: never "
        "retrieve the target question's own node or any val/test practice.")

    # 8 -------------------------------------------------------------------
    doc.block("h1", "8.  Research Frontiers")
    doc.hrule()
    doc.block("body",
        "Single-parent today because the loader hardcodes one parent and the bottom-up "
        "encoder gives one embedding per node. A plain DAG (multiple parents, one "
        "embedding) is mostly a data change — the encoder already tolerates it. "
        "Pluralistic Leaf Existence (a distinct embedding per parent context) needs "
        "top-down flow or per-edge materialisation. Deeper still: hierarchy need not be "
        "specified — soft/learned attachment becomes a well-posed problem once multi-"
        "benchmark data introduces attachment ambiguity.")

    # 9 -------------------------------------------------------------------
    doc.block("h1", "9.  Running the Code")
    doc.hrule()
    doc.block("code",
        "# local\n"
        "python scripts/train.py --d_hidden 128 --n_epochs 300 --device cuda \\\n"
        "      --embed_model all-mpnet-base-v2\n\n"
        "# SLURM (Narval) - A1 ablation matrix\n"
        "for j in baseline no_gki plain_gate pre_agg post_gru dual; do\n"
        "    sbatch jobs/job_a1_$j.sh\n"
        "done")
    doc.block("body",
        "Compute nodes are offline: pre-download dataset + encoder on the login node; "
        "job scripts set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE to load from cache, and "
        "PYTHONUNBUFFERED=1 keeps the .out log live and intact if a job is killed. "
        "pyarrow comes from the 'arrow' module (load before activating the venv), not "
        "pip.")
    doc.block("caption",
        "Companion docs in the repo: README.md, docs/EXPERIMENTS.md, docs/RELATED_WORK.md, docs/GATING.md.")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "PluralTree_Overview.pdf")
    doc = PDFBuilder(out)
    build(doc)
    doc.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
