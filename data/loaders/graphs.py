"""One place to turn a --dataset name into a graph.

Fourteen call sites currently repeat the same three-line if/elif and their own
copy of `choices=[...]`, so adding a third dataset meant fourteen edits and
fourteen chances to miss one. They resolve here instead.

Imports stay INSIDE the branches: loading opinionqa parses the raw ATP CSVs and
loading issp reads Stata files, so a module-level import would make every script
that merely mentions a dataset pay for all of them.

    from data.loaders.graphs import DATASETS, load_graph
    ap.add_argument("--dataset", choices=list(DATASETS), default="opinionqa")
    graph = load_graph(args.dataset, split_seed=args.seed)
"""

from __future__ import annotations

DATASETS = ("opinionqa", "globalopinionqa", "issp")


def load_graph(dataset: str, *, split_seed: int = 42, leakage_safe: bool = True,
               **kwargs):
    """Graph for ``dataset``. Extra kwargs go to that loader only.

    Every loader returns an OpinionQAGraph-compatible object -- issp reuses
    opinionqa's `_build_graph` unchanged -- so the scout, the reward and the
    renderers are dataset-agnostic downstream of this call.
    """
    if dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        return load_opinionqa(split_seed=split_seed, leakage_safe=leakage_safe,
                              **kwargs)
    if dataset == "globalopinionqa":
        from data.loaders.globalopinionqa import load_globalopinionqa
        return load_globalopinionqa(split_seed=split_seed,
                                    leakage_safe=leakage_safe, **kwargs)
    if dataset == "issp":
        from data.loaders.issp import load_issp
        return load_issp(split_seed=split_seed, leakage_safe=leakage_safe,
                         **kwargs)
    raise SystemExit(f"unknown dataset {dataset!r}; choose from {list(DATASETS)}")
