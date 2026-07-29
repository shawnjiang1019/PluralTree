"""Load open-ended queries from INFINITY-CHAT (Artificial Hivemind, Jiang et al.,
NeurIPS 2025).

This is a *query* loader, not a graph: INFINITY-CHAT has no population-attributed
leaf distributions and no relational paths (25 anonymous annotators per (Q,R)),
so it is used as a mode-collapse eval target — sample N responses per query and
measure intra-pool self-similarity (see evaluation/hivemind/). The 100-query
INFINITY-CHAT100 subset is the human-verified open-ended set used in the paper's
repetition study (Fig 4/5).

Dataset IDs (verified July 2026 — ``liweijiang/artificial-hivemind`` is a HF
*collection* page, NOT a loadable dataset; the repos use "infinite-chats"):

    liweijiang/infinite-chats-eval            100 rows, column: query   <- default
    liweijiang/infinite-chats-human-absolute  750 rows: user_query, response,
                                              human_labels (23-26 raters each)
    liweijiang/infinite-chats-human-pairwise  pairwise preferences
    liweijiang/infinite-chats-taxonomy        the 6-category / 17-subcategory taxonomy

``infinite-chats-eval`` IS the 100-query INFINITY-CHAT100 set, so no subsetting
is needed at the default. It exposes only ``query`` — there is no category column,
so ``category`` slicing is a no-op here (every row becomes "uncategorized"); the
per-category breakdown needs a join against ``infinite-chats-taxonomy``.

Field extraction stays tolerant (like the SubPOP loader) so a different config or
a schema change does not break the loader.
"""

from __future__ import annotations

import random

# Candidate field names, in priority order (the gated schema is unversioned).
_QUERY_FIELDS = ("query", "prompt", "instruction", "text", "input", "question")
_CATEGORY_FIELDS = ("category", "top_category", "top_level_category",
                    "label", "type")
# Truthy value in any of these marks a row as part of INFINITY-CHAT100.
_SUBSET100_FIELDS = ("infinity_chat_100", "infinity_chat100", "ic100",
                     "is_subset100", "subset100", "in_100")


def _first_field(row: dict, candidates: tuple[str, ...]) -> str | None:
    """First present, non-empty field value among ``candidates``."""
    for k in candidates:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _is_subset100(row: dict) -> bool:
    for k in _SUBSET100_FIELDS:
        if k in row and bool(row[k]):
            return True
    return False


def load_hivemind_queries(
    n: int = 100,
    *,
    hf_name: str = "liweijiang/infinite-chats-eval",
    split: str = "train",
    config: str | None = None,
    seed: int = 0,
    category: str | None = None,
    subset100: bool = True,
) -> list[tuple[int, str, str]]:
    """``(query_id, query, category)`` triples for the mode-collapse eval.

    ``subset100``: keep only INFINITY-CHAT100 rows when the schema marks them;
    if no such marker exists, fall back to a deterministic sample of ``n``.
    ``category``: restrict to one top-level taxonomy label (case-insensitive
    substring) for per-category slicing. ``n<=0`` keeps everything.
    """
    from datasets import load_dataset

    ds = (load_dataset(hf_name, config, split=split) if config
          else load_dataset(hf_name, split=split))

    rows: list[tuple[int, str, str]] = []
    marked = False
    for row in ds:
        q = _first_field(row, _QUERY_FIELDS)
        if not q:
            continue
        cat = _first_field(row, _CATEGORY_FIELDS) or "uncategorized"
        if subset100 and _is_subset100(row):
            marked = True
        rows.append((q, cat, _is_subset100(row)))

    # Deduplicate on query text, preserving first occurrence.
    seen: dict[str, tuple[str, bool]] = {}
    for q, cat, in100 in rows:
        seen.setdefault(q, (cat, in100))

    items = [(q, cat, in100) for q, (cat, in100) in seen.items()]

    if subset100 and marked:
        items = [(q, cat, _) for (q, cat, _) in items if _]  # keep marked only
    if category:
        c = category.lower()
        items = [t for t in items if c in t[1].lower()]

    items.sort(key=lambda t: t[0])                   # stable, text-ordered
    if n and n > 0 and len(items) > n:
        rng = random.Random(seed)
        items = rng.sample(items, n)
        items.sort(key=lambda t: t[0])

    return [(i, q, cat) for i, (q, cat, _) in enumerate(items)]
