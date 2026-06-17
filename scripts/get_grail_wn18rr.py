"""Download the GraIL inductive WN18RR benchmark (v1-v4) for comparable
inductive link-prediction evaluation.

Each version k has two graphs with DISJOINT entity sets:
    WN18RR_v{k}/      {train,valid,test}.txt  -- the training graph
    WN18RR_v{k}_ind/  {train,valid,test}.txt  -- the inductive graph (new entities)
At inductive test you train on WN18RR_v{k}, then embed the new entities in
WN18RR_v{k}_ind (from its train.txt as the support graph) and predict its test.txt.

Source: GraIL repo (Teru et al., ICML 2020), https://github.com/kkteru/grail
Run on the LOGIN node (compute nodes are offline):
    python scripts/get_grail_wn18rr.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/kkteru/grail/master/data"
OUT = "data/grail"
VERSIONS = (1, 2, 3, 4)
FILES = ("train.txt", "valid.txt", "test.txt")


def fetch(rel_path: str) -> None:
    url = f"{BASE}/{rel_path}"
    dst = os.path.join(OUT, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  skip (exists): {rel_path}")
        return
    print(f"  downloading: {rel_path}")
    urllib.request.urlretrieve(url, dst)


def main():
    print(f"Downloading GraIL inductive WN18RR (v1-v4) into {OUT}/ ...")
    ok = 0
    for v in VERSIONS:
        for sub in (f"WN18RR_v{v}", f"WN18RR_v{v}_ind"):
            for f in FILES:
                try:
                    fetch(f"{sub}/{f}")
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED {sub}/{f}: {e}", file=sys.stderr)
    print(f"Done ({ok} files present).")


if __name__ == "__main__":
    main()
