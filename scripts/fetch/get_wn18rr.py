"""Download + prepare WN18RR into data/wn18rr/ (run on a machine WITH internet).

Narval compute nodes are offline, so run this on the LOGIN node once. It fetches
the KG-BERT WN18RR release (triples + entity text) and writes the files
data/wordnet.py expects:

    data/wn18rr/train.txt          (from train.tsv)
    data/wn18rr/valid.txt          (from dev.tsv)
    data/wn18rr/test.txt           (from test.tsv)
    data/wn18rr/entity2text.txt    (id <TAB> short text)
    data/wn18rr/entity2textlong.txt(id <TAB> definition)

All files are head<TAB>relation<TAB>tail / id<TAB>text — copied verbatim, only
renamed. Usage:

    python scripts/get_wn18rr.py                 # -> data/wn18rr
    python scripts/get_wn18rr.py --out some/dir
"""

from __future__ import annotations

import argparse
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR"

# remote filename -> local filename
FILES = {
    "train.tsv":           "train.txt",
    "dev.tsv":             "valid.txt",
    "test.tsv":            "test.txt",
    "entity2text.txt":     "entity2text.txt",
    "entity2textlong.txt": "entity2textlong.txt",
    "relation2text.txt":   "relation2text.txt",
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pluraltree/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser(description="Download + prepare WN18RR.")
    p.add_argument("--out", default="data/wn18rr", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ok, failed = [], []
    for remote, local in FILES.items():
        url = f"{BASE}/{remote}"
        dest = os.path.join(args.out, local)
        try:
            data = fetch(url)
            with open(dest, "wb") as f:
                f.write(data)
            n_lines = data.count(b"\n")
            ok.append(f"  {local:22s} <- {remote}  ({n_lines} lines)")
        except Exception as e:  # noqa: BLE001
            failed.append(f"  {remote}: {e}")

    print("Downloaded:")
    print("\n".join(ok) if ok else "  (none)")
    if failed:
        print("FAILED (the loader can still run with whatever triples succeeded;")
        print("entity2text files are optional but recommended):")
        print("\n".join(failed))
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
