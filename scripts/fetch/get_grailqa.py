"""Download the GrailQA v1.0 release (train + dev + test) for the grailqa loader.

The HF dataset ``dki-lab/grail_qa`` ships only a loader *script* (modern
``datasets`` won't run it), so we fetch the official release zip directly. It
extracts to ``GrailQA_v1.0/`` containing:

    grailqa_v1.0_train.json
    grailqa_v1.0_dev.json
    grailqa_v1.0_test_public.json   (no annotations; unused by the loader)

Source: https://dki-lab.github.io/GrailQA/  (Gu et al., 2021)
Run on the LOGIN node (compute nodes are offline):
    python scripts/get_grailqa.py --out data/grailqa
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

# The URL used by the official HF loader script (dki-lab/grail_qa/grail_qa.py).
_URL = "https://dl.orangedox.com/WyaCpL?dl=1"
_EXPECTED = ("grailqa_v1.0_train.json", "grailqa_v1.0_dev.json")


def main():
    ap = argparse.ArgumentParser(description="Download GrailQA v1.0 JSON")
    ap.add_argument("--out", default="data/grailqa", help="output directory")
    ap.add_argument("--url", default=_URL, help="override the release zip URL")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Already present (either flat or under GrailQA_v1.0/)?
    if all(os.path.exists(os.path.join(args.out, f))
           or os.path.exists(os.path.join(args.out, "GrailQA_v1.0", f))
           for f in _EXPECTED):
        print(f"GrailQA already present under {args.out}/ — nothing to do.")
        return

    print(f"Downloading GrailQA zip from {args.url} ...")
    try:
        with urllib.request.urlopen(args.url) as r:  # noqa: S310 (trusted release URL)
            data = r.read()
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to download: {e}", file=sys.stderr)
        print("If the mirror is down, grab the files manually from "
              "https://dki-lab.github.io/GrailQA/ and drop the *.json into "
              f"{args.out}/", file=sys.stderr)
        sys.exit(1)

    print(f"  got {len(data) / 1e6:.1f} MB; extracting ...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(args.out)

    present = [f for f in _EXPECTED
               if os.path.exists(os.path.join(args.out, f))
               or os.path.exists(os.path.join(args.out, "GrailQA_v1.0", f))]
    if len(present) < len(_EXPECTED):
        print(f"WARNING: extracted but missing {set(_EXPECTED) - set(present)}; "
              f"check the archive layout under {args.out}/", file=sys.stderr)
        sys.exit(1)
    print(f"Done. GrailQA JSON is under {args.out}/")


if __name__ == "__main__":
    main()
