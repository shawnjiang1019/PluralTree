"""Collect and compare results from SLURM job logs.

Scans a logs directory for `a1_<variant>_<jobid>.out` files (and their `.err`
siblings), extracts the run config, final test metrics, best validation MRR, and
job status (finished / running / cancelled / crashed), then prints a comparison
table sorted by test MRR.

Parses both output formats:
  - the new grep-able line:   RESULT | <config> | best_val_mrr=.. | test_mrr=.. h@1=.. ...
  - the older block:          "Final test evaluation:" followed by "  mrr: 0.4557" lines

Usage:
    python scripts/collect_results.py                 # scans ./logs, pattern a1_*.out
    python scripts/collect_results.py path/to/logs
    python scripts/collect_results.py logs --glob 'a1_*.out' --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"(?P<variant>.+?)_(?P<jobid>\d+)\.out$")

RESULT_RE = re.compile(
    r"RESULT\s*\|\s*(?P<config>.*?)\s*\|\s*best_val_mrr=(?P<bval>[\d.]+)\s*\|\s*"
    r"test_mrr=(?P<mrr>[\d.]+)\s+h@1=(?P<h1>[\d.]+)\s+h@3=(?P<h3>[\d.]+)\s+h@10=(?P<h10>[\d.]+)"
)
BEST_VAL_RE = re.compile(r"best val MRR:\s*([\d.]+)", re.IGNORECASE)
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
METRIC_LINE_RE = re.compile(
    r"^\s*(mrr|hits@1|hits@3|hits@10)\s*:\s*([\d.]+)", re.IGNORECASE | re.MULTILINE
)


def parse_out(text: str) -> dict:
    """Extract config, metrics, best val MRR, and epoch progress from an .out file."""
    rec: dict = {
        "config": None, "best_val": None,
        "mrr": None, "h1": None, "h3": None, "h10": None,
        "epoch": None, "epochs_total": None,
    }

    # 1) Preferred: the single RESULT summary line.
    m = RESULT_RE.search(text)
    if m:
        rec.update(
            config=m.group("config"),
            best_val=float(m.group("bval")),
            mrr=float(m.group("mrr")),
            h1=float(m.group("h1")),
            h3=float(m.group("h3")),
            h10=float(m.group("h10")),
        )
    else:
        # 2) Fallback: the "Final test evaluation:" block.
        if "Final test evaluation" in text:
            tail = text.split("Final test evaluation", 1)[1]
            vals = {k.lower(): float(v) for k, v in METRIC_LINE_RE.findall(tail)}
            rec["mrr"] = vals.get("mrr")
            rec["h1"] = vals.get("hits@1")
            rec["h3"] = vals.get("hits@3")
            rec["h10"] = vals.get("hits@10")

    # config from the banner if not in a RESULT line
    if rec["config"] is None:
        mb = re.search(r"RUN:\s*(.+)", text)
        if mb:
            rec["config"] = mb.group(1).strip()

    # best val MRR (max over all logged "new best" lines) if not already set
    if rec["best_val"] is None:
        bvals = [float(x) for x in BEST_VAL_RE.findall(text)]
        if bvals:
            rec["best_val"] = max(bvals)

    # epoch progress (last "Epoch N/M" line)
    epochs = EPOCH_RE.findall(text)
    if epochs:
        rec["epoch"], rec["epochs_total"] = int(epochs[-1][0]), int(epochs[-1][1])

    return rec


def parse_err(text: str) -> str:
    """Classify the job's terminal status from its .err file."""
    if "DUE TO TIME LIMIT" in text or "CANCELLED" in text:
        return "CANCELLED"
    if "out-of-memory" in text.lower() or "oom-kill" in text.lower():
        return "OOM"
    if "Traceback (most recent call last)" in text or "Error" in text:
        return "CRASH"
    return ""


def status_for(rec: dict, err_status: str) -> str:
    """Combine .out completeness and .err signal into one status."""
    done = rec["mrr"] is not None
    if done:
        return "DONE"
    if err_status:
        return err_status
    if rec["epoch"] is not None:
        return "RUNNING"
    return "NO-OUTPUT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def query_sacct(jobids: list[str]) -> dict[str, dict]:
    """Best-effort live SLURM status for the given job IDs (empty if sacct absent)."""
    try:
        out = subprocess.run(
            ["sacct", "-j", ",".join(jobids), "-n", "-P",
             "--format=JobID,JobName,State,Elapsed"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return {}
    res: dict[str, dict] = {}
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid = parts[0].split(".")[0]  # strip .batch/.extern step suffixes
        if jid in res:               # keep the main step, not sub-steps
            continue
        res[jid] = {"name": parts[1], "state": parts[2].strip(), "elapsed": parts[3]}
    return res


def collect(logs_dir: str, pattern: str, jobs: set[str] | None, use_sacct: bool) -> list[dict]:
    rows: list[dict] = []
    seen_jobids: set[str] = set()

    # When filtering by job id, scan all .out files and match on id, not name.
    scan_pattern = "*.out" if jobs else pattern
    for out_path in sorted(glob.glob(os.path.join(logs_dir, scan_pattern))):
        fname = os.path.basename(out_path)
        m = FNAME_RE.search(fname)
        variant = m.group("variant") if m else fname
        jobid = m.group("jobid") if m else "?"

        if jobs is not None and jobid not in jobs:
            continue
        seen_jobids.add(jobid)

        with open(out_path, "r", errors="replace") as f:
            out_text = f.read()
        rec = parse_out(out_text)

        err_path = out_path[:-4] + ".err"
        err_status = ""
        if os.path.exists(err_path):
            with open(err_path, "r", errors="replace") as f:
                err_status = parse_err(f.read())

        rec.update(variant=variant, jobid=jobid,
                   status=status_for(rec, err_status))
        rows.append(rec)

    # Live SLURM status (fills in pending/running jobs with no logs yet).
    sacct = query_sacct(sorted(jobs)) if (jobs and use_sacct) else {}

    # Requested jobs that produced no .out yet → stub rows.
    if jobs is not None:
        for jid in sorted(jobs):
            if jid in seen_jobids:
                continue
            info = sacct.get(jid, {})
            rows.append({
                "config": None, "best_val": None,
                "mrr": None, "h1": None, "h3": None, "h10": None,
                "epoch": None, "epochs_total": None,
                "variant": info.get("name", "?"), "jobid": jid,
                "status": info.get("state", "NO-LOG"),
            })

    # Overlay live SLURM state onto rows that aren't finished.
    for r in rows:
        if r["mrr"] is None and r["jobid"] in sacct:
            state = sacct[r["jobid"]]["state"]
            if state and state.upper() not in ("COMPLETED",):
                r["status"] = state
    return rows


def fmt(v, width, prec=4):
    if v is None:
        return "-".ljust(width)
    if isinstance(v, float):
        return f"{v:.{prec}f}".ljust(width)
    return str(v).ljust(width)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No matching .out files found.")
        return

    # Sort: DONE first by test MRR desc, then the rest.
    rows.sort(key=lambda r: (r["mrr"] is None, -(r["mrr"] or 0)))

    hdr = (f"{'VARIANT':22}{'JOBID':10}{'STATUS':11}{'EPOCH':8}"
           f"{'BEST_VAL':10}{'TEST_MRR':10}{'H@1':8}{'H@3':8}{'H@10':8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        epoch = f"{r['epoch']}/{r['epochs_total']}" if r["epoch"] else "-"
        print(
            fmt(r["variant"], 22) + fmt(r["jobid"], 10) + fmt(r["status"], 11)
            + fmt(epoch, 8) + fmt(r["best_val"], 10) + fmt(r["mrr"], 10)
            + fmt(r["h1"], 8) + fmt(r["h3"], 8) + fmt(r["h10"], 8)
        )

    done = [r for r in rows if r["status"] == "DONE"]
    print(f"\n{len(done)}/{len(rows)} runs finished.")
    # Highlight the headline A1 comparison if both are present.
    by_var = {r["variant"]: r for r in done}
    base = next((by_var[v] for v in by_var if "baseline" in v), None)
    nogki = next((by_var[v] for v in by_var if "no_gki" in v), None)
    if base and nogki:
        delta = base["mrr"] - nogki["mrr"]
        verdict = "GKI helps" if delta > 0 else "GKI does NOT help (no-GKI >= full)"
        print(f"A1 key comparison: baseline MRR {base['mrr']:.4f} vs "
              f"no_gki MRR {nogki['mrr']:.4f}  (delta {delta:+.4f}) -> {verdict}")


def write_csv(rows: list[dict], path: str) -> None:
    cols = ["variant", "jobid", "status", "epoch", "epochs_total",
            "best_val", "mrr", "h1", "h3", "h10", "config"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Collect SLURM job results into a table.")
    p.add_argument("logs_dir", nargs="?", default="logs", help="directory of .out/.err logs")
    p.add_argument("--glob", default="a1_*.out", help="filename pattern (default a1_*.out)")
    p.add_argument("--jobs", default=None,
                   help="only these job IDs (comma/space separated); matches by id, any name")
    p.add_argument("--sacct", action="store_true",
                   help="query live SLURM status (sacct) for pending/running jobs")
    p.add_argument("--csv", default=None, help="also write results to this CSV path")
    args = p.parse_args()

    jobs = None
    if args.jobs:
        jobs = {j for j in re.split(r"[,\s]+", args.jobs.strip()) if j}

    rows = collect(args.logs_dir, args.glob, jobs, args.sacct)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
