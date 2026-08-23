#!/usr/bin/env python3
"""Merge a split wandb run back into the run it should have continued.

    scripts/merge_wandb_runs.py --into <target_id> --from <source_id> [options]

The hazard this repairs is the mirror image of the one `metric_receiver.py`
guards against. That guard refuses a run id it cannot account for, which is the
right default -- it fails toward "start a new run" rather than "overwrite
someone else's". The cost of that default is that a legitimate resume whose
checkpoint lost its `run_id` silently forks: the label keeps writing one CSV,
while wandb gains a second run holding the tail of the same experiment.

This script reunites them. It appends SOURCE's history onto TARGET, so TARGET
keeps its id, its URL and its creation time, and refuses to run unless the two
are genuinely contiguous in `Total Timesteps`.

It optionally repairs the two pieces of local state that let the fork happen:
the CSV (`--repair-csv`, which prepends TARGET rows the CSV is missing) and the
ownership sidecar plus each checkpoint's `RUNNING_STATS.json` run_id
(`--claim-label`), so the next resume of that label continues TARGET.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

STEP_KEY = "Total Timesteps"


def fetch(api, path):
    run = api.run(path)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")}
            for r in run.scan_history()]
    return run, rows


def order(rows):
    return sorted(rows, key=lambda r: float(r[STEP_KEY]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="edbaker-university-of-sussex")
    p.add_argument("--project", default="hivemind-rl")
    p.add_argument("--into", required=True, help="run id to keep (the original)")
    p.add_argument("--source", required=True, action="append",
                   help="run id whose history is appended; repeatable, applied in order")
    p.add_argument("--repair-csv", help="CSV to prepend the target's missing rows to")
    p.add_argument("--claim-label", help="label whose sidecar and checkpoint run_ids point at --into")
    p.add_argument("--checkpoints", default="bot/build/checkpoints")
    p.add_argument("--metrics-dir", default="bot/build/metrics")
    p.add_argument("--allow-running", action="store_true",
                   help="proceed even if a source run is still live (it will keep logging)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import wandb
    api = wandb.Api()
    base = f"{args.entity}/{args.project}"

    target, t_rows = fetch(api, f"{base}/{args.into}")
    t_rows = order(t_rows)
    print(f"target   {args.into} '{target.name}' state={target.state} "
          f"rows={len(t_rows)} ts={float(t_rows[0][STEP_KEY])/1e6:.1f}M"
          f"->{float(t_rows[-1][STEP_KEY])/1e6:.1f}M")

    batches = []
    cursor = float(t_rows[-1][STEP_KEY])
    for sid in args.source:
        src, s_rows = fetch(api, f"{base}/{sid}")
        s_rows = order(s_rows)
        print(f"source   {sid} '{src.name}' state={src.state} "
              f"rows={len(s_rows)} ts={float(s_rows[0][STEP_KEY])/1e6:.1f}M"
              f"->{float(s_rows[-1][STEP_KEY])/1e6:.1f}M")
        if src.state == "running" and not args.allow_running:
            sys.exit(f"REFUSING: {sid} is still running. Stop it first, or pass --allow-running.")
        s_rows = [r for r in s_rows if float(r[STEP_KEY]) > cursor]
        if not s_rows:
            sys.exit(f"REFUSING: {sid} adds nothing past {cursor/1e6:.1f}M timesteps.")
        cursor = float(s_rows[-1][STEP_KEY])
        batches.append((sid, s_rows))

    total = sum(len(b) for _, b in batches)
    start_step = target.lastHistoryStep + 1
    print(f"\nwould append {total} rows to {args.into} at _step {start_step}"
          f"..{start_step + total - 1}")

    if args.dry_run:
        print("(dry run: nothing written)")
    else:
        run = wandb.init(entity=args.entity, project=args.project,
                         id=args.into, resume="must")
        step = start_step
        for sid, rows in batches:
            for r in rows:
                run.log({k: v for k, v in r.items() if v is not None}, step=step)
                step += 1
            print(f"  appended {len(rows)} rows from {sid}")
        run.finish()
        print(f"merged: {target.url}")

    # --- local CSV: prepend whatever of the target the CSV is missing --------
    if args.repair_csv:
        path = Path(args.repair_csv)
        have = list(csv.DictReader(open(path)))
        first = float(have[0][STEP_KEY]) if have else float("inf")
        missing = [r for r in t_rows if float(r[STEP_KEY]) < first]
        print(f"\nCSV {path}: {len(have)} rows from {first/1e6:.1f}M; "
              f"{len(missing)} target rows missing")
        if missing and not args.dry_run:
            shutil.copy2(path, str(path) + ".prerepair")
            cols = list(have[0].keys()) if have else []
            for r in missing:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in missing + have:
                    w.writerow(["" if r.get(k) is None else r.get(k, "") for k in cols])
            print(f"  prepended; backup at {path}.prerepair")

    # --- ownership: sidecar + every checkpoint's run_id ----------------------
    if args.claim_label:
        label = args.claim_label
        sidecar = Path(args.metrics_dir) / f"{label}.wandb-id"
        print(f"\nsidecar {sidecar}: "
              f"{sidecar.read_text().strip() if sidecar.exists() else '<none>'} -> {args.into}")
        if not args.dry_run:
            sidecar.write_text(args.into)
        ckpt_root = Path(args.checkpoints) / label
        for stats in sorted(ckpt_root.glob("*/RUNNING_STATS.json")):
            try:
                d = json.loads(stats.read_text())
            except Exception as e:
                print(f"  {stats}: unreadable ({e!r})")
                continue
            if d.get("run_id") == args.into:
                continue
            print(f"  {stats.parent.name}/RUNNING_STATS.json: "
                  f"{d.get('run_id')!r} -> {args.into}")
            if not args.dry_run:
                d["run_id"] = args.into
                stats.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
