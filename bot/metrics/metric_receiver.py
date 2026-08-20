"""Metrics receiver for GigaLearn.

Replaces the stock receiver (which only forwards to wandb). GigaLearn imports
this as `python_scripts.metric_receiver` from its working directory; the build
copies this file over the stock one.

Two changes from stock:

1. **Every metric is written to a CSV**, one row per iteration. The console only
   prints a fixed subset chosen by GigaLearn -- `Rating/*`, `Player/*` and
   `Phase/*` never appear there. Comparing two runs from console output alone is
   therefore impossible, and reading wandb's offline binary format to recover
   them is worse than just writing a CSV.

2. **wandb is optional.** If it is missing, or WANDB_DISABLED is set, the CSV is
   still written. A missing wandb login should not cost you a training run.

The CSV lands next to the checkpoints: metrics/<run_name>.csv
"""

import csv
import os
import site
import sys

_wandb_run = None
_csv_path = None
_csv_columns = []
_csv_rows = []


def _csv_dir():
    return os.environ.get("HIVE_METRICS_DIR", "metrics")


def _id_sidecar(name):
    """Where this run LABEL's own wandb run id is recorded."""
    return os.path.join(_csv_dir(), f"{name}.wandb-id")


def _claim_run_id(name, id, csv_existed):
    """Decide whether `id` may be used to RESUME a wandb run under `name`.

    GigaLearn stores the wandb run id inside the checkpoint
    (RUNNING_STATS.json -> run_id) and passes it back on resume, so
    `wandb.init(id=..., resume="allow")` continues the same remote run. That is
    correct when a label resumes its own checkpoints, and silently destructive
    when a checkpoint is COPIED between labels: the new label adopts the old
    run's id, so it renames the old wandb run to the new label and appends its
    own iterations to the old run's history.

    That is not hypothetical. Seeding a p13strike calibration probe by copying
    `checkpoints/main-p12goal/250006016` into `checkpoints/main-p13cal/` carried
    run_id i4oupqtr with it, and two 2M-step probes renamed the 250M p12goal run
    to "main-p13cal" and stapled ~76 iterations onto its tail at x = 250-252M.
    The local CSVs were untouched -- they key off the label, not the id -- so
    only the remote copy was affected, but nothing warned about it at all.

    The invariant: a run id belongs to the LABEL that created it. This records
    that ownership in a sidecar next to the CSV and refuses any id it cannot
    account for, which is cheap, needs no network, and fails toward "start a new
    run" rather than toward "overwrite someone else's".
    """
    if not id:
        return None

    sidecar = _id_sidecar(name)
    if os.path.exists(sidecar):
        try:
            owned = open(sidecar).read().strip()
        except Exception:
            owned = ""
        if owned == id:
            return id
        print(f"[metrics] REFUSING wandb id {id!r}: {name!r} owns {owned!r}. "
              f"Starting a NEW wandb run. This means a checkpoint was copied "
              f"between run labels.")
        return None

    # No sidecar. Either this label predates the sidecar (legacy) or the id came
    # from somewhere else. A label that has run before under this name has a CSV
    # already; one that has not, has no business holding a run id.
    if csv_existed:
        print(f"[metrics] adopting wandb id {id!r} for {name!r} (pre-existing "
              f"run, recording ownership)")
        return id

    print(f"[metrics] REFUSING wandb id {id!r}: {name!r} has never run here, so "
          f"that id belongs to another label -- a checkpoint was copied. "
          f"Starting a NEW wandb run instead of hijacking it.")
    return None


def init(py_exec_path, project, group, name, id=None):
    """Called once by GigaLearn at startup. Returns a run id string."""
    global _wandb_run, _csv_path, _csv_columns, _csv_rows

    # --- CSV -------------------------------------------------------------
    os.makedirs(_csv_dir(), exist_ok=True)
    _csv_path = os.path.join(_csv_dir(), f"{name}.csv")

    # Rows are held in memory and the whole file is rewritten on each flush.
    #
    # Appending would be cheaper, but metrics do not all appear on iteration
    # one: `Rating/*` only exists once the version pool is non-empty, and
    # several PPO metrics first appear after the initial learn step. With an
    # append-only writer the header is fixed by the first row and every
    # late-arriving metric -- including the ratings, which are the whole point
    # of a self-play comparison -- silently never gets a column.
    #
    # A few hundred rows by a few dozen columns is trivial to rewrite.
    _csv_columns = []
    _csv_rows = []
    csv_existed = os.path.exists(_csv_path)
    if csv_existed:
        try:
            with open(_csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    _csv_rows.append({k: v for k, v in row.items() if v not in (None, "")})
            for row in _csv_rows:
                for k in row:
                    if k not in _csv_columns:
                        _csv_columns.append(k)
        except Exception as e:
            print(f"[metrics] could not read existing {_csv_path}: {e!r}")

    print(f"[metrics] logging to {_csv_path}")

    # --- wandb (optional) -------------------------------------------------
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        print("[metrics] wandb disabled by WANDB_DISABLED")
        return id or "csv-only"

    # GigaLearn passes its own interpreter path; without this fix wandb can
    # re-launch the training binary instead of Python.
    sys.executable = py_exec_path

    try:
        site_packages = os.path.join(os.path.dirname(py_exec_path), "Lib", "site-packages")
        if os.path.isdir(site_packages):
            sys.path.append(site_packages)
            site.addsitedir(site_packages)
        import wandb
    except Exception as e:
        print(f"[metrics] wandb unavailable ({e!r}); CSV only")
        return id or "csv-only"

    # A copied checkpoint carries the ORIGINAL run's wandb id. Resuming on it
    # would rename that run and append to its history. See _claim_run_id.
    resume_id = _claim_run_id(name, id, csv_existed)

    try:
        if resume_id:
            _wandb_run = wandb.init(project=project, group=group, name=name,
                                    id=resume_id, resume="allow")
        else:
            _wandb_run = wandb.init(project=project, group=group, name=name)
        try:
            with open(_id_sidecar(name), "w") as f:
                f.write(_wandb_run.id)
        except Exception as e:
            print(f"[metrics] could not record run-id ownership: {e!r}")
        return _wandb_run.id
    except Exception as e:
        print(f"[metrics] wandb.init failed ({e!r}); CSV only")
        return resume_id or "csv-only"


def add_metrics(metrics):
    """Called once per training iteration with a dict of metric name -> value."""
    _write_csv(metrics)

    if _wandb_run is not None:
        try:
            _wandb_run.log(metrics)
        except Exception as e:
            print(f"[metrics] wandb log failed: {e!r}")


def finish():
    """Called on every clean exit path (Q, SIGINT, step budget) so wandb
    doesn't mark the run crashed."""
    global _wandb_run
    if _wandb_run is not None:
        _wandb_run.finish()
        _wandb_run = None


def _write_csv(metrics):
    global _csv_columns, _csv_rows

    if _csv_path is None:
        return

    try:
        _csv_rows.append(dict(metrics))

        for k in sorted(metrics.keys()):
            if k not in _csv_columns:
                _csv_columns.append(k)

        # Write to a temp file and replace, so a crash mid-write cannot leave a
        # truncated CSV where a complete one used to be.
        tmp = _csv_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_csv_columns)
            for row in _csv_rows:
                w.writerow([row.get(c, "") for c in _csv_columns])
        os.replace(tmp, _csv_path)

    except Exception as e:
        print(f"[metrics] CSV write failed: {e!r}")
