"""
Bundles EVERYTHING about a run into one zip you can drag to another machine - so nobody
has to open TensorBoard on the training box and read numbers off a web UI.

    python export_run.py masked_v1
    python export_run.py masked_v1 other_run     # several at once
    python export_run.py --all

Produces `run_export_<timestamp>.zip` next to runs/, containing per run:
  - manifest.json          the exact config + git commit that produced it
  - eval_log.csv           the diagnostic eval rows (win rate, illegal/noop, margins)
  - scalars.csv            EVERY scalar TensorBoard recorded (train/*, rollout/*, eval/*,
                           time/*), flattened to tag,step,value - no TensorBoard needed
  - summary.txt            analyze_run.py's report, already rendered

Deliberately excludes model .zip checkpoints - they're large and aren't needed to analyze
what happened. Pass --with-model to include the final model.
"""
import argparse
import csv
import io
import os
import sys
import time
import zipfile

from analyze_run import RUNS_DIR, report


def find_event_files(run_dir):
    hits = []
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            if name.startswith("events.out.tfevents"):
                hits.append(os.path.join(root, name))
    return sorted(hits)


def scalars_to_rows(run_dir):
    """Flattens TensorBoard event files to (tag, step, wall_time, value) rows.

    Imported lazily and failure-tolerant on purpose: a missing/incompatible TensorBoard must
    not cost you the rest of the export, since eval_log.csv already carries the metrics that
    matter most and this is the supplementary detail."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        return None, f"tensorboard not importable ({exc}) - scalars.csv skipped"

    rows = []
    for path in find_event_files(run_dir):
        try:
            acc = EventAccumulator(path, size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                for ev in acc.Scalars(tag):
                    rows.append((tag, ev.step, ev.wall_time, ev.value))
        except Exception as exc:
            rows.append(("_export_error", 0, 0.0, f"{os.path.basename(path)}: {exc}"))
    if not rows:
        return None, "no TensorBoard scalars found"
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, None


def render_summary(name):
    """Captures analyze_run.report()'s printed output as text."""
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        report(name)
    except Exception as exc:
        print(f"(analyze_run failed: {type(exc).__name__}: {exc})")
    finally:
        sys.stdout = stdout
    return buf.getvalue()


def export(names, with_model=False):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.abspath(os.path.join(RUNS_DIR, "..", f"run_export_{stamp}.zip"))
    notes = []

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            run_dir = os.path.join(RUNS_DIR, name)
            if not os.path.isdir(run_dir):
                notes.append(f"{name}: NOT FOUND at {run_dir}")
                continue

            for fname in ("manifest.json", "eval_log.csv"):
                fpath = os.path.join(run_dir, fname)
                if os.path.exists(fpath):
                    z.write(fpath, f"{name}/{fname}")
                else:
                    notes.append(f"{name}: no {fname}")

            rows, err = scalars_to_rows(run_dir)
            if rows:
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["tag", "step", "wall_time", "value"])
                w.writerows(rows)
                z.writestr(f"{name}/scalars.csv", buf.getvalue())
                tags = sorted({r[0] for r in rows})
                notes.append(f"{name}: {len(rows)} scalar points across {len(tags)} tags")
            else:
                notes.append(f"{name}: {err}")

            z.writestr(f"{name}/summary.txt", render_summary(name))

            if with_model:
                model_zip = os.path.join(run_dir, "model.zip")
                if os.path.exists(model_zip):
                    z.write(model_zip, f"{name}/model.zip")
                else:
                    notes.append(f"{name}: no model.zip")

        z.writestr("EXPORT_NOTES.txt", "\n".join(notes) or "(nothing to report)")

    size_mb = os.path.getsize(out_path) / 1e6
    print("\n".join(notes))
    print(f"\nWROTE {out_path}  ({size_mb:.1f} MB)")
    print("Copy that one file over; it has everything needed to analyze these runs.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run names; omit with --all")
    parser.add_argument("--all", action="store_true", help="export every run found")
    parser.add_argument("--with-model", action="store_true",
                         help="also include model.zip (much larger file)")
    args = parser.parse_args()

    names = args.runs
    if args.all or not names:
        names = sorted(d for d in os.listdir(RUNS_DIR)
                        if os.path.isdir(os.path.join(RUNS_DIR, d)))
    export(names, with_model=args.with_model)
