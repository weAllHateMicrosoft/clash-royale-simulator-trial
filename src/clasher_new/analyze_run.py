"""
Prints a readable report for one or more training runs, from the files the run already
wrote - no TensorBoard, no graph-hunting, no exporting CSVs by hand from a web UI.

    python analyze_run.py                      # every run under runs/
    python analyze_run.py parallel_real1       # one run
    python analyze_run.py run_a run_b          # side-by-side comparison

Reads `runs/<name>/manifest.json` (what produced the run) and `runs/<name>/eval_log.csv`
(what it actually did). Runs from before eval_log.csv existed will only show the manifest
part - that's expected, not an error.

The reason this reports win rate rather than mean reward: reward totals include whatever
shaping terms were active in that run, so two runs with different reward designs are not
comparable by reward - which was exactly how earlier runs were being compared. Win rate
against a fixed random opponent has no such problem.
"""
import csv
import json
import os
import sys

def _resolve_runs_dir():
    """train.py writes to `runs/` relative to whatever directory it was LAUNCHED from, so
    the same repo can end up with runs in more than one place (e.g. `src/clasher_new/runs/`
    when launched from there, and `<repo>/runs/` when launched from the root). Checking both
    means the analyzer finds runs regardless of how training was invoked, instead of
    reporting a perfectly good run as missing."""
    candidates = [
        os.path.join(os.getcwd(), "runs"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


RUNS_DIR = _resolve_runs_dir()

# An eval point is this many games (see eval_diagnostics.N_EVAL_GAMES). With ~30 games the
# standard error on a win rate is roughly +-9 percentage points, so differences smaller than
# that between two runs are not real differences. Printed alongside results so the number
# doesn't get over-read.
def _stderr_pct(win_rate, games):
    if not games:
        return float("nan")
    return 100.0 * (win_rate * (1 - win_rate) / games) ** 0.5


def load_run(name):
    run_dir = os.path.join(RUNS_DIR, name)
    manifest_path = os.path.join(run_dir, "manifest.json")
    csv_path = os.path.join(run_dir, "eval_log.csv")
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
                rows.append(parsed)
    return manifest, rows


def _trend(rows, key, tail=3):
    """Compares the mean of the first `tail` eval points against the last `tail`, rather than
    first-vs-last single points - single points at 30 games are too noisy to read a trend
    from, which is the mistake that made earlier eval curves look meaningful."""
    if len(rows) < 2:
        return None, None, None
    head_vals = [r[key] for r in rows[:tail] if isinstance(r.get(key), float)]
    tail_vals = [r[key] for r in rows[-tail:] if isinstance(r.get(key), float)]
    if not head_vals or not tail_vals:
        return None, None, None
    start = sum(head_vals) / len(head_vals)
    end = sum(tail_vals) / len(tail_vals)
    return start, end, end - start


def report(name):
    manifest, rows = load_run(name)
    print("=" * 72)
    print(f"RUN: {name}")
    print("=" * 72)

    if manifest:
        print(f"  note        : {manifest.get('note') or '(none)'}")
        print(f"  status      : {manifest.get('status')}   "
              f"steps={manifest.get('final_timesteps')}")
        args = manifest.get("args", {})
        print(f"  config      : n_envs={args.get('n_envs')} device={args.get('device')} "
              f"eval_games={args.get('eval_games', '5 (pre-fix default)')}")
        git = manifest.get("git", {})
        dirty = " (UNCOMMITTED CHANGES - exact code not recoverable)" if git.get("dirty") else ""
        print(f"  git         : {str(git.get('commit'))[:12]}{dirty}")
    else:
        print("  (no manifest.json)")

    if not rows:
        print("\n  No eval_log.csv - this run predates the diagnostic eval, so only the\n"
              "  manifest is available. Its TensorBoard curve, if any, was 5-game noise;\n"
              "  see eval_diagnostics.py for why that can't be trusted.\n")
        return

    last = rows[-1]
    games = int(last.get("games") or 0)
    err = _stderr_pct(last.get("win_rate", 0.0), games)
    print(f"\n  FINAL ({games} games/eval, +-{err:.0f}pp standard error on win rate)")
    print(f"    win / loss / draw : {last['win_rate']:.0%} / {last['loss_rate']:.0%} / "
          f"{last['draw_rate']:.0%}")
    print(f"    crowns for/against: {last['mean_crowns_for']:.2f} / "
          f"{last['mean_crowns_against']:.2f}")
    print(f"    tower HP left     : own {last['mean_own_tower_hp']:.0f}  "
          f"enemy {last['mean_enemy_tower_hp']:.0f}")
    print(f"    mean battle time  : {last['mean_battle_time']:.0f}s")
    print(f"    mean reward       : {last['mean_reward']:.2f}  "
          f"(NOT comparable across runs with different reward shaping)")

    print("\n  BEHAVIOR (these separate 'not learning' from 'pressing dead buttons')")
    print(f"    illegal deploys   : {last['illegal_rate']:.0%} of deploy attempts rejected")
    print(f"    no-op actions     : {last['noop_rate']:.0%} of decisions")
    print(f"    at elixir cap     : {last['elixir_capped_rate']:.0%} of decisions (wasted elixir)")

    print("\n  TREND (mean of first 3 eval points -> mean of last 3)")
    for key, label, pct in [("win_rate", "win rate", True),
                             ("illegal_rate", "illegal rate", True),
                             ("noop_rate", "no-op rate", True),
                             ("mean_reward", "mean reward", False)]:
        start, end, delta = _trend(rows, key)
        if start is None:
            continue
        if pct:
            print(f"    {label:<14}: {start:.0%} -> {end:.0%}  ({delta:+.0%})")
        else:
            print(f"    {label:<14}: {start:.2f} -> {end:.2f}  ({delta:+.2f})")

    win_start, win_end, win_delta = _trend(rows, "win_rate")
    print("\n  READING")
    if win_end is not None and games:
        # 2 standard errors as the "is this even a real change" bar - deliberately
        # conservative, because reading noise as signal is exactly what went wrong before.
        bar = 2 * _stderr_pct(win_end, games) / 100.0
        if abs(win_delta) < bar:
            print(f"    Win rate change ({win_delta:+.0%}) is within noise (+-{bar:.0%}).")
            print("    This run does NOT show learning. Do not read the reward curve as if")
            print("    it does - check illegal/no-op rates above for whether the policy is")
            print("    even able to act meaningfully.")
        elif win_delta > 0:
            print(f"    Win rate improved by {win_delta:+.0%}, beyond the +-{bar:.0%} noise")
            print("    floor. This is real learning.")
        else:
            print(f"    Win rate DROPPED by {win_delta:+.0%}, beyond the +-{bar:.0%} noise")
            print("    floor. Something in this configuration is actively harmful.")
    if last.get("illegal_rate", 0) > 0.5:
        print(f"    >{last['illegal_rate']:.0%} of deploy attempts are illegal. Action masking")
        print("    (sb3-contrib MaskablePPO) is the known fix and is likely to matter more")
        print("    than any reward tuning until this comes down.")
    print()


def main(names):
    if not names:
        if not os.path.isdir(RUNS_DIR):
            print(f"No runs/ directory at {os.path.abspath(RUNS_DIR)}")
            return
        names = sorted(d for d in os.listdir(RUNS_DIR)
                        if os.path.isdir(os.path.join(RUNS_DIR, d)))
    for name in names:
        report(name)


if __name__ == "__main__":
    main(sys.argv[1:])
