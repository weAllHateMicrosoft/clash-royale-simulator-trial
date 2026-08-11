# HANDOFF — read this first (2026-08-11)

Everything needed to keep working alone, or to hand this to someone else.
Full history: `TRAINING.md`. Engine: `SIMULATOR_STATUS.md`. Calibration: `CALIBRATION_PLAYBOOK.md`.

## Machines

- **Mac** (analysis, playing, editing): repo `~/Documents/GitHub/clash-royale-simulator-trial`.
  Python for everything: `/Users/leafer/Documents/GitHub/clash-royale-simulator/.venv/bin/python3`
  (Python 3.13, has sb3-contrib). Run scripts from `src/clasher_new`.
- **Windows** (training, RTX 4080 SUPER, Python 3.11, conda env `Arron`), project at
  `C:\Users\Administrator\Desktop\Arron\clash-royale-simulator-portable`.
  **Shared with other people — leave cores free, 12 envs is a polite footprint.**

## The three Windows traps that cost hours

1. **QuickEdit freeze.** Clicking inside a PowerShell window puts it in selection mode
   (title shows `选择`), which BLOCKS stdout and **freezes training**. Symptom: CPU drops to
   ~2%, no output, then a keypress makes everything appear at once. This wasted most of one
   night and looked exactly like a hang.
   **Always redirect output**: append `*> run.log` to every training command. Then watch it
   from a *different* window with `Get-Content run.log -Tail 20`. The training console
   staying blank is then CORRECT, not a hang.
2. **git paths.** `git` commands need the repo root; `python train.py` needs
   `src\clasher_new` (the sim opens `gamedata.json` relative to the working directory).
3. **conda.** If the prompt says `(base)` instead of `(Arron)`, `sb3_contrib` will be
   missing. `conda activate Arron` first, every window.

## Moving files (GitHub is unreliable from that network)

Try first, from the repo root:

    git fetch origin main; git checkout origin/main -- src/clasher_new

Do NOT use plain `git pull` there — the local branch is `master`, the remote is `main`, and
there are local edits; `checkout` overwrites cleanly and leaves untracked `runs/` alone.

**When GitHub is blocked, hand-copy over ToDesk.** Code files are tiny (~10–25 KB each), so
throttling doesn't matter. Copy only what changed. Move the SCRIPT to the model, not the
model to the script — checkpoints are 38 MB each.

ToDesk gotchas actually hit:
- A file being copied looks corrupt until finished (valid ZIP header, no central directory).
  Check it's still growing before concluding it's broken.
- `runs\<name>\model.zip` is **0 bytes while a run is still going** — the final save only
  happens at the end. Use `runs\<name>\checkpoints\cr_*_steps.zip` instead; those are complete.
- **Ctrl+C exactly once** to stop a run and wait for `Saving model.`; a second Ctrl+C during
  the save truncates the file.

## Cross-Python-version checkpoints

Windows trains on **Python 3.11**, the Mac runs **3.13**. SB3 pickles the feature-extractor
class *by value*, so loading a Windows checkpoint on the Mac fails with
`SystemError: unknown opcode`. Weights themselves are fine. Load like this (already built
into `play_vs_ai.py` and `responsiveness_probe.py`):

    MaskablePPO.load(path, device="cpu", custom_objects={
        "policy_kwargs": dict(features_extractor_class=CRFeatureExtractor,
                               features_extractor_kwargs=dict(features_dim=512, net_scale=2.0))})

## Current state

`maxed_b` (6.72M steps) went from 60% → ~100% win rate vs a masked random opponent, and beat
its own 120k-step checkpoint **68.5% over 200 games (±6.6%)** — real improvement, measured.
`explained_variance` reached 0.72 (was ~0 in every earlier run).

**But it plays an open-loop script.** Confirmed by watching it and by
`responsiveness_probe.py`:

| measure | value | meaning |
|---|---|---|
| divergence, Giant pushing at it | 0.040 (noise floor 0.053) | does not react at all |
| spells cast with enemy cluster present | 3.7% vs 7.3% empty | casts LESS when there's a target |
| deploys on own half under push | 99.7% vs 99.7% | identical, no defensive response |

It spams MiniPekka at the bridge, never uses Giant (alone it's a bad trade, and a Giant+support
push is never sampled), and fires Arrows on a timer.

**Root cause**: it trained against a single frozen snapshot of itself, which is also a script,
so the enemy board was predictable *from the clock*. A feature redundant with time gets
ignored — the enemy-troop channels never earned weight. Responsiveness can't be installed by
a reward term; it only pays off against opponents you can't predict.

## What was just built to fix it (untested at scale — this is the live experiment)

- `opponents.py` — per-episode mixture: random / bridge-spammer / defender / giant-pusher /
  past checkpoints. **They are opponents, not teachers** — the learner never imitates them,
  so its strategy stays self-discovered.
- **Elixir handicap** (`PlayerState.elixir_rate`, 1.0–1.6 for weak bots only, checkpoints
  always 1.0). Makes weak bots threatening **without touching card stats** — those are
  calibrated against real telemetry and changing them would teach facts about a game that
  doesn't exist.
- **Universal anti-leak guard.** Measured: DefenderBot sat at capped elixir 96% of the match
  with 0 deploys; GiantPushBot deadlocked at 97% because cards only cycle when played. Both
  now 0.0% capped.
- **`--ent-coef`** — SB3 defaults to 0.0, so nothing stopped the policy collapsing early
  (entropy fell to -0.09). Keeps exploration alive without teaching any pattern.
- Randomised starting elixir; `play_vs_ai.py` now shuffles the AI deck (it didn't, which is
  why every match you played looked identical).

## The command

    python train.py --run-name responsive_a --device cuda --n-envs 12 --timesteps 12000000 --batch-size 2048 --n-steps 2048 --net-scale 2.0 --features-dim 512 --ent-coef 0.01 --opponent-pool --handicap-max 1.6 --eval-freq 300000 --eval-games 20 --seed 1 --note "opponent mixture" *> responsive_a.log

**Expect win-rate-vs-random to DROP from 96%.** That's correct — it now faces defenders and
handicapped attackers. Judge this run by the probe, not that number.

## How to tell if it worked

Copy a checkpoint to the Mac, then:

    python responsiveness_probe.py <new_checkpoint> cr_6720000_steps

Three numbers decide it, all currently flat:
1. **Giant divergence > 0.05** (now 0.040)
2. **spell rate with a cluster > spell rate on empty board** (now backwards)
3. **own-half deploys higher under push** (now identical)

Other tools: `head_to_head.py A B --games 200` (does the new model beat the old one — the
only test that still works once win-rate-vs-random saturates at 100%), `export_run.py NAME`
(bundles manifest + eval log + all TensorBoard scalars into one zip), `analyze_run.py NAME`
(prints a report with a noise floor and says when a change is *within noise*),
`preflight.py --device cuda --n-envs 12` (60s check before committing hours).

## Hard-won rules

- **Never trust a number without its noise floor.** 20-game win rates carry ±11pp. Four
  multi-hour runs produced conclusions that had to be thrown out because the eval played
  5 games and reported mean reward.
- **Run `preflight.py` before any long run.** Parallel training has broken three times for
  machine-specific reasons.
- **Don't change card stats to steer the model.** It breaks the only thing that makes this
  simulator worth having.
- Reward totals are NOT comparable across runs with different shaping. Win rate is.
- `maxed_a` died at 1.7M steps with `EOFError` (SubprocVecEnv worker death) — a crash-resume
  wrapper is still not built and would have saved 3.8 hours.

## Open / next

- Does the opponent mixture produce responsiveness? (the live experiment)
- Giant pushes: entropy keeps the option alive but does not guarantee discovery. Honest
  uncertainty.
- Phantom Arrows hypothesis, untested: the elixir-overflow penalty may be paying it to dump
  elixir into nothing. Check whether those casts correlate with being near the cap.
- Crash-resume wrapper.
- Web deployment of a trained model.
- Collaboration with `github.com/vegetableleaf/ClashAI` — he already wrote a parser for our
  telemetry (`memory_ingest.py`) and verified our frames match to 0.8% on HP, but has no way
  to capture it. We have the capture rig. Share data, not code; split by card.
