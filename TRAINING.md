# Training Status — Handoff Summary

Scope: PPO self-play training for the 8-card simulator, its reward design, action space,
and actual training runs. Simulator/engine calibration lives in `SIMULATOR_STATUS.md`
instead — this doc assumes that side is a black box that works, not something to re-verify
here. Read this instead of re-deriving context from scratch.

## Bottom line

Pipeline confirmed working end-to-end, including on a remote CUDA machine (RTX 4080 SUPER,
i7-12700KF, 20 logical processors). Three full 1M-step overnight runs completed, plus a
fourth (`parallel_real1`) combining reward v2 + joint action space + parallel collection —
see "Where things stand right now" below, its completion status is unconfirmed as of this
writing. **No "good" agent yet, and none was expected** — ~2,800 self-play games per run is
very little by RL standards; the point of these runs was validating the pipeline and
comparing isolated changes, not producing strong play.

**If you're picking this up cold** (returning after a gap, or a new person entirely — e.g.
someone with spare AWS/Azure/OpenAI/Anthropic credits being asked to help run training),
read "Where things stand right now" first, then the sections below for the reasoning
behind each piece.

## Infra that exists (`src/clasher_new/`)

- `environment.py` - Gymnasium `CREnv`: observation (32x18 grid, 15 channels/tile + hand +
  elixir), action space, reward function.
- `train.py` - PPO + custom CNN feature extractor (`CRFeatureExtractor`, embeds entity
  types), self-play (opponent = periodic snapshot of the learner's own weights, copied
  every 50k steps via `WeightsCopyingCallback`), eval-vs-random logging
  (`RandomEvalCallback`), per-run output folders (`runs/<name>/`: model, checkpoints,
  tensorboard log, `manifest.json` with full config + git commit + note).
- `play_vs_ai.py` - load any checkpoint, play against it yourself in a pygame window.
  Handles both action-space formats (see below) automatically via
  `isinstance(model.action_space, gym.spaces.Discrete)`.
- Run a training job: `python train.py --run-name X --device cuda --timesteps N --note
  "..."`. `--checkpoint-name runs/<name>/model` to continue from a checkpoint (loads on
  CPU first, then moves to target device - loading a cross-platform checkpoint straight
  onto CUDA has caused hard native crashes on Windows, this avoids that).
- Compare runs: `tensorboard --logdir runs`.

## Action space: two formats exist, know which is which

- **Old**: `MultiDiscrete([5, 32, 18])` - slot/y/x sampled as three independent,
  simultaneous choices from the same snapshot. Card choice and board position aren't
  actually conditioned on each other in this format.
- **Current**: flat `Discrete(5*32*18)` = 2880, via `encode_action`/`decode_action` in
  `environment.py`. Forces card+position into one joint decision - the position logits
  can only be expressed in terms of "good for this specific card," not evaluated
  independently. Uses SB3's standard categorical distribution, not a custom policy - no
  new log-prob math, avoids the correctness risk a full autoregressive-head
  implementation would carry.
- **Was the single biggest defect; FIXED 2026-08-10.** The 2880 choices accounted for
  legality not at all - most troop/tile combinations are outside the deploy zone, and most
  cards are unaffordable at any given moment. **Measured: 91% of a policy's deploy attempts
  were rejected by the engine**, i.e. ~9 in 10 actions did literally nothing, so training
  went into discovering which buttons were connected rather than into play. Fixed with
  `CREnv.action_masks()` + `sb3-contrib`'s `MaskablePPO`; measured illegal rate afterwards
  is **0%**. `test_action_masks.py` verifies the mask against the engine on all 2880 actions
  (0 mismatches) - run it after touching either the mask or `deploy_card`, since they encode
  the same rules in two places and will drift. Details in the 2026-08-10 section below.

## The three overnight runs (2026-08-09/10, all 1,001,472 steps, ~2,800 games each)

All isolated against the same fresh self-play baseline for a clean, attributable
comparison - deliberately not stacked together.

1. **`overnight_1`** - pure baseline. `eval/mean_reward_vs_random` final: 4.23 (smoothed
   12.25) - clearly beats random, healthy-looking curves.
2. **`overnight_2_shaped`** - baseline + reward shaping: leak weight bumped 0.0012->0.003,
   spell whiff penalty -1.0/hit bonus +0.3 per entity (via new `battle.spell_impact_log`,
   populated in `Projectile._deal_splash_damage`). **Concerning result**: `ep_rew_mean`
   sat around -15 to -20 the whole run, and `eval/mean_reward_vs_random`'s smoothed trend
   ended *negative* (-6.17) - doesn't clearly beat random by the end, unlike the other two.
   Suspect the whiff penalty (possibly compounded by the known multi-wave-spell bug below)
   taught something that doesn't generalize to beating even a random opponent. Not yet
   root-caused - do that before reusing this reward shape.
   - **Known bug in this reward, unfixed**: `spell_impact_log` gets one entry per
     projectile *wave*, and Arrows fires 3 waves vs Fireball's 1 - so a fully-whiffed
     Arrows costs -3.0 vs Fireball's -1.0, an unintended 3x asymmetry from an
     implementation detail, not a design choice.
3. **`overnight_3_jointaction`** - baseline reward (unshaped, for isolation) + the flat
   joint action space. **Best result of the three**: `eval/mean_reward_vs_random` final
   16.94 (smoothed 14.98), clearly ahead of both others. Real, if early, evidence the
   joint-action fix helps, not just "safe."

**READ THIS BEFORE TRUSTING ANY NUMBER IN THIS SECTION (added 2026-08-10).** Every
`eval/mean_reward_vs_random` figure quoted for these runs came from an evaluation that
played **5 games** and reported the mean. For a game this variable, 5 games is noise — the
observed swings (e.g. `parallel_real1` moving between -32 and +5 across eval points) are
consistent with a policy that isn't changing at all. **No comparison between these runs is
statistically supportable**, including "overnight_3 was the best" and "reward shaping hurt".
Two further problems with that metric: total *reward* embeds whatever shaping terms were
active, so runs with different reward designs were never comparable by it in the first
place; and it says nothing about whether the policy could even act (see the 91% illegal-
action measurement below). Fixed in `eval_diagnostics.py` — 30 games, win rate plus behavior
diagnostics, written to `runs/<name>/eval_log.csv`. Re-run before drawing conclusions.

Across all three: `train/explained_variance` was near zero (some readings slightly
negative) in every diagnostic checked so far - only spot-checked early iterations, not the
full 1M-step curves. **Worth verifying properly before trusting these runs' value
estimates at all** - near-zero means the critic (the part of PPO judging "how good is this
situation") may not be learning much beyond predicting the average reward regardless of
board state, which would make every other diagnostic here less trustworthy than it looks.

Playing against any of the three checkpoints yourself, they're indistinguishable to a
human eye - expected at ~2,800 games, not evidence something's broken. "Beats random" and
"looks like competent play to a person" are very different bars; only the first has been
cleared.

## 2026-08-10: action masking, honest evaluation, and what that invalidates

Three defects were found and fixed in one pass. All three mean **earlier run comparisons
cannot be trusted** - not "were probably noisy", cannot be trusted.

1. **~91% of all deploy actions were illegal and silently discarded** (measured, not
   estimated). The flat `Discrete(2880)` space had no legality filter, so the policy spent
   training discovering which buttons were even connected. Fixed: `CREnv.action_masks()`
   + `sb3-contrib`'s `MaskablePPO`. Verified against the engine on all 2880 actions across
   multiple mid-game states by `test_action_masks.py` - 0 mismatches. Post-fix illegal rate
   measured at **0%**.
2. **The random baseline was crippled the same way.** `random_strategy` also drew uniformly
   from 2880, so ~91% of ITS moves did nothing - "beats random" meant "beats an opponent
   that mostly stands still". Fixed via `masked_random_opponent`. **Win rates measured after
   this change are not comparable to any measured before it**; they will be lower and they
   mean more. The self-play opponent is masked too, so self-play no longer trains against a
   crippled copy of itself.
3. **Evaluation used 5 games and reported only mean total reward.** Far too small for a game
   this variable, and total reward embeds whatever shaping is active so it was never
   comparable across reward designs anyway. Replaced by `eval_diagnostics.py`: 30 games,
   win rate + margin + behavior diagnostics (`illegal_rate`, `noop_rate`,
   `elixir_capped_rate`), written to `runs/<name>/eval_log.csv`. `analyze_run.py` prints a
   report and explicitly flags when a change is **within noise** rather than letting a trend
   be read into it.

Also fixed: the spell whiff/hit shaping charged multi-wave spells once per wave (whiffed
Arrows cost 3x a whiffed Fireball - an implementation artifact). Now normalized per cast.
**Those terms are set to 0.0 (disabled) by default** - they were invented mid-session, were
present in the only run that failed to beat random, and were still silently active in
`parallel_real1` while this document claimed they had been reverted. Re-enable only as an
isolated experiment.

Practical note: `train.py` writes `runs/` relative to the directory it is launched from, so
runs can end up in more than one place. `analyze_run.py` searches the likely locations.

## Where things stand right now

In dependency order — each item below assumes the ones above it are true.

1. **Reward v2 is implemented and live**, not just designed (the "in progress" framing
   below is about two *specific unbuilt ideas*, not the whole reward — don't be misled by
   it). Current `environment.py` reward = tower/crown HP deltas + win/loss (baseline, see
   "three overnight runs" below) **plus**: a discounted potential-based term over
   elixir+troop-value advantage (`resource_potential()`, phase-staircased by elixir speed,
   proper `reward += GAMMA*Phi(next) - Phi(current)` form so it can't be reward-hacked by
   oscillating), and an elixir-overflow penalty (`-0.05` once elixir ≥ 9.9). Exact constants
   are in `environment.py` lines ~37-56 — read them there, not from memory, they may have
   moved.
   - **CORRECTION (2026-08-10) — this doc previously claimed the opposite.** An earlier pass
     stated that `overnight_2_shaped`'s spell whiff/hit shaping was "not part of this" and
     "never re-adopted". **That was wrong.** Only the leak-weight bump was reverted.
     `SPELL_WHIFF_PENALTY = -1.0` and `SPELL_HIT_BONUS = 0.3` are still live in
     `environment.py`'s `step()`, including the known 3x-Arrows asymmetry bug. So
     `parallel_real1` was carrying the exact shaping suspected of causing
     `overnight_2_shaped`'s regression. It was never a clean test of "reward v2", and its
     poor result cannot be attributed to the resource-potential term.
2. **Joint action space is validated and is the one to keep using** — `overnight_3` beat
   both the baseline and the shaped-reward run. Old `MultiDiscrete` checkpoints and new flat
   `Discrete` checkpoints both still load correctly in `play_vs_ai.py` (auto-detected).
3. **Parallel collection (`SubprocVecEnv`, `--n-envs`) is built and validated** — real ~2x
   throughput on the actual Windows target, not just in theory. Two real bugs were hit and
   fixed getting there (see Errors section below) — if parallel training breaks again on a
   *different* machine, check those two first before assuming a new bug.
4. **`parallel_real1`** is the first run combining all three of the above at real scale (1M
   steps, 6 envs, CUDA, on the Windows RTX 4080 SUPER machine). Its outcome was not yet
   confirmed as of the last working session — check `runs/parallel_real1/manifest.json` on
   that machine (`Get-Content runs\parallel_real1\manifest.json` in PowerShell) and compare
   its `eval/mean_reward_vs_random` against the three below in TensorBoard before drawing
   any conclusion about whether reward v2 actually helped.
5. **Two things are still genuinely open, not yet started:**
   - `explained_variance` near zero in every run so far, unresolved — see below, this
     undermines trust in every other number until it's investigated.
   - The two reward ideas under "Reward design v2" below (elixir-efficiency-via-deaths,
     successful-defense) are proposals only, not implemented — don't assume they're in
     `parallel_real1`'s reward, they aren't.
6. **Separately, on the data-collection/calibration side** (not training, but feeds into
   how trustworthy the simulator itself is): a memory-reader-based real-match capture tool
   was investigated (`cr-memory-reader`, github.com/Jason-XII/cr-memory-reader — Frida-based,
   ARM64-Mac only) and a real, defensible finding was produced from it —
   `telemetry/target_switch_analyzer.py` shows MiniPekka redirects to a still-alive new
   target 1.71×/instance vs Knight's 1.00× and Giant's 0.44× (lowest, consistent with
   Giant's known building-only targeting) — real evidence some troops are more
   "distractible" than others, not yet reflected in the simulator's targeting logic.
   `CALIBRATION_PLAYBOOK.md` has the reusable method for adding more findings like this.
   - **Correction to an earlier note in this doc**: the capture tool's target package is
     `nullsroyale.rel.free` (confirmed by reading `run_raw_capture.py`) — a private-server
     client, not Supercell's live service. It never touches real matchmaking or another
     player's real account, so the "ToS risk from scraping other people's live matches"
     concern flagged in an earlier pass of this doc doesn't apply the way it was written.
     What's still an open, undecided question: running an unofficial private server at all,
     and (once built, see below) automating input against it - lower-stakes than the
     original framing, but still worth a deliberate decision, not silent drift.
   - **File-writing gap is fixed**: `run_raw_capture.py`'s `mainloop()` now takes an
     `output_path` arg and appends every raw event as one JSON line (same schema
     `calibrate.py`/`target_switch_analyzer.py` already parse) - run it as `python
     run_raw_capture.py out.jsonl`. Previously captures only lived in memory and vanished on
     exit.
   - **New: scenario deploy automation, started but not finished** -
     `deploy/scenario_deploy.py` (in the `cr-memory-reader` checkout, currently at
     `~/Downloads/cr-memory-reader-main`, not yet a git remote) automates
     `CALIBRATION_PLAYBOOK.md` step 5 ("run the same script 5-10 times") via ADB touch
     injection instead of doing it by hand. What it already has: `ArenaGrid` (tile→pixel
     using the same 4-corner calibration `queue_overlay/run_calibration.sh` already
     produces, general enough to survive a "Flip Arena 180°" recalibration), and
     `CardIdResolver`, which learns the hand's `data_id_40`→card-name mapping *empirically*
     at runtime (cross-referencing the entity that appears after a deploy, via
     `calibrate.py`'s existing `ID_TO_NAME`) instead of a hand-guessed table. **Two real
     gaps, explicitly not guessed at, both need a live emulator session to close**:
     `HAND_SLOT_PIXELS` (screen position of the 4 hand-card icons - `arena_grid_profile.json`
     only calibrated the arena, not the hand) is still empty and will `KeyError` rather than
     tap a made-up spot; and the tap-to-select-card-then-tap-to-place-tile sequence/timing in
     `deploy_card()` is standard CR touch UX but has never been run against the real client -
     watch the first few runs before trusting an unattended batch.
   - **Windows path is no longer a dead end, but it's not solved either**: upstream added an
     `x86_64/` folder on 2026-08-09 - a real, working-in-principle prototype for Windows, but
     the author's own words describe it as an ugly, unfinished, LLM-written first draft
     ("我实在没有精力去手动修改了" - "I don't have the energy to fix it by hand"), not
     something to trust as-is. It doesn't use Frida at all (that's fundamentally broken on
     Windows' x86_64-translated emulator, confirmed independently by the author, not just
     this project's own earlier finding) - instead a compiled native helper does raw memory
     scanning, driven via the **MuMu emulator specifically**, not Android Studio's. Output
     schema uses different field names for the same concepts (`card_id` vs. `card_id_ac`,
     `own_elixir` vs. `own_elixir_1e0`, `battle_clock` vs. `battle_clock_220`) - a small
     `calibrate.py` adapter would unify them, not a rewrite. If someone wants a concrete,
     bounded, verifiable task (e.g. for the credit-holder mentioned elsewhere in project
     chat) - hardening this prototype so it reliably reads correct state over a full match on
     the Windows training machine itself is a strong candidate: bounded scope, a real
     starting point already exists, and "does it read the right numbers" is checkable without
     deep CR domain knowledge (compare against `minimal_visualizer.py`, already provided in
     that same `x86_64/` folder).

## Reward design v2 - two specific ideas still not finished, don't jump ahead of this

Scope note: the resource-potential term and overflow penalty described above are already
built (see item 1 above) — what follows is about two *additional* ideas that went through
the framework below but were never implemented. Explicit user decision behind the
framework: stop making incremental reward tweaks, design it properly first. Working
framework, apply to every candidate term before implementing:

1. **Observable** - computable from state already tracked, or does it need new
   instrumentation?
2. **Attributable** - can we cleanly say what *caused* the event, or are we guessing?
3. **Gameable** - if the agent maximized *only* this term, would the resulting play look
   like real skill, or a weird exploit?
4. **Correctly scaled** - does it drown out the terms already trusted (tower damage,
   crowns, win/loss), or stay in proportion? (Current scale, for reference: full princess
   tower destroyed ≈ 1.4-4.2 depending on run; one crown = 5; winning = 10, once.)
5. **Precedented or novel** - a known-working idea, or untested?

**Reviewed `wty-yy/KataCR`** (github.com/wty-yy/KataCR) for a real external data point -
important caveat, it's architecturally unrelated (offline RL from human replay video via
computer vision, not self-play PPO on a hand-built sim like ours), so not a direct
comparison, but its actual reward design is real precedent:
- Tower destroyed: ±1 (±3 for King Tower). Tower damage while alive: `(old_hp-new_hp) /
  full_hp` - a *fraction* of max HP, not raw HP points (scale-invariant, unlike our raw-HP
  weights).
- King Tower "exposure" (all other towers down) after all others fall: small -/+0.1,
  specifically to avoid over-rewarding pure aggression.
- **Elixir at max (10) for 10+ frames: -0.05 penalty.** Real precedent for the
  elixir-efficiency gap identified below - notably *simpler* than what was being
  considered (a cap-penalty, not full trade-value attribution).
- Notably absent even there: no "good trade" or troop-vs-troop attribution signal at all.

**Two ideas actively being worked through, not yet built:**
- **Elixir efficiency**: full version (attribute which unit's attack killed which unit,
  compare elixir cost) fails the Attributable test - kill attribution isn't tracked at all
  currently, only HP deltas. Simpler candidate that might get most of the value without
  needing attribution: each tick, `reward += k * (elixir_value_of_enemy_units_that_died -
  elixir_value_of_own_units_that_died)` - only needs death events + each unit's known
  elixir cost, no "who killed whom" tracking. Passes the Gameable check on paper (can't
  fake a unit dying) - not yet stress-tested against real play.
- **"Successful defense"**: current working hypothesis is this might not need a dedicated
  mechanism at all - it may just be what the elixir-efficiency term above and the existing
  leak-penalty produce *together* (enemy investment dying + your tower staying healthy),
  not a third signal to invent. Needs the user's CR judgment to confirm or override this,
  not something to decide unilaterally.

**Process going forward, explicit user request**: go through every reward term - existing
and proposed - against the 5-question framework above, together, before implementing
"reward v2" as one deliberate, fully-documented pass. Not more one-at-a-time tweaks.

## Where things live

- `src/clasher_new/environment.py` - env, action space, reward (see above)
- `src/clasher_new/train.py` - training script, self-play, run tracking
- `src/clasher_new/play_vs_ai.py` - human-vs-checkpoint pygame tool
- `runs/` - one folder per training run (gitignored except `manifest.json`, which is kept
  - small, human-readable record of what produced each run)
- `ROADMAP.md` - longer-horizon: when a vectorized (JAX/CUDA) engine rewrite would
  actually make sense (it doesn't yet - Phase 1/2 not exhausted), CPU-core parallelism as
  the next real throughput lever before that
- `CALIBRATION_PLAYBOOK.md` - reusable method for turning real-match captures into
  calibration findings (one unknown at a time, minimal isolating scenario, multiple runs).
  Apple-Silicon-Mac-only, see its section 0 for why.
- `telemetry/target_switch_analyzer.py` - real finding derived via that method: which
  troops redirect to a new target while their original target is still alive, filtered to
  exclude the trivial "target just died" case. Run it against more `match_*.jsonl` captures
  to extend the finding, or use it as a template for a similarly-scoped analyzer on a
  different question (e.g. Giant's nonzero rate hints at a building-vs-building switch case
  not yet isolated from troop-target switches - a good next one-unknown question).
- `cr-memory-reader` (external, at `~/Downloads/cr-memory-reader-main` on this machine,
  real repo: github.com/Jason-XII/cr-memory-reader) - the capture tool `target_switch_
  analyzer.py` depends on. Confirmed to have **no JSONL file-writing at all** currently
  (`deploy/run_raw_capture.py` only holds captures in memory + optional live visualizer) -
  a small, scoped fix, not yet built. Also: likely a ToS violation to run against other
  players' matches at any real scale - flagged, not resolved, decide deliberately before
  scaling this up with collaborators.
- Remote training machine: Windows, RTX 4080 SUPER, conda env `Arron`, project at
  `C:\Users\Administrator\Desktop\Arron\clash-royale-simulator-portable` - a portable zip
  export, not a git clone (no reliable network access to GitHub from that machine assumed)
- `requirements-train.txt` - minimal deps for training only (no fastapi/uvicorn/starlette,
  those are simulator-web-app-only)
