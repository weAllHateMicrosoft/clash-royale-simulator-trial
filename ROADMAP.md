# Roadmap

Where this project is headed past the current prototype stage, and specifically *when* a
vectorized (JAX/CUDA) rewrite of the simulator makes sense - not "someday," a real trigger
condition. Written to be shareable as-is with other people (a Discord server, collaborators,
another AI session) for planning discussion.

## Why this doc exists

The current simulator (`src/clasher_new/battle.py` etc.) is a plain, single-threaded Python
simulation - one match, one entity at a time, ordinary objects and loops. On a real GPU
(RTX 4080 SUPER) during actual training, GPU utilization measured at **5%**, 1.4/16GB memory -
confirming the engine itself, not the GPU, is the throughput bottleneck. Training the neural
net is fast; *simulating one Clash Royale match in Python* is slow, and that's what dominates
wall-clock time during training.

The fix that actually addresses this - rewriting the simulator as batched array operations in
JAX (or CUDA), so a single GPU can advance thousands of matches at once instead of one - is a
real, well-established technique (see: Brax, Gymnax, NVIDIA Isaac Gym, all built this way for
exactly this reason). It is also a full rewrite of the simulation core, not a tweak. This doc
lays out why that shouldn't happen yet, what has to happen first, and what it actually involves
when the time comes.

## Phase 1 (current): validate the design, not the throughput

Open, unvalidated questions right now:
- **Reward shaping** - currently a simple baseline (tower/crown HP deltas + win/loss). Real
  CR-knowledge-driven shaping (spell efficiency, value-weighted kills, leak penalties) hasn't
  been tried yet.
- **Action space design** - card choice and board position are currently sampled
  independently/simultaneously (`MultiDiscrete([5, 32, 18])`), not conditioned on each other.
  Concrete concern raised: placing a Knight vs. a Musketeer in front of a Mega Knight/Pekka
  should get different position values, and the current architecture is structurally weaker at
  representing that than it could be (see git history / prior conversation for the full
  reasoning). Unconfirmed whether this actually shows up as bad behavior in practice - needs
  checking against a real trained checkpoint before deciding it's worth fixing.

This is genuinely *better* done on the current simple engine than on a vectorized one - a bug
here shows up as a wrong number or a Python traceback; the same class of bug in JIT-compiled
JAX code shows up as a silently wrong gradient several layers removed from the cause. This
project has already hit several subtle simulation bugs (tower range miscalibration, targeting
math, King Tower activation) even in the simple version - debugging those in JAX would be
substantially harder. Fast, cheap, debuggable iteration here is deliberate, not a stopgap.

## Phase 2 (next lever, cheap, low-risk): parallelize across CPU cores

Before touching the engine's internals: `stable_baselines3`'s `SubprocVecEnv`, running N copies
of the *exact same, already-validated* simulator across N real CPU cores (separate OS
processes, so Python's GIL isn't a bottleneck). Zero new correctness risk - it's the same code
running more times at once, not new code. Realistic win: 10-20x rollout throughput depending on
core count, for a small, contained code change.

**Do this before Phase 3, always** - it's nearly free compared to a rewrite, and may turn out to
be enough on its own for a while.

## Phase 3: the vectorized rewrite

**Trigger condition** (not a date): Phase 1's reward/action-space design has gone through real
iteration and is trusted, AND Phase 2's CPU parallelism has become the actual limiting factor on
how many experiments/how much training can happen in reasonable time.

Rewriting before that trigger risks spending weeks vectorizing a design that gets thrown out
once real training reveals the reward or action space needs to fundamentally change - and the
observation/action interface is deeply coupled to the engine, so that rework wouldn't be free
either.

**What it actually involves**, concretely, so this stays a real plan and not a hand-wave:
- Every entity becomes a fixed-size, padded array - JAX requires static shapes, no more Python
  objects/dicts per troop.
- Targeting, combat, and movement get rewritten as batched array operations across a
  "which of the N parallel matches" dimension (`vmap`/`jit`), replacing Python loops and `if`
  statements with masks and `jax.lax` control-flow primitives (`cond`, `while_loop`).
- **A\* pathfinding is the hardest single piece to vectorize** - it's inherently
  sequential/branchy. Realistic path forward is a vectorizable substitute (flow-field /
  potential-field navigation, as used in real vectorized crowd-simulation engines), accepting a
  fidelity tradeoff specifically there, rather than a literal vectorized A*.
- The rewritten engine needs its own correctness pass against the same real match telemetry
  already used to calibrate the current engine (see `SIMULATOR_STATUS.md`) - it is effectively
  new code with the same accuracy bar, not a drop-in performance upgrade to trusted code.
- **JAX over hand-written CUDA**: JAX's `jit`/`vmap` are built for exactly this, and there's real
  reference code to learn from (Brax, Gymnax solve near-identical problems). Hand-written CUDA
  is a much heavier, more specialized lift with far less to learn from.

Realistic scope once started: weeks of focused work, not a side quest bolted onto something
else.

## Status log

- 2026-08-09: Phase 1 in progress on a borrowed RTX 4080 SUPER machine (12-core i7-12700KF,
  20 logical processors - confirmed enough headroom to run several single-threaded training
  processes simultaneously without meaningfully slowing any of them down). Three isolated
  overnight runs launched in parallel for a clean, attributable comparison:
  - `overnight_1` - pure baseline (fresh self-play, unmodified reward, `MultiDiscrete` action
    space).
  - `overnight_2_shaped` - baseline + reward shaping (leak-weight bump 0.0012->0.003, spell
    whiff penalty -1.0/hit bonus +0.3 per entity hit, tracked via a new
    `battle.spell_impact_log`). Known limitation: multi-wave spells (Arrows) log once per
    wave, so a whiffed Arrows costs ~3x a whiffed Fireball - not normalized yet.
  - `overnight_3_jointaction` - the card/placement joint-decision fix from Phase 1's open
    question, resolved sooner than planned: action space changed from `MultiDiscrete([5,32,18])`
    to a flat `Discrete(5*32*18)`, so card choice and board position are one joint decision
    instead of three independent simultaneous ones. Uses SB3's standard categorical
    distribution - no custom policy/log-prob code, avoids the correctness risk a full
    autoregressive-head implementation would have carried. Baseline reward, unshaped, to
    isolate this specific change against `overnight_1`.

  Next session: compare all three, watch checkpoints play, decide what's actually working
  before deciding what's next. Phase 3 (vectorized engine) still not started, still pending
  the trigger condition above.

- **Phase 2 built and validated on the real Windows target**: `SubprocVecEnv` parallel
  rollout collection, with each worker process reloading the self-play opponent from a
  checkpoint file (in-memory references don't cross process boundaries) refreshed
  periodically by `WeightsCopyingCallback`. Two real bugs found and fixed via actual runs on
  the target machine, not assumed: `set_env()` can't change env count (must reload via
  `PPO.load()` instead), and loading a same-session checkpoint through the CPU-first
  cross-platform-safe path left the rollout buffer's device out of sync with the
  GPU-moved policy, crashing the first real `train()` call - fixed by loading straight onto
  the target device for this specific same-machine reload (the cross-platform risk that
  motivated the CPU-first path doesn't apply here). Measured result: ~75fps with 6 parallel
  workers vs ~35-40fps single-threaded - real, roughly 2x throughput, not linear (PPO's
  update phase doesn't parallelize), but genuine. `--n-envs` flag on `train.py`.
- Reward v2 (resource-advantage potential: elixir + troop value, phase-staircased, proper
  `gamma*Phi(next)-Phi(current)` form; overflow penalty; reverted leak weight to baseline)
  combined with the joint action space and parallel collection into `parallel_real1`, the
  first run testing all three together at real scale (1M steps, 6 envs, CUDA).
- Still open: `explained_variance` sitting near zero across every run so far (first overnight
  batch through today) - the critic doesn't appear to be learning well, independent of
  whatever the reward or action space design is. Worth investigating directly once
  `parallel_real1` has produced enough data, since it may be limiting everything else.
- `parallel_real1`'s completion/result was not confirmed before this session's access
  ended - check `runs/parallel_real1/manifest.json` on the Windows machine and compare its
  `eval/mean_reward_vs_random` against the three overnight runs (see `TRAINING.md`) before
  drawing any conclusion about whether reward v2 + joint action + parallel collection
  together actually improved things.
- Separately, real-match telemetry work (data collection/calibration side, not training)
  produced one real finding this session: troop-specific "target switch while original
  target still alive" rates, via `telemetry/target_switch_analyzer.py` - see `TRAINING.md`'s
  "Where things stand right now" section 6 for the numbers and what's still open there
  (memory-reader has no file output yet, ToS question undecided).
