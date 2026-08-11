# Simulator Status — Handoff Summary

Scope of this whole effort: an 8-card mirror-matchup simulator (Knight, Giant, Musketeer,
MiniPekka, Minions, Archer, Fireball, Arrows) accurate enough to train an RL agent on.
This doc is the condensed version of a very long calibration session — **read this instead
of re-deriving everything from scratch, and paste it into a new chat (with a human
collaborator or another AI model) as the starting context.**

## Bottom line

This doc now covers **simulator/engine work only** — training (reward design, PPO, action
space, actual training runs) moved to a separate session, tracked in `TRAINING.md` and
`ROADMAP.md`. Read those separately if you need that side of the picture.

**Heads up, read before assuming anything is fine**: training was started anyway despite
the open issue below never being resolved (time pressure, not a considered call that it
didn't matter) - three real overnight runs have already happened on top of an engine whose
troop-vs-troop interaction quality was never actually verified after the last round of
placement-bug fixes. That's a real gap, not a solved problem quietly carried forward. If
you're picking up simulator work now, the honest starting point is: the "disastrous
interactions" report below is still exactly as unresolved as it was when first flagged.

## What's validated (confirmed against real data, not assumed)

- **Speed**: Knight/Musketeer/Archer 1.17 tiles/s, Giant 1.0, Minions/MiniPekka 1.75 —
  all corrected from stale defaults, confirmed across 6+ matches.
- **Tower range**: King 7.0 tiles, Princess 7.5 tiles - the original values. An earlier
  pass had "corrected" both to 8.5 based on a calibration bug (`calibrate.py` never
  actually distinguished which physical tower fired a given hit, so Princess Tower hits
  could get mislabeled as King Tower hits). Caught via a user-reported gameplay
  observation (King Tower can't reach a Princess Tower under attack) and confirmed by
  re-running the attribution with hits grouped by each tower's own recorded position
  (`telemetry/verify_king_range.py`): correctly-identified King Tower hits are
  sparse/noisy (n=4, no tight cluster), Princess Tower hits cluster tightly near 7.0-7.5,
  not 8.5. **Lesson for next session: this calibration session's "measured" constants have
  been wrong before in a way that looked confident (tight cluster, clean writeup) but
  wasn't - re-verify claims like this against raw telemetry before trusting them, don't
  just trust the code comments.**
- **Troop-vs-troop damage**: Knight 202, Minions 107, Musketeer 217, Archer 112,
  MiniPekka 755 - all exact matches to real data (match_11's isolated-duel capture).
- **Spell mechanics**: Fireball/Arrows knockback (was completely unimplemented), weighted
  by target mass (Giant ~40% of normal knockback, measured), spell travel time scaling
  with distance from the caster's own King Tower, reduced spell damage against towers -
  all implemented and verified.
- **Core AI rules**: Giant is building-only (verified - ignores troops, never attacks
  them), ground troops fully ignore air units except Musketeer/Archer, the "chase a
  fleeing target" hit-rate reduction (verified: 17 hits/30s vs 25 theoretical max when
  chasing, driven by real catch-up gaps, not a slowed steady-state rate).
- **Pathfinding**: real A* replacing the original angle-scan hack, treats buildings as
  dynamic obstacles, excludes the entity's own current target so it can path all the way
  to attack range. Passes an automated stuck-troop sweep (`fuzz_pathing.py`).
- **Match rules**: deploy-zone boundaries after a tower falls (confirmed via real
  boundary-testing play), sudden-death tiebreak at 300s (was silently defaulting every
  tie to player 1 - fixed to represent real draws).

## Fixed since the last handoff: King Tower / Projectile crash

`take_damage()`'s King Tower activation check fired on *any* entity named `KingTower`,
including the King Tower's own projectile (which carries that name for damage
attribution, but isn't a `Building` and has no `tower_active` attribute) - crashed with
`AttributeError: 'Projectile' object has no attribute 'tower_active'` the moment a King
Tower actually fired. Only surfaced under real training load (a Fireball smoke test), not
during earlier interactive playtesting - worth remembering that some bugs only show up
once you're running enough matches, not just playing a few by hand. Fixed by guarding both
call sites (`take_damage`, and the analogous check in `die()`) with
`isinstance(entity, Building)`.

## Fixed since the last handoff: "I still can't place troops"

Re-tested with literal simulated mouse clicks (not JS shortcuts) and found three real bugs,
all now fixed in `web/static/index.html`:

1. Clicking the arena with no hand card selected was a **silent no-op** - indistinguishable
   from the app being broken. Now shows "Tap a card in your hand first...".
2. On a normal desktop browser window (1280x720), the canvas's fixed 396x704 size pushed the
   **hand bar below the fold** - a player would never see their own cards without scrolling
   mid-match, and likely wouldn't think to. Canvas is now responsive (`max-height:56vh`,
   width auto-scales), so the whole UI (canvas, elixir, time, hand, status) fits in a normal
   window with no scrolling.
3. That responsive resize exposed a latent bug: the deploy click handler read raw CSS pixel
   offsets without correcting for the canvas's CSS-vs-intrinsic-resolution ratio, so once the
   canvas was no longer rendered at exactly 396x704 CSS px, click positions would have mapped
   to the wrong tile. Fixed to scale through `canvas.width/rect.width` properly.

Verified end-to-end with real coordinate-based clicks (not `dispatchEvent`/`.click()`
shortcuts): select a hand card → click the arena → troop deploys, elixir drops, status
clears. This should be solid now, but this is exactly the kind of thing that looked fine
under scripted testing before too - if it's still broken for a real user, get **specific**
repro details (browser/window size, exact click sequence) rather than assuming it's fixed.

## Open issue (start here next session)

User playtested the web app and reported: **"I can click and drop any troop and the
interactions are disastrous."** Not yet diagnosed - no specific repro steps captured yet,
this was flagged and the session was intentionally paused rather than chased blind. This is
about troop-vs-troop *combat behavior* once deployed, separate from the placement-click bugs
above. Likely candidates worth checking first, roughly in order of suspicion:

1. **Collision/separation between troops** - do same-side or opposing troops overlap,
   jitter, or push each other in obviously wrong ways when several are near each other?
   (`battle.py`, search for the pairwise overlap-resolution code near
   `movement_ratio`/`direction_vector`.)
2. **Target acquisition/switching** - do troops flip targets erratically, attack through
   walls of other troops, or ignore the closest valid target?
3. **Deploy-then-immediately-fight** - does a freshly-deployed troop behave correctly
   during its `deploy_delay_remaining` window, or does it start interacting before it
   should be able to?
4. Anything specific to the *web app's* rendering vs the actual sim state - i.e. is this a
   real engine bug, or does it just *look* wrong because of a rendering issue (stale
   position interpolation, entities popping instead of moving smoothly at 30Hz)? Worth
   ruling this out early since it's cheap to check and would mean the engine itself is
   fine.

Recommended first step for whoever picks this up: play one full match in the web app
(`http://localhost:8000`, 2 tabs or vs the bot fallback) while watching closely, and write
down 2-3 *concrete* moments that look wrong (which cards, roughly when, what happened) -
right now there's a strong reaction but no reproducible description to debug against.

## Known secondary gap

Full-match trajectory replay (`telemetry/replay.py`) shows growing position error over
time - tight early (~0.5 tiles in the first 5s after a deploy), 1-2+ tiles by mid-game.
This may turn out to be the *same* root cause as the "disastrous interactions" report
above (both point at decision-level/interaction bugs, not stat-level - every stat-level
hypothesis has already been checked and ruled out or fixed). Worth investigating together
rather than as two separate problems.

## Where things live

- `src/clasher_new/battle.py` - core engine (targeting, movement, combat, pathfinding)
- `src/clasher_new/card_utils.py` - card stats, including the sourced override tables for
  every corrected constant (search `_MEASURED_` and `# measured`)
- `src/clasher_new/pathfinding.py` - the A* implementation
- `src/clasher_new/environment.py` - the Gymnasium `CREnv` used for training
- `src/clasher_new/train.py` - PPO + self-play training script (already built, untouched
  this session except for what it inherits automatically from the engine fixes)
- `telemetry/` - all captured real matches (`match_1.jsonl` - `match_11.jsonl`),
  `calibrate.py` (constant validation - **has a known attribution bug, see above**),
  `verify_king_range.py` (the corrected re-analysis), `replay.py` (full-match trajectory
  comparison), `fuzz_pathing.py` (automated stuck-troop detector)
- `web/` - browser-playable host (FastAPI + WebSocket backend, canvas frontend), no
  terminal needed to play. Bot fallback after 5s if nobody else joins. This is where the
  "disastrous interactions" report came from.

## How to try it

- **Browser** (easiest, shareable, no terminal needed once running):
  ```
  cd web && source ../.venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port 8000
  ```
  then open `http://localhost:8000`. Pick exactly 8 cards (the picker is restricted to the
  validated pool), click "Find Match" - a scripted bot opponent joins after 5s if nobody
  else does. Click a hand card, then click the arena to deploy it.
- **Desktop, vs the old checkpoint**: `python play_vs_ai.py cr_checkpoint` (note: that
  checkpoint was trained on the pre-fix engine, so its play style reflects old physics -
  don't use it to judge current sim quality).
- **Desktop, vs a friend on LAN**: `server.py` + `client.py`, unchanged from before.

## Sharing this with others (a person, or another AI session)

This folder (`clash-royale-simulator-trial/`) is already cloned from your own repo
(`origin` = `https://github.com/weAllHateMicrosoft/clash-royale-simulator-trial.git`) - the
older `clash-royale-simulator/` folder (still pointed at the original upstream author's
repo, no push access) is now superseded, work from this one going forward. To publish a
checkpoint:

```
git add -A
git commit -m "Checkpoint: calibrated 8-card sim, web playtest app, open interaction bug"
git push -u origin main
```

(Review `git status` before the `git add -A` - `.venv/` is a symlink to the original
folder's virtualenv for convenience; if you want this repo fully standalone, delete that
symlink and create a real venv here instead before sharing, since a symlink pointing
outside the repo won't resolve on someone else's machine. `__pycache__/` and `.DS_Store`
are gitignored and should NOT show up.)

Share the repo URL with your collaborator, or paste it (plus this file's contents) into a
new chat with another AI model as the starting context - this doc is written to be
self-sufficient for that.

## Training (moved out of this doc)

Reward design, PPO setup, action-space design, and actual training runs are tracked
separately now - see `TRAINING.md` (session-level status, the three overnight runs and
their results, the reward-design framework) and `ROADMAP.md` (when/how a vectorized
engine rewrite might make sense, longer-horizon planning). This doc stays scoped to the
simulator/engine itself.
