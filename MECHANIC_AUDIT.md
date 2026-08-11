# Mechanic Audit — code vs. documented behavior

Method (see chat for the reasoning): numeric stats and documented mechanics don't need real
captures - they're settled by `gamedata.json` (already the source of truth for base stats)
plus wiki/recall for behavior. Captures are reserved only for what's left after this - things
neither data nor documentation resolve. Scope: the 9 mechanic clusters in
`src/clasher_new/card_mechanics.py` - every other card in the pool is a plain
`BasicCharacter` with nothing bespoke to audit.

Confidence key: **DATA** = confirmed directly against `gamedata.json`, no guessing.
**SEARCH** = confirmed via a wiki search this session, treat as good-not-perfect.
**UNKNOWN** = neither settles it - this is the genuinely capture-worthy list.

## Confirmed bugs / gaps (real fixes, not calibration)

1. **Golem/Golemite death splash+knockback: not implemented at all.**
   [card_mechanics.py:32-40](src/clasher_new/card_mechanics.py:32) only spawns the two
   Golemites on Golem's death - no area damage, no knockback, for either Golem's own death or
   a Golemite's. **DATA**: `gamedata.json` already has `Golem.deathDamage: 88`,
   `Golemite.deathDamage: 39` sitting unused. Radius/knockback distance: not present in
   `gamedata.json` for these - not guessing, see the unknown list below.

2. **DarkPrince's charge has no area damage.**
   [card_mechanics.py:80-81](src/clasher_new/card_mechanics.py:80) is `class
   DarkPrince(Prince): pass` - pure single-target inheritance. **DATA**:
   `gamedata.json`'s DarkPrince has `areaDamageRadius: 1100` (1.1 tiles) that nothing reads.
   This is a real, data-backed, immediately-fixable gap - the number already exists, just
   needs wiring into an overridden `on_attack`.

3. **Charge isn't interrupted by stun/knockback**, only by landing a hit. Confirmed by
   reading all of `Prince.on_tick()`/`on_attack()` - nothing there reads a stun/debuff state.
   Shared by Prince/DarkPrince/BattleRam (DarkPrince/BattleRam both subclass Prince).

4. **Miner's "burrow travel time" doesn't match real Miner.**
   [card_mechanics.py:112-126](src/clasher_new/card_mechanics.py:112): `freeze_time =
   distance_to_enemy_king_tower / (650/60)`, i.e. invincible/untargetable for longer the
   further from the enemy king tower it's deployed. **SEARCH** (Miner wiki + others,
   consistent across sources): real Miner has a flat ~1 second deploy time regardless of
   where it's placed - `gamedata.json` agrees (`deployTime: 1000`, same as any normal troop,
   no distance-dependent field anywhere in Miner's data). The sim's distance-based freeze
   looks like an invented mechanic, not a real one - recommend replacing `freeze_time` with
   the flat `deploy_time`, dropping the distance calculation entirely.
   - Bonus, currently unimplemented, low-priority: real Miner burrowing under standing
     troops pushes them aside and can re-aggro them onto a new target. Not in the sim at all,
     probably not worth the effort relative to the above three.

5. **`TimedExplosive` (Balloon's and Giant Skeleton's death bombs) never applies knockback**,
   even though the mechanism to do it already exists and is used elsewhere -
   `Projectile._deal_splash_damage()` at
   [battle.py:566-573](src/clasher_new/battle.py:566) already reads `proj.pushback` and
   `entity.data.knockback_multiplier` and pushes hit entities back. `TimedExplosive.update()`
   just never calls into that same logic. This is a smaller fix than "invent knockback from
   scratch" - reuse the existing pushback code path. **DATA check**: neither
   `BalloonBomb` nor `GiantSkeletonBomb`'s `deathSpawnCharacterData` in `gamedata.json` has a
   `pushback` field at all (confirmed by direct lookup) - so the magnitude isn't sitting in
   data waiting to be read, unlike the damage numbers. See unknown list.

## Confirmed already correct - don't touch these

- Prince/DarkPrince/BattleRam: switching target does **not** cancel a charge, landing a hit
  **does** - both match [card_mechanics.py:57-78](src/clasher_new/card_mechanics.py:57)
  exactly.
- LavaHound death = spawn LavaPups only, no bomb, no splash - correct as-is.
- Rage's one-time damage-on-cast - `deal_area_damage()` fires in `__init__`, before the
  ongoing buff loop starts. Matches "Rage deals a one-time damage too."
- Ice Wizard's **on-hit** slow (not just the deploy pulse) - turns out this is *already*
  generically implemented, not missing as I first suspected. `gamedata.json`'s
  `IceWizard.projectileData.targetBuffData` (`speedMultiplier: -35`, `hitSpeedMultiplier:
  -35`, `buffTime: 2500`) is read generically by `Projectile`'s hit-handling
  ([battle.py:543](src/clasher_new/battle.py:543) and
  [battle.py:566](src/clasher_new/battle.py:566)), which sets `speed_debuff` +
  `debuff_time_remaining` on every hit, splash or single-target. Only one multiplier
  (`speedMultiplier`) gets applied, not `hitSpeedMultiplier` separately - but since both are
  -35 for this card, the existing generic code already produces the right combined effect by
  coincidence, not by explicitly modeling two separate multipliers. Worth knowing if a future
  card ever has *different* speed vs. hit-speed slow percentages - this generic path would
  under-model that case, but Ice Wizard itself is fine.
- Witch's summon numbers - `spawn_number = 4` / `next_spawn_remaining = 7.0` (hardcoded in
  [card_mechanics.py:8-24](src/clasher_new/card_mechanics.py:5)) match `gamedata.json`'s
  `spawnNumber: 4` / `spawnPauseTime: 7000` exactly. `spawn_radius = 2` is NOT backed by an
  explicit data field (no such key exists for Witch) - lower-confidence, lowest priority of
  anything on this page.
- "Time to first hit after switching target" (the thing you were worried the sim might get
  wrong per-troop) - already generically correct.
  [battle.py:378](src/clasher_new/battle.py:378) floors `attack_cooldown` at
  `hit_speed - load_time` while approaching any target, using each card's own real
  `load_time`/`hit_speed` - Balloon vs. Lava Hound will already differ correctly, driven by
  data, not something missing.

## Genuinely unknown - this is the real capture-worthy list, small on purpose

- **Golem/Golemite death splash radius**, and **knockback distance for Golem/Golemite,
  Balloon bomb, Giant Skeleton bomb** (confirmed larger for Giant Skeleton than Balloon per
  your report) - none of these have a value in `gamedata.json`, and a wiki search isn't
  reliably going to give an exact tile number for knockback distance specifically (unlike
  damage, which is usually well-documented). This is the one place where a
  `CAPTURE_QUEUE.md`-style scenario earns its keep: single enemy troop standing at a few
  different fixed distances from where the death-effect triggers, see where it stops getting
  pushed/damaged.
- Everything else on this page is resolved without a single capture.
