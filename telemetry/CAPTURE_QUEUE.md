# Capture Queue — what to actually play, in order

A ready-to-execute worklist, one scenario per row, so you can just play without deciding
anything on the spot. Method/background is in `CALIBRATION_PLAYBOOK.md` - this is that
method already applied to specific cards, not a new approach.

**Why these cards specifically**: `src/clasher_new/card_mechanics.py` has bespoke logic for
Witch, Balloon, Golem, LavaHound, Prince, DarkPrince, BattleRam, GiantSkeleton, IceWizard,
Miner, Rage, RageBarbarian - real, already-implemented mechanics that could easily be
mistimed or mis-scaled, unlike the ~30 other cards in the pool that just use `gamedata.json`
numbers directly through generic troop behavior (nothing bespoke to verify there yet - skip
those for now). This list only covers cards with actual code to check against.

**Every scenario below: 5 runs minimum, one enemy troop/target only, don't touch anything
else, don't deviate mid-run.** Save each as `telemetry/raw/<scenario_name>_<01-05>.jsonl`
(create the `raw/` folder if it doesn't exist) using `python run_raw_capture.py
telemetry/raw/<name>_01.jsonl` from the `cr-memory-reader` checkout. Write one
`<scenario_name>_notes.md` per scenario (template at the bottom) - that note is what makes
the capture usable by someone who wasn't there when you recorded it.

## 1. `golem_death_split`
**Question**: how long after Golem dies do the two Golemites appear, and where (exact
offset from the Golem's death position, not "nearby")?
**Script**: Deploy a single Golem at (9, 24) with nothing else on the field. Let it walk to
the enemy tower and get destroyed by tower fire alone (don't deploy anything else to help or
hinder). Keep recording 10s past the Golem's death.

## 2. `lavahound_death_split`
Same question/script as above, substituting LavaHound for Golem - it's a separate code path
(`card_mechanics.py`), don't assume the timing matches Golem's.

## 3. `balloon_death_bomb`
**Question**: does Balloon's death bomb correctly measure damage/radius against a nearby
troop, and does it apply the crown-tower-damage-percent reduction correctly against a tower?
**Script**: Run twice - once with a single low-HP enemy troop (e.g. Skeletons) standing near
where the Balloon will die, once with the Balloon dying directly next to the enemy tower with
no other troops present. 5 runs each (10 total).

## 4. `giantskeleton_death_bomb`
Same shape as `balloon_death_bomb`, GiantSkeleton instead - separate implementation, don't
assume shared code actually behaves identically.

## 5. `prince_charge_trigger`
**Question**: at what distance traveled does Prince's charge actually activate (damage/speed
multiplier kicks in), and does it turn off if he changes target before then?
**Script**: Deploy Prince far from any target (back of your own half) so he has a long
uninterrupted walk before reaching the enemy tower. Also do one run where you deploy a single
weak distraction troop partway through his walk, to see if the charge cancels/resets when he
retargets to it.

## 6. `battleram_charge` and `darkprince_charge`
Same question as `prince_charge_trigger`, run separately for each - `battle.py` shows
`DarkPrince`/`BattleRam` both subclass `Prince` in `card_mechanics.py`, but that's a code
relationship, not proof their actual behavior matches - confirm both independently.

## 7. `icewizard_slow_aoe`
**Question**: on deploy, what's the actual radius and duration of Ice Wizard's slow, and does
it also apply on every subsequent hit or only the deploy pulse? (This is the same category of
question as the earlier Ice Wizard example in `CALIBRATION_PLAYBOOK.md` §1 - if that one was
already answered elsewhere, skip this and just point at that result instead of re-running it.)
**Script**: Deploy Ice Wizard, then a single enemy Knight walking toward it from just outside
plausible slow range, so first contact is clearly visible in the position deltas.

## 8. `rage_buff` and `ragebarbarian_buff`
**Question**: buff radius, the actual speed/attack-speed multiplier applied, and duration.
**Script**: Deploy 2-3 of your own troops together, then Rage centered on them. Track their
speed via position deltas before/during/after the buff window (same method as
`calibrate.py`'s `measure_speed`).

## 9. `witch_skeleton_summon`
**Question**: summon interval, skeleton count per summon, spawn radius around the Witch.
**Script**: Deploy a lone Witch with no target nearby (so she doesn't move/die early) and just
record 20+ seconds of continuous skeleton summons.

---

## Notes template (`<scenario_name>_notes.md`)

```
Question: <the exact one-sentence question from above>
Script: <exact steps actually followed, copy from above or note any deviation and why>
Runs: <N>, filenames <scenario_name>_01.jsonl .. _0N.jsonl
Anything inconsistent between runs: <don't smooth this over, report it plainly>
```

## For whoever analyzes these later (may not be you)

Don't derive the calibrated value by eyeballing the jsonl. Follow `calibrate.py`'s existing
pattern (`load_match`, then a `measure_*` function per question) or
`target_switch_analyzer.py`'s pattern for anything involving repeated events across a match -
both already handle the raw event format these captures use.
