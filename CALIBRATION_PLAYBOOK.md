# Card Calibration Playbook

How to add or fix one card's mechanics in the simulator using real captured game data,
in a way that doesn't require the project owner to debug your machine for hours. If you're
picking up a task from here, this doc should be everything you need - if it isn't, that's a
bug in the doc, flag it.

## 0. Before you touch anything: can your machine even do this?

The primary data-collection path (`cr-memory-reader`'s `deploy/` tool) hooks specific memory
offsets via Frida, reverse-engineered from the **ARM64 build** of the Clash Royale APK,
running on an ARM64 Android emulator. That's the natural, fast emulator choice on an
**Apple Silicon Mac** (M1/M2/M3+). On other hardware, Android Studio's emulator normally runs
the x86_64 build instead - a *different compiled binary* with different memory addresses -
and worse, per the upstream author's own investigation, the x86_64 emulator runs the ARM
binary through a translation layer that breaks Frida's hooking mechanism entirely, not just
the offsets. This isn't a "redo the RE work" problem, it's a "Frida fundamentally doesn't
work here" one.

**Apple Silicon Mac + the `deploy/` Frida path is still the reliable, well-tested route.**
There is now a second path for Windows: the upstream repo's `x86_64/` folder (added
2026-08-09), a **rough, author-acknowledged-unfinished prototype** using a different
technique entirely (a compiled native helper doing raw memory scanning, run via the MuMu
Android emulator, not Frida). The author's own description: "ugly, GPT-written, I don't have
the energy to fix it, contributions welcome." Field names differ from the Mac path
(`card_id`/`own_elixir`/`battle_clock` vs. this project's `card_id_ac`/`own_elixir_1e0`/
`battle_clock_220`) - same concepts, needs a small adapter in `calibrate.py`, not a rewrite,
to unify them. Treat it as a real starting point that needs hardening (verify it reads
correct state at all, then verify it's stable over a full match), not a drop-in tool yet -
if you're picking this up, that hardening pass is itself the task, not a blocker to work
around first.

If you're on neither (e.g. Intel Mac with no Windows machine available), stop here rather
than spending hours on a setup that can't work yet - that specific combination still has no
path.

**Self-check before starting real work** - confirm your own setup is producing real data:
1. Get the test client running in an ARM64 Android Studio emulator, `frida-server` deployed
   inside it (see the main `cr-memory-reader` README for the one-time setup).
2. Start `minimal_visualizer.py`, join any match (even a bot match).
3. You should see a live-rendering visualization of both towers and units, updating in real
   time. If the screen stays black once you're in a match, or the tower HP/positions look
   obviously wrong (e.g. frozen, or not matching what you see on screen), your setup isn't
   working - fix that before collecting any calibration data, since bad captures are worse
   than no captures (they look real and quietly corrupt whatever gets calibrated from them).

Only move on once you've personally confirmed step 3 looks right.

## 1. Pick ONE specific unknown, not "calibrate [card]"

"Calibrate Ice Wizard" is too vague to design a test for. Break it into specific, individually
answerable questions first. Example, for a card with a slow-on-hit effect:

- Does the slow apply instantly on the hit, or ramp up over time?
- What's the actual slow percentage (movement speed multiplier)?
- How long does the slow last after the last hit?
- Does it stack if hit multiple times, or just refresh the duration?

Each of those is its own bounded task. Pick one. Write it down as a literal question before
doing anything else - if you can't state exactly what you're trying to find out in one
sentence, the scenario you design next won't be able to answer it cleanly either.

## 2. Design the MINIMAL scenario that isolates just that one question

Strip out everything that isn't needed to answer it. Earlier calibration work in this project
used exactly this principle - e.g. deploying a single troop against a tower with no other
units on the field, specifically to measure damage cleanly without combat noise from other
troops. That looks like "unrealistic" play if you're used to watching real matches - that's
correct and intentional, not a mistake. A real match is far too noisy (multiple units,
targeting decisions, pathfinding) to isolate one specific number from.

For the slow-duration example: deploy the slowing card, deploy exactly one enemy troop where
it will get hit, then deploy nothing else and don't touch anything until the effect has
clearly ended. No other units, no other player actions.

## 3. Write the scenario as an exact, literal script

Precise enough that it doesn't require game-skill or judgment calls to execute - just
following steps. Example:

```
Scenario: ice_wizard_slow_duration
1. Start a friendly/practice match (2-player, both same account is fine).
2. At t=0 (start of recording), deploy Ice Wizard at tile (9, 20).
3. At t=0, deploy a single Knight (enemy side) at tile (9, 8) - far enough that it has to
   walk into Ice Wizard's range, so the moment of first contact is visible in the data.
4. Do not deploy anything else. Do not touch either card again.
5. Let it run until the Knight has clearly returned to normal speed after moving out of
   range, or 30 seconds after first contact, whichever is later.
6. Stop recording.
```

If a step requires "use good judgment" instead of an exact instruction, it's not specific
enough yet - go back and tighten it.

## 4. Name which exact data fields answer the question, before recording

Don't record first and figure out what to look at later - decide up front, so you know while
you're still in the match whether you're capturing the right thing. For the slow-duration
example: track the Knight's `pos_x_7c`/`pos_y_80` deltas between consecutive samples
(30Hz, so consecutive timestamps ~33ms apart) - a real speed reduction shows up as smaller
position deltas per sample immediately following contact, and you're looking for exactly
when those deltas return to the pre-contact baseline.

## 5. Run it multiple times

Real game timing has some noise (network/frame variance). One run isn't enough to trust a
number confidently - run the same exact script 5-10 times before treating the result as
solid. This is mechanical repetition once the script is written, not something that needs
new judgment calls each time - a good candidate for a bot to eventually automate, or for a
collaborator to just grind through manually in the meantime.

## 6. Submit results in a consistent format

For each completed scenario, produce:
- The raw `.jsonl` capture file(s), one per run, named `<scenario_name>_<run_number>.jsonl`
  (e.g. `ice_wizard_slow_duration_03.jsonl`).
- A short `<scenario_name>_notes.md` with: the exact question from step 1, the exact script
  from step 3, how many runs, and anything that looked off/inconsistent between runs (don't
  smooth this over - inconsistency between runs is itself a real, useful finding, not noise
  to hide).

That's everything needed to hand off to someone else (including a future session with me) to
actually derive the calibrated value - the person collecting data doesn't need to also be the
person who does that analysis.

## Common mistakes to avoid

- **Testing two unknowns in one scenario.** If both the slow *and* the splash radius are
  unknown and both affect the same run, you can't cleanly attribute what you observe to
  either one. One unknown per scenario.
- **Deviating from the script mid-run** ("oh let me also try..."). Breaks repeatability -
  finish the script as written, then design a *new* scenario for the new idea.
- **Discarding a "weird" run instead of reporting it.** An inconsistent result is either real
  game behavior you didn't expect (valuable) or a setup problem (also valuable to know) -
  either way, report it rather than quietly dropping it.
