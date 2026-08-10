"""
Verifies CREnv.action_masks() agrees with what BattleState.deploy_card actually accepts,
for every one of the 2880 actions, across a variety of real mid-game states.

Why this test matters more than most: the mask is a *duplicate* of deploy_card's legality
rules, and duplicated rules drift. Drift in one direction (mask allows something the engine
rejects) silently reintroduces exactly the dead-action problem masking was built to remove.
Drift in the other (mask forbids something legal) permanently hides a real move from the
policy - which would be invisible in training curves and nearly impossible to diagnose later.

Run: python test_action_masks.py
"""
import sys

from environment import CREnv, decode_action, random_strategy, Y_TILES, X_TILES
from core import Position


def engine_would_accept(env, slot, y, x):
    """Ground truth: does the real engine accept this action? Checked against a deep copy so
    the probe itself can't mutate the state being tested.

    slot 0 means "do nothing", and the engine ignores the tile coordinates entirely in that
    case - so all 576 slot-0 encodings are 576 identical ways to pass. The mask deliberately
    exposes only the canonical (0,0) one and hides the other 575 duplicates: they're
    redundant probability mass in the policy's distribution, not distinct choices. This is
    an intentional narrowing of the engine's behavior, so it's asserted here explicitly
    rather than treated as drift.
    """
    import copy
    if slot == 0:
        return (y, x) == (0, 0)
    probe = copy.deepcopy(env.battle)
    p0 = probe.players[0]
    card_name = p0.cycle[slot - 1]
    return bool(probe.deploy_card(0, card_name, Position(x + 0.5, y + 0.5)))


def check_state(env, label):
    mask = env.action_masks()
    mismatches = []
    for action in range(5 * Y_TILES * X_TILES):
        slot, y, x = decode_action(action)
        expected = engine_would_accept(env, slot, y, x)
        if bool(mask[action]) != expected:
            mismatches.append((action, slot, y, x, bool(mask[action]), expected))
    legal = int(mask.sum())
    total = len(mask)
    print(f"  {label}: {legal}/{total} legal ({legal/total:.1%}), "
          f"{len(mismatches)} mismatches")
    for m in mismatches[:5]:
        action, slot, y, x, got, want = m
        print(f"    action={action} slot={slot} tile=({y},{x}) mask={got} engine={want}")
    if len(mismatches) > 5:
        print(f"    ... and {len(mismatches) - 5} more")
    return mismatches


def main():
    total_mismatches = 0

    env = CREnv(opponent_model=lambda obs: random_strategy(obs))
    obs, _ = env.reset()
    total_mismatches += len(check_state(env, "fresh reset (5.0 elixir)"))

    # Advance through real play so buildings exist on the board, elixir varies, and towers
    # may have taken damage - the states where the mask's building-collision and
    # tower-fallen branches actually get exercised, unlike a fresh board.
    for checkpoint in (10, 40, 100):
        while env.battle.time < checkpoint and not env.battle.game_over:
            action = random_strategy(env.observe(0))
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
        if env.battle.game_over:
            break
        p0 = env.battle.players[0]
        total_mismatches += len(check_state(
            env, f"t={env.battle.time:.0f}s elixir={p0.elixir:.1f}"))

    print()
    if total_mismatches == 0:
        print("PASS - mask matches engine exactly on every action in every state tested.")
        return 0
    print(f"FAIL - {total_mismatches} total mismatches. The mask and deploy_card have "
          f"drifted apart; fix before training, results will be misleading otherwise.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
