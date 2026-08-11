"""
Measures whether a policy REACTS to the board, or just runs a script.

    python responsiveness_probe.py cr_6720000_steps
    python responsiveness_probe.py A B          # compare two checkpoints

The problem this exists for: a policy trained against a predictable opponent can score well
while completely ignoring what the opponent does - firing spells on a timer, placing
defenders at fixed moments. From win rate alone that is indistinguishable from real play.
This gives it a number.

METHOD
------
Build two states that are IDENTICAL except for an enemy threat, and compare the policy's
action distribution between them. A responsive policy must act differently when a Giant is
walking at its tower than when the lane is empty; a script cannot tell the difference.

Reported per scenario:
  divergence  - total variation distance between the two action distributions, 0..1.
                0.0 = literally identical behaviour (pure script).
  react_rate  - how often the threat state produces a deploy that the empty state does not.
  top_action  - the single most likely action in each state, for eyeballing what changed.

A control scenario ("no_change") compares a state against ITSELF through the same sampling
path. Its divergence is pure sampling noise and sets the floor: any scenario scoring at or
below the control has shown no evidence of reacting.
"""
import argparse
import sys
from collections import Counter

import numpy as np

import battle
import player
from core import Position
from environment import compute_action_masks, decode_action, player_0_deck, player_1_deck


# FIXED order (not the shuffled module list) so hand slots mean the same thing every run:
# slot1=Knight slot2=MiniPekka slot3=Arrows slot4=Fireball, with Giant/Musketeer next.
PROBE_DECK = ['Knight', 'MiniPekka', 'Arrows', 'Fireball', 'Giant', 'Musketeer', 'Minions', 'Archer']
SPELL_SLOTS = (3, 4)          # Arrows, Fireball


def fresh_state(elixir=8.0):
    b = battle.BattleState(player.PlayerState(0, PROBE_DECK[:], elixir),
                            player.PlayerState(1, player_1_deck[:], elixir))
    # Let deploy delays resolve so the board is settled rather than mid-spawn.
    for _ in range(30):
        b.step(1 / 60)
    return b


def add_enemy(b, card, x, y, ticks=60):
    b.deploy_card(1, card, Position(x, y))
    for _ in range(ticks):
        b.step(1 / 60)
    return b


def action_distribution(model, b, samples=200):
    """Samples the policy stochastically in a FIXED state and returns a distribution over
    (slot, y, x). Stochastic sampling matches how the policy behaves during rollouts."""
    from eval_diagnostics import play_eval_games  # noqa: F401  (import kept for parity)
    obs = observe_state(b, 0)
    masks = compute_action_masks(b, 0)
    counts = Counter()
    for _ in range(samples):
        action, _ = model.predict(obs, deterministic=False, action_masks=masks)
        counts[decode_action(action)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def observe_state(b, player_id):
    from head_to_head import observe
    return observe(b, player_id)


def total_variation(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


SCENARIOS = {
    # name: (card, x, y) deployed by the ENEMY, in arena coordinates
    "giant_pushing_left": ("Giant", 4.5, 20.0),
    "minipekka_at_bridge": ("MiniPekka", 3.5, 17.0),
    "musketeer_mid": ("Musketeer", 9.0, 19.0),
    "minions_air": ("Minions", 9.0, 18.0),
}


def probe(model, samples=200):
    results = {}

    base = fresh_state()
    baseline = action_distribution(model, base, samples)

    # Control: same state sampled twice. Divergence here is pure sampling noise and is the
    # floor against which every real scenario must be judged.
    control = action_distribution(model, fresh_state(), samples)
    results["_control_no_change"] = (total_variation(baseline, control), None, None)

    for name, (card, x, y) in SCENARIOS.items():
        threatened = add_enemy(fresh_state(), card, x, y)
        dist = action_distribution(model, threatened, samples)
        div = total_variation(baseline, dist)
        top_base = max(baseline, key=baseline.get) if baseline else None
        top_threat = max(dist, key=dist.get) if dist else None
        results[name] = (div, top_base, top_threat)
    return results


def report(label, results):
    control = results["_control_no_change"][0]
    print(f"\n=== {label} ===")
    print(f"  control (same state twice): divergence {control:.3f}   <- noise floor\n")
    scored = []
    for name, (div, tb, tt) in results.items():
        if name.startswith("_"):
            continue
        verdict = "REACTS" if div > 2 * control else "no evidence"
        scored.append(div)
        print(f"  {name:22s} divergence {div:.3f}   {verdict}")
        if tb is not None:
            print(f"      top action  empty: slot={tb[0]} tile=({tb[1]},{tb[2]})"
                  f"   threatened: slot={tt[0]} tile=({tt[1]},{tt[2]})")
    mean = sum(scored) / len(scored) if scored else 0.0
    reacting = sum(1 for d in scored if d > 2 * control)
    print(f"\n  MEAN divergence {mean:.3f}  (noise floor {control:.3f})")
    print(f"  reacted in {reacting}/{len(scored)} scenarios")
    # Judge on the FRACTION of scenarios that react, not the mean: one strongly-reacting
    # scenario drags the mean above threshold while the policy ignores everything else,
    # which is precisely the failure mode this tool exists to catch.
    if reacting * 2 < len(scored):
        print("  VERDICT: behaves like a SCRIPT - its actions barely change when the board does.")
    else:
        print("  VERDICT: responsive - actions change materially with the board state.")
    return mean


# ---------------------------------------------------------------------------------------
# Targeted metrics: spells and defense specifically.
#
# Divergence alone can look healthy while the two failures that actually matter persist:
# spells fired at nothing, and no defensive response to a push. These measure those two
# directly rather than inferring them.
# ---------------------------------------------------------------------------------------

def spell_behaviour(model, samples=300):
    """Does the policy cast spells AT enemy clusters, or on a timer?

    Compares an empty board against one with three enemy troops bunched together - an
    unambiguous spell target. Reports how much of its action mass goes to spell slots in
    each case, and how far its chosen spell tile lands from the cluster.
    """
    import math

    def measure(b, centroid):
        obs = observe_state(b, 0)
        masks = compute_action_masks(b, 0)
        spell_hits, dists, total = 0, [], 0
        for _ in range(samples):
            action, _ = model.predict(obs, deterministic=False, action_masks=masks)
            slot, y, x = decode_action(action)
            total += 1
            if slot in SPELL_SLOTS:
                spell_hits += 1
                if centroid is not None:
                    dists.append(math.hypot(x + 0.5 - centroid[0], y + 0.5 - centroid[1]))
        med = sorted(dists)[len(dists) // 2] if dists else None
        return spell_hits / total, med

    empty_rate, _ = measure(fresh_state(), None)

    # Three Minions clustered on OUR half - the classic Arrows/Fireball target.
    b = fresh_state()
    cx, cy = 9.0, 12.0
    for dx in (-0.6, 0.0, 0.6):
        b.deploy_card(1, 'Minions', Position(cx + dx, cy))
    for _ in range(60):
        b.step(1 / 60)
    cluster_rate, median_dist = measure(b, (cx, cy))

    return {
        "spell_rate_empty": empty_rate,
        "spell_rate_cluster": cluster_rate,
        "median_aim_error_tiles": median_dist,
    }


def defense_behaviour(model, samples=300):
    """When a push is coming, does it place troops on ITS OWN half to intercept?

    A policy that only ever attacks will keep deploying near the enemy tower regardless of
    what is walking at its own. Reports the share of deploys landing on the defending half.
    """
    def own_half_share(b):
        obs = observe_state(b, 0)
        masks = compute_action_masks(b, 0)
        own, deploys = 0, 0
        for _ in range(samples):
            action, _ = model.predict(obs, deterministic=False, action_masks=masks)
            slot, y, x = decode_action(action)
            if slot == 0:
                continue
            deploys += 1
            if y < 16:            # player 0 defends the low-y half
                own += 1
        return (own / deploys) if deploys else 0.0

    empty = own_half_share(fresh_state())
    b = fresh_state()
    b.deploy_card(1, 'Giant', Position(4.5, 20.0))
    for _ in range(90):
        b.step(1 / 60)
    threatened = own_half_share(b)
    return {"own_half_empty": empty, "own_half_under_push": threatened}


def load(path):
    from sb3_contrib import MaskablePPO
    from train import CRFeatureExtractor
    for feats, scale in ((512, 2.0), (256, 1.0)):
        try:
            return MaskablePPO.load(path, device="cpu", custom_objects={
                "policy_kwargs": dict(
                    features_extractor_class=CRFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=feats, net_scale=scale)),
            })
        except Exception:
            continue
    raise SystemExit(f"could not load {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()
    for path in args.models:
        model = load(path)
        report(path, probe(model, args.samples))
        sb = spell_behaviour(model)
        print(f"\n  SPELLS   empty board: {sb['spell_rate_empty']:.1%} of actions are spells")
        print(f"           enemy cluster: {sb['spell_rate_cluster']:.1%}"
              f"   (should be clearly HIGHER)")
        aim = sb['median_aim_error_tiles']
        print(f"           median aim error: "
              + (f"{aim:.1f} tiles from the cluster" if aim is not None else "never cast"))
        db = defense_behaviour(model)
        print(f"  DEFENSE  deploys on own half, empty board: {db['own_half_empty']:.1%}")
        print(f"           deploys on own half, under push:  {db['own_half_under_push']:.1%}"
              f"   (should be clearly HIGHER)")


if __name__ == "__main__":
    main()
