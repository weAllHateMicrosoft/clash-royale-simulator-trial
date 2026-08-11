"""
Plays two checkpoints against each other and reports a win rate with a real noise floor.

    python head_to_head.py runs/maxed_b/checkpoints/cr_6000000_steps runs/maxed_b/checkpoints/cr_2000000_steps
    python head_to_head.py A B --games 200

Why this exists: `eval_log.csv`'s win-rate-vs-random saturates. Once a policy beats the
masked random baseline ~100% of the time, that metric cannot distinguish "good" from
"better", so it stops being able to tell you whether more training is still helping.
Beating your own earlier self is the direct test of that, and it is the standard way
self-play progress is measured.

Both sides are masked exactly as in training. Sides are SWAPPED halfway through: player 0
and player 1 do not have identical situations (deploy zones, who acts first within a tick),
so playing every game from one side would fold that asymmetry into the result and read as
a skill difference. Half the games each way cancels it.
"""
import argparse
import math

from sb3_contrib import MaskablePPO

import battle
import player
from core import Position
from environment import (compute_action_masks, decode_action, player_0_deck,
                          player_1_deck, entity_names, card_types)
import numpy as np


def observe(b, player_id):
    """Same observation construction as CREnv.observe, without needing a gym env."""
    obs = np.zeros((32, 18, 15), dtype=np.float32)
    for each in b.entities.values():
        if not each.is_alive or isinstance(each, battle.Projectile):
            continue
        name = each.name
        entity_id = entity_names.index(name) if name in entity_names else 0
        ctype = card_types.index(each.data.type) if each.data.type in card_types else card_types.index('character')
        x, y = int(each.position.x), int(each.position.y)
        if player_id == 1:
            x, y = 17 - x, 31 - y
        if not (0 <= y < 32 and 0 <= x < 18):
            continue
        obs[y][x] = np.array([
            entity_id, each.player, each.data.elixir, ctype, each.data.speed,
            int(each.data.is_air_unit), int(each.data.attack_ground), int(each.data.attack_air),
            np.log(max(each.hp, 1)) / 10,
            each.hp / each.data.hp if each.data.hp else 0,
            each.data.hit_speed, each.data.range / 3, each.data.sight_range / 3,
            each.data.damage / 200, each.data.projectile_data.damage / 200,
        ])
    hand = np.array([entity_names.index(c) if c in entity_names else 0
                      for c in b.players[player_id].cycle[:5]], dtype=np.int32)
    return {'grid': obs, 'hand': hand,
            'elixir': np.array([b.players[player_id].elixir], dtype=np.float32)}


def act(model, b, player_id):
    action, _ = model.predict(observe(b, player_id), deterministic=False,
                               action_masks=compute_action_masks(b, player_id))
    slot, y, x = decode_action(action)
    if slot == 0:
        return
    name = b.players[player_id].cycle[slot - 1]
    if player_id == 0:
        b.deploy_card(0, name, Position(x + 0.5, y + 0.5))
    else:
        b.deploy_card(1, name, Position(18 - (x + 0.5), 32 - (y + 0.5)))


def play_game(model_a, model_b, a_is_player0=True):
    """Returns 1 if model_a wins, 0 if model_b wins, None on a draw."""
    from random import shuffle
    d0, d1 = player_0_deck[:], player_1_deck[:]
    shuffle(d0); shuffle(d1)
    b = battle.BattleState(player.PlayerState(0, d0, 5.0), player.PlayerState(1, d1, 5.0))
    p0_model = model_a if a_is_player0 else model_b
    p1_model = model_b if a_is_player0 else model_a

    while not b.game_over:
        act(p0_model, b, 0)
        act(p1_model, b, 1)
        for _ in range(30):
            if b.game_over:
                break
            b.step(1 / 60)

    if b.winner is None:
        return None
    winner_is_a = (b.winner == 0) == a_is_player0
    return 1 if winner_is_a else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("--games", type=int, default=100)
    args = ap.parse_args()

    a = MaskablePPO.load(args.model_a, device="cpu")
    b = MaskablePPO.load(args.model_b, device="cpu")
    print(f"A = {args.model_a}\nB = {args.model_b}\nplaying {args.games} games "
          f"(sides swapped halfway)\n")

    wins = losses = draws = 0
    for i in range(args.games):
        r = play_game(a, b, a_is_player0=(i < args.games // 2))
        if r is None:
            draws += 1
        elif r:
            wins += 1
        else:
            losses += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.games}: A {wins}W {losses}L {draws}D")

    decided = wins + losses
    if decided == 0:
        print("\nAll games drawn - no signal.")
        return
    rate = wins / decided
    se = math.sqrt(rate * (1 - rate) / decided)
    print(f"\nA wins {rate:.1%} of decided games (+-{2*se:.1%} at 2 SE, n={decided})")
    if rate - 2 * se > 0.5:
        print("A is STRONGER than B - the gap clears the noise floor.")
    elif rate + 2 * se < 0.5:
        print("B is STRONGER than A - the gap clears the noise floor.")
    else:
        print("No detectable difference: the two are within noise of each other.\n"
              "If these are early vs late checkpoints of one run, that means training\n"
              "has stopped producing measurable improvement.")


if __name__ == "__main__":
    main()
