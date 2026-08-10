"""
Deterministic simulation trace, used to prove an optimization changed nothing.

    python trace_sim.py > before.txt      # on the old code
    python trace_sim.py > after.txt       # on the optimized code
    diff before.txt after.txt             # must be empty

Runs a fixed scripted sequence of deploys (no RNG in the policy, fixed deck order) and
prints a digest of full battle state every tick. Any behavioral difference - a troop one
millimetre off, a different target chosen, a different death - changes the digest.

This exists because the optimizations it guards are the kind that look obviously safe and
occasionally are not: swapping "scan every entity and isinstance-check it" for "iterate a
maintained list" is only equivalent if the list is genuinely complete, and a silent
divergence there would corrupt the simulation everywhere at once.
"""
import hashlib

import battle
import player
from core import Position

DECK0 = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
DECK1 = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']

# (tick_index, player, hand_slot_index, x, y) - a fixed script, no randomness anywhere.
SCRIPT = [
    (10, 0, 0, 8.5, 6.5), (10, 1, 0, 9.5, 25.5),
    (120, 0, 1, 4.5, 10.5), (150, 1, 1, 13.5, 21.5),
    (300, 0, 2, 9.5, 20.5), (330, 1, 2, 8.5, 11.5),
    (500, 0, 3, 3.5, 14.5), (520, 1, 3, 14.5, 17.5),
    (700, 0, 0, 11.5, 8.5), (760, 1, 0, 6.5, 23.5),
    (900, 0, 1, 9.5, 15.5), (960, 1, 1, 9.5, 16.5),
]


def digest(b):
    parts = []
    for eid in sorted(b.entities):
        e = b.entities[eid]
        parts.append(f"{eid}:{e.card_name}:{e.player}:{e.is_alive:d}:"
                     f"{e.position.x:.6f}:{e.position.y:.6f}:{e.hp:.4f}:"
                     f"{getattr(e, 'target_id', None)}")
    for p in b.players:
        parts.append(f"P{p.player_id}:{p.elixir:.6f}:{p.king_tower_hp}:"
                     f"{p.left_tower_hp}:{p.right_tower_hp}:{'|'.join(p.cycle)}")
    return hashlib.sha256(";".join(parts).encode()).hexdigest()[:16]


def main():
    b = battle.BattleState(player.PlayerState(0, DECK0[:], 10.0),
                            player.PlayerState(1, DECK1[:], 10.0))
    script = {}
    for tick, pid, slot, x, y in SCRIPT:
        script.setdefault(tick, []).append((pid, slot, x, y))

    for tick in range(1200):
        for pid, slot, x, y in script.get(tick, []):
            name = b.players[pid].cycle[slot]
            ok = b.deploy_card(pid, name, Position(x, y))
            print(f"tick={tick} deploy p{pid} {name} at ({x},{y}) -> {ok}")
        if b.game_over:
            print(f"tick={tick} GAME OVER winner={b.winner}")
            break
        b.step(1 / 60)
        if tick % 20 == 0:
            print(f"tick={tick} t={b.time:.4f} n={len(b.entities)} {digest(b)}")

    print(f"FINAL t={b.time:.4f} entities={len(b.entities)} digest={digest(b)}")


if __name__ == "__main__":
    main()
