"""
Automated bug-hunter for pathing/collision deadlocks, replacing manual "hey this
looks stuck" reports. The signature that caught the real BattleRam bug generalizes:
a troop with a live target that stops gaining ground AND stops landing hits for
more than STUCK_SECONDS is broken, regardless of which card it is.

Runs every supported troop card through three scenarios (solo vs tower, 3-pack vs
tower, single unit with a building blocking the direct path) and reports anything
that trips the stuck detector, with the tick range and position so it can be
reproduced and root-caused the same way BattleRam was.
"""
import battle, player
from core import Position
from card_utils import Card, card_data

SUPPORTED_TROOPS = [
    "Knight", "Giant", "Archer", "Goblins", "Pekka", "MiniPekka",
    "Minions", "Skeletons", "SkeletonArmy", "Balloon", "Witch",
    "Barbarians", "Golem", "Valkyrie", "Bomber", "Musketeer",
    "BabyDragon", "Prince", "Wizard", "SpearGoblins",
    "GiantSkeleton", "HogRider", "MinionHorde", "RoyalGiant",
    "Princess", "ThreeMusketeers", "BlowdartGoblin", "AngryBarbarians",
    "Bats", "DartBarrell", "RoyalHogs", "IceWizard", "SkeletonWarriors",
    "DarkPrince", "LavaHound", "IceSpirits", "FireSpirits", "Miner",
    "Bowler", "RageBarbarian", "BattleRam", "ZapMachine",
]

STUCK_SECONDS = 2.5
STUCK_MOVE_EPS = 0.05  # tiles of net progress toward target that counts as "still advancing"
SIM_SECONDS = 30
DT = 1 / 60


def resolvable(name):
    try:
        c = Card(name)
        return c.type == 'character'
    except Exception:
        return False


def run_scenario(deck_card, scenario):
    """Returns a list of stuck-report dicts, empty if nothing tripped."""
    if scenario == 'obstacle':
        filler_deck = [deck_card, 'Cannon', deck_card, deck_card, deck_card, deck_card, deck_card, deck_card]
    else:
        filler_deck = [deck_card] * 8
    b = battle.BattleState(player.PlayerState(0, filler_deck, 10.0), player.PlayerState(1, ['Knight'] * 8, 10.0))

    if scenario == 'solo':
        b.deploy_card(0, deck_card, Position(9.0, 6.0))
    elif scenario == 'crowd':
        for x in (8.0, 9.0, 10.0):
            b.deploy_card(0, deck_card, Position(x, 6.0))
    elif scenario == 'obstacle':
        b.deploy_card(0, 'Cannon', Position(9.0, 9.0))
        b.deploy_card(0, deck_card, Position(9.0, 6.0))

    # per-entity tracking: id -> {'last_dist': float, 'best_dist': float, 'stuck_since': float or None, 'reported': bool}
    tracked = {}
    reports = []

    for tick in range(int(SIM_SECONDS / DT)):
        b.step(DT)
        for e in b.entities.values():
            if not e.is_alive or e.player != 0 or e.card_name not in SUPPORTED_TROOPS and e.card_name != deck_card:
                continue
            if not hasattr(e, 'target_id'):
                continue
            tgt = b.entities.get(e.target_id)
            if not tgt or not tgt.is_alive:
                tracked.pop(e.id, None)
                continue
            dist = e.position.distance_to(tgt.position)
            rng = e.data.range + tgt.data.collision_radius
            st = tracked.setdefault(e.id, {'best_dist': dist, 'stuck_since': None, 'reported': False, 'card': e.card_name})
            if dist < st['best_dist'] - STUCK_MOVE_EPS:
                st['best_dist'] = dist
                st['stuck_since'] = None
            elif dist > rng:  # not yet close enough to attack, and not making progress
                if st['stuck_since'] is None:
                    st['stuck_since'] = b.time
                elif b.time - st['stuck_since'] > STUCK_SECONDS and not st['reported']:
                    st['reported'] = True
                    reports.append({
                        'card': st['card'], 'scenario': scenario, 't': round(b.time, 2),
                        'pos': (round(e.position.x, 2), round(e.position.y, 2)),
                        'dist': round(dist, 3), 'range_needed': round(rng, 3),
                        'target': tgt.name,
                    })
            else:
                st['stuck_since'] = None
        if b.game_over:
            break
    return reports


def main():
    cards = [c for c in SUPPORTED_TROOPS if resolvable(c)]
    print(f'Testing {len(cards)} troop cards x 3 scenarios = {len(cards)*3} runs...\n')
    all_reports = []
    for card in cards:
        for scenario in ('solo', 'crowd', 'obstacle'):
            reports = run_scenario(card, scenario)
            for r in reports:
                print(f"STUCK  {r['card']:16s} [{r['scenario']:8s}] t={r['t']:>6.2f}s pos={r['pos']} "
                      f"dist={r['dist']} needed<={r['range_needed']} target={r['target']}")
                all_reports.append(r)

    print(f'\n{len(all_reports)} stuck-signature hits across {len(cards)} cards.')
    flagged_cards = sorted(set(r['card'] for r in all_reports))
    if flagged_cards:
        print('Cards to investigate:', ', '.join(flagged_cards))
    else:
        print('No cards tripped the stuck detector.')


if __name__ == '__main__':
    main()
