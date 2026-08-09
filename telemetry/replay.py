"""
Replay comparison: takes a real match's actual deploy sequence (card, side, position,
timestamp - reconstructed from when each ptr first appears in the telemetry) and feeds
the identical sequence into our own battle.BattleState, then diffs the simulated
trajectory against the real one tick-by-tick.

This is the actual fidelity test - constant-by-constant comparison (calibrate.py)
validates numbers in isolation, this validates behavior together. Every position-error
spike in the diff points at a specific bug the same way the BattleRam trace did, just
sourced from real gameplay instead of a hunch.

First-pass scope: deploys are injected directly (bypassing elixir/hand-legality checks)
since captures often start mid-match with unknown exact elixir state - this isolates
physics/behavior fidelity from economy modeling, which can be layered in later once the
underlying movement/combat sim is trustworthy.

Usage: python replay.py match_5.jsonl
"""
import os
import sys
import json
from collections import defaultdict

_ORIGINAL_CWD = os.getcwd()
TELEMETRY_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(TELEMETRY_DIR, '..', 'src', 'clasher_new')
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)

import battle
import player as player_mod
from core import Position

from card_utils import card_data as _card_data
_english_to_internal = {v.get('englishName'): k for k, v in _card_data.items() if v.get('englishName')}
with open(os.path.join(TELEMETRY_DIR, 'cards.json')) as f:
    ID_TO_NAME = {c['id']: _english_to_internal.get(c['name'], c['name']) for c in json.load(f)['items']}

TILE = 1000.0
TROOP_KIND = 15


def load_events(path):
    entities = defaultdict(list)
    snapshots = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('event') == 'entity_observed':
                entities[d['ptr']].append(d)
            elif d.get('event') == 'runtime_snapshot':
                snapshots.append(d)
    for ptr in entities:
        entities[ptr].sort(key=lambda e: e['t_ms'])
    snapshots.sort(key=lambda s: s['t_ms'])
    return entities, snapshots


def build_time_map(snapshots):
    """Returns a function t_ms -> real battle-clock seconds, fit from the runtime_snapshot
    stream (which carries the actual in-game match timer, not just wall-clock capture time).
    The last snapshot(s) of a capture are often a post-game reset artifact (battle_clock_220
    drops to 0 once the match ends and the client moves to the results screen) - trusting that
    directly as the fit's endpoint silently breaks the whole time mapping, so only the
    monotonically-increasing prefix is used for the fit."""
    snapshots = [s for s in snapshots if s.get('battle_clock_220') is not None]
    clean = [snapshots[0]]
    for s in snapshots[1:]:
        if s['battle_clock_220'] >= clean[-1]['battle_clock_220']:
            clean.append(s)
        else:
            break  # first backward jump = end-of-match reset, stop here
    t0_ms, bc0 = clean[0]['t_ms'], clean[0]['battle_clock_220']
    t1_ms, bc1 = clean[-1]['t_ms'], clean[-1]['battle_clock_220']
    span_ms = t1_ms - t0_ms
    span_bc = bc1 - bc0
    rate = span_bc / span_ms if span_ms else 1 / 1000.0

    def to_battle_time(t_ms):
        return bc0 + (t_ms - t0_ms) * rate
    return to_battle_time, bc0


EIGHT_CARD_POOL = {'Knight', 'Giant', 'Musketeer', 'MiniPekka', 'Minions', 'Archer', 'Fireball', 'Arrows'}
SPELL_CARDS = ['Fireball', 'Arrows']


def extract_deploys(entities, to_battle_time):
    """One deploy event per ptr: (battle_time, side, card_name, x_tile, y_tile), taken from
    each entity's first observed sample. Restricted to the 8-card pool (real troops,
    kind_30==15) - anything else (other cards the opponent might have brought, buildings)
    is out of scope and intentionally dropped."""
    deploys = []
    for ptr, events in entities.items():
        first = events[0]
        if first['kind_30'] != TROOP_KIND:
            continue
        name = ID_TO_NAME.get(first['card_id_ac'])
        if name is None or name not in EIGHT_CARD_POOL:
            continue
        deploys.append({
            'ptr': ptr,
            't': to_battle_time(first['t_ms']),
            'side': first['side_78'],
            'card': name,
            'x': first['pos_x_7c'] / TILE,
            'y': first['pos_y_80'] / TILE,
            'real_events': events,
        })
    deploys.sort(key=lambda d: d['t'])
    return deploys


def extract_spell_casts(entities, to_battle_time):
    """Fireball/Arrows don't persist as tracked entities - they only show up as a burst of
    simultaneous flat-damage hits across several troops at once (see the knockback
    investigation earlier this session). Detected the same way: cluster hp-drops within
    150ms, match the per-hit damage value against each spell's known non-tower damage, and
    back-compute the actual cast time from the known travel-time formula (spell spawns at
    the caster's own king tower and flies to the target at proj.speed - matches the
    move-time verification done earlier) so the projectile lands at the same real moment."""
    from card_utils import Card
    import arena as arena_mod
    spell_damage = {name: Card(name).projectile_data.damage for name in SPELL_CARDS}
    spell_speed = {name: Card(name).projectile_data.speed for name in SPELL_CARDS}

    hit_events = []
    for ptr, events in entities.items():
        if events[0]['kind_30'] != TROOP_KIND:
            continue
        for i in range(1, len(events)):
            drop = events[i - 1]['hp_10'] - events[i]['hp_10']
            if drop > 0:
                hit_events.append((events[i]['t_ms'], ptr, drop, events[i]))
    hit_events.sort(key=lambda x: x[0])

    casts = []
    i = 0
    while i < len(hit_events):
        j = i
        cluster = [hit_events[i]]
        while j + 1 < len(hit_events) and hit_events[j + 1][0] - hit_events[i][0] < 150:
            j += 1
            cluster.append(hit_events[j])
        distinct = {c[1]: c for c in cluster}  # dedupe multiple hits on the same ptr in-window
        if len(distinct) >= 2:
            drop_vals = [c[2] for c in distinct.values()]
            common_drop = max(set(drop_vals), key=drop_vals.count)
            for name, dmg in spell_damage.items():
                if abs(common_drop - dmg) <= 2:
                    hit_side = list(distinct.values())[0][3]['side_78']
                    caster_side = 1 - hit_side
                    xs = [c[3]['pos_x_7c'] / TILE for c in distinct.values() if c[2] == common_drop]
                    ys = [c[3]['pos_y_80'] / TILE for c in distinct.values() if c[2] == common_drop]
                    tx, ty = sum(xs) / len(xs), sum(ys) / len(ys)
                    impact_t_ms = cluster[0][0]
                    king = arena_mod.TileGrid.BLUE_KING_TOWER if caster_side == 0 else arena_mod.TileGrid.RED_KING_TOWER
                    dist = ((king.x - tx) ** 2 + (king.y - ty) ** 2) ** 0.5
                    travel_s = dist / spell_speed[name]
                    cast_bt = to_battle_time(impact_t_ms) - travel_s
                    casts.append({'t': cast_bt, 'side': caster_side, 'card': name, 'x': tx, 'y': ty})
                    break
        i = j + 1
    casts.sort(key=lambda c: c['t'])
    return casts


def spawn_direct(b, side, card_name, position):
    """Bypasses deploy_card's elixir/hand/zone-legality checks - see module docstring."""
    from card_utils import Card
    info = Card(card_name)
    eid = b.next_entity_id
    if info.type == 'building':
        ent = battle.Building(eid, position, side, card_name)
    else:
        ent = battle.Troop(eid, position, side, card_name, b)
    b._spawn_entity(ent)
    return ent


def cast_spell_direct(b, side, card_name, position):
    """Same as deploy_card's spell-cast path, minus the elixir/hand bookkeeping - the
    projectile still spawns at the caster's own king tower and travels normally."""
    from card_utils import Card
    from core import BlankEntity
    info = Card(card_name)
    king = b.arena.BLUE_KING_TOWER if side == 0 else b.arena.RED_KING_TOWER
    target = BlankEntity(position)
    ent = battle.Projectile(b.next_entity_id, Position(king.x, king.y), side, card_name, target, False, b)
    b._spawn_entity(ent)
    return ent


def run_replay(path, duration_cap=90.0):
    print(f'\n{"="*70}\n{path}\n{"="*70}')
    entities, snapshots = load_events(path)
    if not snapshots:
        print('no runtime_snapshot events, cannot build a time reference - skipping')
        return
    to_battle_time, t0_bc = build_time_map(snapshots)
    deploys = extract_deploys(entities, to_battle_time)
    deploys = [d for d in deploys if 0 <= d['t'] - t0_bc <= duration_cap]
    spell_casts = extract_spell_casts(entities, to_battle_time)
    spell_casts = [c for c in spell_casts if 0 <= c['t'] - t0_bc <= duration_cap]
    print(f'{len(deploys)} deploy events, {len(spell_casts)} spell casts reconstructed '
          f'(capped to first {duration_cap}s of capture)')

    b = battle.BattleState(
        player_mod.PlayerState(0, ['Knight'] * 8, 10.0),
        player_mod.PlayerState(1, ['Knight'] * 8, 10.0),
    )
    sim_entity_for_ptr = {}
    sim_trajectory = defaultdict(list)  # ptr -> [(battle_time, x, y), ...] logged every tick
    pending = sorted(deploys + [dict(c, ptr=None) for c in spell_casts], key=lambda e: e['t'])
    dt = 1 / 60
    t = t0_bc
    end_t = t0_bc + duration_cap

    while t < end_t:
        while pending and pending[0]['t'] <= t:
            d = pending.pop(0)
            if d.get('ptr') is None:
                cast_spell_direct(b, d['side'], d['card'], Position(d['x'], d['y']))
            else:
                ent = spawn_direct(b, d['side'], d['card'], Position(d['x'], d['y']))
                sim_entity_for_ptr[d['ptr']] = ent
        b.step(dt)
        t += dt
        for ptr, ent in sim_entity_for_ptr.items():
            if ent.is_alive:
                sim_trajectory[ptr].append((t, ent.position.x, ent.position.y))

    # diff at matching battle-time offsets, not "wherever each side ended up" - a unit that
    # died in real life at t=10s gets compared to the sim unit's position AT t=10s, not
    # wherever the sim unit wandered to by the end of the whole run.
    print('\n--- Position error (real vs simulated), matched at the same battle-clock time ---')
    errors_by_card = defaultdict(list)
    for d in deploys:
        traj = sim_trajectory.get(d['ptr'])
        if not traj:
            continue
        traj_times = [p[0] for p in traj]
        for real_ev in d['real_events'][::5]:  # every 5th real sample (~6/sec) is plenty
            real_t = to_battle_time(real_ev['t_ms'])
            # nearest sim sample at or before real_t
            idx = max(0, min(len(traj) - 1, _bisect(traj_times, real_t)))
            sim_t, sx, sy = traj[idx]
            if abs(sim_t - real_t) > 0.2:
                continue  # sim unit already dead/gone or not yet spawned at this moment
            rx, ry = real_ev['pos_x_7c'] / TILE, real_ev['pos_y_80'] / TILE
            err = ((rx - sx) ** 2 + (ry - sy) ** 2) ** 0.5
            errors_by_card[d['card']].append(err)

    for card, errs in sorted(errors_by_card.items()):
        errs.sort()
        n = len(errs)
        print(f'  {card:16s} n={n:4d} samples  median_error={errs[n//2]:.2f} tiles  '
              f'p90_error={errs[int(n*0.9)]:.2f} tiles')


def _bisect(sorted_list, x):
    import bisect
    return bisect.bisect_left(sorted_list, x)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        run_replay(os.path.join(_ORIGINAL_CWD, p))
