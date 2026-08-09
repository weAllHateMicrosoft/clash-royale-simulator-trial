"""
Re-check whether the 8.5-tile King Tower range is real or a King/Princess mislabeling
artifact. calibrate.py's attribute_tower_hits() groups every tower ptr under a single
'Tower' bucket and never records which physical tower (King vs Left/Right Princess) each
ptr actually is - so a prior one-off analysis claiming "King Tower hits at 8.4-8.6 tiles"
could easily have mislabeled a Princess Tower's ptr as the King Tower's.

This script fixes that: for each tower ptr, it looks at that ptr's own recorded position
and matches it against the known arena tower coordinates (arena.py) to positively identify
King vs Left/Right Princess Tower, then reports attacker-distance stats per *identified*
tower, per match.
"""
import os, sys, json, glob
from collections import defaultdict

TELEMETRY_DIR = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = "/Users/leafer/Documents/GitHub/clash-royale-simulator/telemetry"
SIM_DIR = "/Users/leafer/Documents/GitHub/clash-royale-simulator/src/clasher_new"
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)

sys.path.insert(0, TELEMETRY_DIR)
from calibrate import load_match, card_name, TOWER_KINDS, ID_TO_NAME, TILE, _nearest_sample

KNOWN_TOWERS = {
    'BLUE_KING':  (9.0, 3.0),
    'BLUE_LEFT':  (3.5, 6.5),
    'BLUE_RIGHT': (14.5, 6.5),
    'RED_KING':   (9.0, 29.0),
    'RED_LEFT':   (3.5, 25.5),
    'RED_RIGHT':  (14.5, 25.5),
}

def identify_tower(events):
    x = sum(e['pos_x_7c'] for e in events[:5]) / len(events[:5]) / TILE
    y = sum(e['pos_y_80'] for e in events[:5]) / len(events[:5]) / TILE
    best, best_d = None, 1e9
    for name, (tx, ty) in KNOWN_TOWERS.items():
        d = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        if d < best_d:
            best, best_d = name, d
    return best, best_d, x, y

ATTACKER_RANGE_TILES = 10.0  # widen vs calibrate.py's 7.0 so we don't pre-exclude the case we're testing

def measure_damage_events(events, hp_field='hp_10', min_drop=1):
    hits = []
    for i in range(1, len(events)):
        a, b = events[i - 1], events[i]
        drop = a[hp_field] - b[hp_field]
        if drop >= min_drop:
            hits.append((b['t_ms'], a[hp_field], b[hp_field], drop))
    return hits

def attribute(tower_ptr, tower_events, tower_side, all_entities):
    hits = measure_damage_events(tower_events)
    attributed = []
    for t_ms, hp_before, hp_after, drop in hits:
        t_ref = _nearest_sample(tower_events, t_ms)
        if t_ref is None:
            continue
        tx, ty = t_ref['pos_x_7c'], t_ref['pos_y_80']
        candidates = []
        for ptr, events in all_entities.items():
            if ptr == tower_ptr:
                continue
            first = events[0]
            if first['side_78'] == tower_side:
                continue
            sample = _nearest_sample(events, t_ms, tolerance_ms=100)
            if sample is None:
                continue
            dx = (sample['pos_x_7c'] - tx) / TILE
            dy = (sample['pos_y_80'] - ty) / TILE
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist <= ATTACKER_RANGE_TILES:
                candidates.append((dist, ptr, sample))
        if len(candidates) == 1:
            name = card_name(all_entities[candidates[0][1]][0]['card_id_ac'], all_entities[candidates[0][1]][0]['kind_30'])
            attributed.append((t_ms, drop, candidates[0][0], name))
    return attributed

for path in sorted(glob.glob(os.path.join(TELEMETRY_DIR, 'match_*.jsonl'))):
    entities, snapshots = load_match(path)
    by_card = defaultdict(list)
    for ptr, events in entities.items():
        first = events[0]
        name = card_name(first['card_id_ac'], first['kind_30'])
        by_card[name].append((ptr, events))

    king_dists = []
    princess_dists = []
    for tower_ptr, tower_events in by_card.get('Tower', []):
        ident, ident_err, tx, ty = identify_tower(tower_events)
        tower_side = tower_events[0]['side_78']
        hits = attribute(tower_ptr, tower_events, tower_side, entities)
        # only trust hits against a STATIONARY attacker (ranged unit not still closing distance)
        # since a melee unit's distance-at-hit-registration is contaminated by charge-up time
        ranged_hits = [h for h in hits if h[3] in ('Musketeer', 'Archer')]
        bucket = king_dists if 'KING' in ident else princess_dists
        for t_ms, drop, dist, name in ranged_hits:
            bucket.append((os.path.basename(path), ident, ident_err, dist, name))

    if king_dists or princess_dists:
        print(f"\n{os.path.basename(path)}")
        for label, bucket in (('KING', king_dists), ('PRINCESS', princess_dists)):
            if not bucket:
                continue
            dists = sorted(d for *_, d, _ in bucket)
            print(f"  {label}: n={len(dists)} dists={[round(d,2) for d in dists]}")
