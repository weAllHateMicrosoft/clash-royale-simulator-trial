"""
Parses real-game telemetry captured via cr-memory-reader (entity_observed / runtime_snapshot
events, ~30Hz) and derives measured constants (speed, hit interval, damage-per-hit, attack
range, tower activation delay) to compare against what's coded in card_utils.py / gamedata.json.

Usage: python calibrate.py match_1.jsonl [match_2.jsonl ...]
"""
import os
import sys
import json
import glob
from collections import defaultdict

_ORIGINAL_CWD = os.getcwd()
TELEMETRY_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(TELEMETRY_DIR, '..', 'src', 'clasher_new')
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)  # card_utils.py opens gamedata.json etc. relative to cwd

TILE = 1000.0  # pos_x_7c/pos_y_80 units per tile

with open(os.path.join(TELEMETRY_DIR, 'cards.json')) as f:
    _cards = json.load(f)['items']

# The official API's card names ("Archers", "Mini P.E.K.K.A") don't match this sim's internal
# keys ("Archer", "MiniPekka") - card_data's own englishName field is the bridge between them.
from card_utils import card_data as _card_data
_english_to_internal = {v.get('englishName'): k for k, v in _card_data.items() if v.get('englishName')}
ID_TO_NAME = {c['id']: _english_to_internal.get(c['name'], c['name']) for c in _cards}

# kind_30 taxonomy inferred from data: 15 = troop/character, 12/13/14 = tower variants (card_id=-1)
TOWER_KINDS = {12, 13, 14}


def load_match(path):
    """Returns (entities: {ptr: [events...]}, snapshots: [runtime_snapshot events])."""
    entities = defaultdict(list)
    snapshots = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
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
    snapshots.sort(key=lambda e: e['t_ms'])
    return entities, snapshots


def card_name(card_id_ac, kind_30):
    if card_id_ac == -1 and kind_30 in TOWER_KINDS:
        return 'Tower'
    return ID_TO_NAME.get(card_id_ac, f'Unknown({card_id_ac})')


MAX_PLAUSIBLE_SPEED = 5.0  # tiles/s - no real card is anywhere near this; guards against ptr reuse
# (the same memory address getting recycled by a new unit right after the old one dies, which
# looks like one entity teleporting between two unrelated positions)


def measure_speed(events, min_samples=10):
    """Estimate tiles/sec from consecutive samples where the entity is making steady net
    progress in one direction (filters out stationary combat periods, jitter, and ptr-reuse
    teleport artifacts)."""
    speeds = []
    for i in range(1, len(events)):
        a, b = events[i - 1], events[i]
        dt = (b['t_ms'] - a['t_ms']) / 1000.0
        if dt <= 0 or dt > 0.5:  # skip gaps (entity re-observed after a hole in capture)
            continue
        dx = (b['pos_x_7c'] - a['pos_x_7c']) / TILE
        dy = (b['pos_y_80'] - a['pos_y_80']) / TILE
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < 0.002:  # essentially stationary this frame (fighting / idle)
            continue
        speed = dist / dt
        if speed > MAX_PLAUSIBLE_SPEED:  # ptr-reuse teleport, not real movement
            continue
        speeds.append(speed)
    if len(speeds) < min_samples:
        return None
    speeds.sort()
    # median is more robust than mean against acceleration/knockback/collision-jitter frames
    return speeds[len(speeds) // 2]


def measure_damage_events(events, hp_field='hp_10', min_drop=1):
    """Returns list of (t_ms, hp_before, hp_after, drop) whenever hp drops between samples."""
    hits = []
    for i in range(1, len(events)):
        a, b = events[i - 1], events[i]
        drop = a[hp_field] - b[hp_field]
        if drop >= min_drop:
            hits.append((b['t_ms'], a[hp_field], b[hp_field], drop))
    return hits


def _nearest_sample(events, t_ms, tolerance_ms=150):
    """events must be sorted by t_ms. Returns the sample closest to t_ms, or None if
    nothing falls within tolerance (entity not alive/observed at that moment)."""
    best, best_dt = None, None
    for e in events:
        dt = abs(e['t_ms'] - t_ms)
        if best_dt is None or dt < best_dt:
            best, best_dt = e, dt
        if e['t_ms'] > t_ms + tolerance_ms:
            break
    if best_dt is not None and best_dt <= tolerance_ms:
        return best
    return None


ATTACKER_RANGE_TILES = 7.0  # generous ceiling covering melee through max troop range


def attribute_tower_hits(tower_ptr, tower_events, tower_side, all_entities):
    """For each HP-drop on this tower, find opposing-side entities within plausible attack
    range at that moment. A single unambiguous nearby candidate = confident attribution,
    letting us group hits by the *individual attacking unit* (not just card type) so hit
    interval/damage isn't averaged across multiple simultaneous attackers."""
    hits = measure_damage_events(tower_events)
    tower_pos_events = {e['t_ms']: (e['pos_x_7c'], e['pos_y_80']) for e in tower_events}
    # attributed[attacker_ptr] = list of (t_ms, drop)
    attributed = defaultdict(list)
    ambiguous = 0
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
                continue  # same side as the tower, can't be the attacker
            sample = _nearest_sample(events, t_ms, tolerance_ms=100)
            if sample is None:
                continue
            dx = (sample['pos_x_7c'] - tx) / TILE
            dy = (sample['pos_y_80'] - ty) / TILE
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist <= ATTACKER_RANGE_TILES:
                candidates.append((dist, ptr, sample))
        if len(candidates) == 1:
            attributed[candidates[0][1]].append((t_ms, drop))
        elif len(candidates) > 1:
            ambiguous += 1
    return attributed, ambiguous


def analyze_match(path):
    print(f'\n{"="*70}\n{path}\n{"="*70}')
    entities, snapshots = load_match(path)
    print(f'{len(entities)} unique entities (ptr), {len(snapshots)} runtime snapshots')

    by_card = defaultdict(list)
    for ptr, events in entities.items():
        first = events[0]
        name = card_name(first['card_id_ac'], first['kind_30'])
        by_card[name].append((ptr, events))

    print('\n--- Movement speed (measured vs coded) ---')
    from card_utils import Card
    for name, instances in sorted(by_card.items()):
        if name in ('Tower', 'Unknown(-1)'):
            continue
        speeds = []
        for ptr, events in instances:
            s = measure_speed(events)
            if s is not None:
                speeds.append(s)
        if not speeds:
            continue
        speeds.sort()
        med = speeds[len(speeds) // 2]
        try:
            coded = Card(name).speed
            coded_str = f'{coded:.3f}'
        except Exception:
            coded_str = 'n/a (not in sim card pool)'
        print(f'  {name:16s} measured={med:.3f} tiles/s (n={len(speeds)} instances)  coded={coded_str}')

    print('\n--- Tower damage attributed to individual attacking units (not just conflated by card type) ---')
    per_card_hits = defaultdict(list)  # name -> list of (drop, interval_or_None)
    total_ambiguous = 0
    for tower_ptr, tower_events in by_card.get('Tower', []):
        tower_side = tower_events[0]['side_78']
        attributed, ambiguous = attribute_tower_hits(tower_ptr, tower_events, tower_side, entities)
        total_ambiguous += ambiguous
        for attacker_ptr, hits in attributed.items():
            hits.sort()
            first_ev = entities[attacker_ptr][0]
            name = card_name(first_ev['card_id_ac'], first_ev['kind_30'])
            for i, (t_ms, drop) in enumerate(hits):
                interval = (hits[i][0] - hits[i - 1][0]) / 1000.0 if i > 0 else None
                per_card_hits[name].append((drop, interval))

    for name, samples in sorted(per_card_hits.items()):
        drops = [d for d, _ in samples]
        intervals = [iv for _, iv in samples if iv is not None and 0.1 < iv < 5.0]
        drops.sort()
        med_drop = drops[len(drops) // 2]
        interval_str = 'n/a'
        if intervals:
            intervals.sort()
            interval_str = f'{intervals[len(intervals)//2]:.3f}s median (n={len(intervals)})'
        print(f'  {name:16s} damage/hit(median)={med_drop} (n={len(drops)} hits, unique values sample={sorted(set(drops))[:5]})  hit_interval={interval_str}')
    print(f'  ({total_ambiguous} hp-drop events had multiple candidate attackers nearby - skipped as ambiguous)')


if __name__ == '__main__':
    paths = [os.path.join(_ORIGINAL_CWD, p) for p in sys.argv[1:]] or sorted(glob.glob(os.path.join(TELEMETRY_DIR, 'match_*.jsonl')))
    for p in paths:
        analyze_match(p)
