"""
Detects candidate target-switch events from raw captured telemetry - specifically switches
that happen while the entity's apparent previous target is STILL ALIVE, not switches caused
by the previous target simply dying (every troop does that, it's not interesting/card-specific
behavior, and high-damage cards like MiniPekka will trivially show more of it just because they
kill things faster - that's a real confound this file specifically corrects for, not a nuance
being ignored).

The raw capture has no explicit "current target" field (verified directly against real match
data - the second position pair, pos_x2_84/pos_y2_88, turned out to just be each entity's own
position from the previous sample, not a target/destination). So both "what was it walking
toward" and "did that thing survive" have to be inferred from position data, not read directly.

This is a heuristic, not ground truth - treat "candidate" events as leads to inspect against
purpose-built Calibration Playbook scenarios, not confirmed retargets on their own.

Usage: python target_switch_analyzer.py [match_1.jsonl match_2.jsonl ...]
       (defaults to every match_*.jsonl in this directory)
"""
import glob
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import load_match, card_name, TOWER_KINDS, _nearest_sample

MIN_MOVE_TILES = 0.05      # ignore near-stationary samples (fighting/idle jitter, not travel)
MIN_BEARING_SAMPLES = 2    # need at least this many valid movement samples to compare bearings
BEARING_CHANGE_DEG = 60    # a turn sharper than this, between consecutive movement segments,
                            # counts as a candidate redirect - not a proven one
MIN_GAP_MS = 200           # don't double-count the same redirect across near-simultaneous samples

APPROACH_WINDOW_MS = 1000  # how far back to look for "what was it walking toward"
SURVIVAL_CHECK_MS = 800    # the implied previous target must still be observed at least this
                            # long after the switch to count as "still alive", not "just died"
TILE_SCALE = 1000.0


def _bearing(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def _angle_diff(a, b):
    """Smallest difference between two angles in degrees, 0-180."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _position_at(events, t_ms, tolerance_ms=150):
    s = _nearest_sample(events, t_ms, tolerance_ms)
    if s is None:
        return None
    return s['pos_x_7c'] / TILE_SCALE, s['pos_y_80'] / TILE_SCALE


def find_bearing_changes(events):
    """Raw candidate redirects from movement direction alone - the same detector as before,
    now just the first stage. Returns [(t_ms, angle_change_deg), ...]."""
    moves = []
    for i in range(1, len(events)):
        a, b = events[i - 1], events[i]
        dx = (b['pos_x_7c'] - a['pos_x_7c']) / TILE_SCALE
        dy = (b['pos_y_80'] - a['pos_y_80']) / TILE_SCALE
        dist = math.hypot(dx, dy)
        if dist < MIN_MOVE_TILES:
            continue
        moves.append((b['t_ms'], dx, dy))

    if len(moves) < MIN_BEARING_SAMPLES:
        return []

    candidates = []
    last_flag_ms = -math.inf
    for i in range(1, len(moves)):
        _, dx_prev, dy_prev = moves[i - 1]
        t_cur, dx_cur, dy_cur = moves[i]
        change = _angle_diff(_bearing(dx_prev, dy_prev), _bearing(dx_cur, dy_cur))
        if change >= BEARING_CHANGE_DEG and (t_cur - last_flag_ms) >= MIN_GAP_MS:
            candidates.append((t_cur, round(change, 1)))
            last_flag_ms = t_cur
    return candidates


def _implied_previous_target(mover_events, all_entities, mover_side, switch_t_ms):
    """What was the mover most plausibly walking toward in the window before switch_t_ms?
    Picks the enemy entity it was net-approaching (distance decreasing) and closest to at
    switch time - a proxy for 'what it was heading for', not a certainty."""
    pos_before = _position_at(mover_events, switch_t_ms - APPROACH_WINDOW_MS, tolerance_ms=300)
    pos_at = _position_at(mover_events, switch_t_ms, tolerance_ms=150)
    if pos_before is None or pos_at is None:
        return None

    best_ptr, best_dist_at = None, math.inf
    for ptr, events in all_entities.items():
        if not events or events[0].get('side_78') == mover_side:
            continue  # only enemy entities are valid targets
        enemy_before = _position_at(events, switch_t_ms - APPROACH_WINDOW_MS, tolerance_ms=300)
        enemy_at = _position_at(events, switch_t_ms, tolerance_ms=150)
        if enemy_before is None or enemy_at is None:
            continue
        dist_before = math.hypot(pos_before[0] - enemy_before[0], pos_before[1] - enemy_before[1])
        dist_at = math.hypot(pos_at[0] - enemy_at[0], pos_at[1] - enemy_at[1])
        if dist_at >= dist_before:
            continue  # wasn't net-approaching this one
        if dist_at < best_dist_at:
            best_ptr, best_dist_at = ptr, dist_at
    return best_ptr


def _still_observed_after(events, t_ms, min_duration_ms=SURVIVAL_CHECK_MS):
    """Does this entity's data stream keep going for at least min_duration_ms past t_ms?
    If its last sample is at/before t_ms (or shortly after), treat it as having died right
    around the switch - not a genuine 'still alive' case."""
    if not events:
        return False
    last_t = events[-1]['t_ms']
    return (last_t - t_ms) >= min_duration_ms


def analyze_match(path):
    """Returns {card_name: {'total_redirects': n, 'while_target_alive': n, 'instances': n}}."""
    entities, _ = load_match(path)
    per_card = defaultdict(lambda: {'total_redirects': 0, 'while_target_alive': 0, 'instances': 0})

    for ptr, events in entities.items():
        if not events:
            continue
        first = events[0]
        if first.get('kind_30') in TOWER_KINDS or first.get('kind_30') != 15:
            continue  # ground/air troop instances only
        name = card_name(first['card_id_ac'], first['kind_30'])
        if name in ('Tower', 'Unknown(-1)'):
            continue
        side = first.get('side_78')

        redirects = find_bearing_changes(events)
        stats = per_card[name]
        stats['instances'] += 1
        stats['total_redirects'] += len(redirects)

        for t_ms, _angle in redirects:
            prev_target_ptr = _implied_previous_target(events, entities, side, t_ms)
            if prev_target_ptr is None:
                continue  # couldn't identify a plausible previous target - don't guess
            if _still_observed_after(entities[prev_target_ptr], t_ms):
                stats['while_target_alive'] += 1

    return per_card


def main(paths):
    overall = defaultdict(lambda: {'total_redirects': 0, 'while_target_alive': 0, 'instances': 0})
    for path in paths:
        for name, stats in analyze_match(path).items():
            for k in stats:
                overall[name][k] += stats[k]

    print(f"{'card':16s} {'instances':>10s} {'raw redirects':>14s} {'while target ALIVE':>20s} {'alive/instance':>15s}")
    for name in sorted(overall):
        s = overall[name]
        if s['instances'] == 0:
            continue
        rate = s['while_target_alive'] / s['instances']
        print(f"{name:16s} {s['instances']:>10d} {s['total_redirects']:>14d} {s['while_target_alive']:>20d} {rate:>15.2f}")


if __name__ == '__main__':
    telemetry_dir = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(telemetry_dir, 'match_*.jsonl')))
    main(paths)
