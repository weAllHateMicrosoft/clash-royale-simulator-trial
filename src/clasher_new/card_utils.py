import json
from fastcore.all import nested_idx

with open('gamedata.json') as f:
    data = json.load(f)

with open('cards_stats_characters.json') as f:
    characters_data = json.load(f)
    air_units = [each['name'] for each in characters_data if each['flying_height'] != 0]
    characters = {each['name']:each for each in characters_data}
with open('cards_stats_spell.json') as f:
    spells_data = json.load(f)
    spells = {each['name']:each for each in spells_data}
with open('cards_stats_building.json') as f:
    buildings_data = json.load(f)
    buildings = {each['name']:each for each in buildings_data}
with open('cards_stats_projectile.json') as f:
    projectiles = {each['name']:each for each in json.load(f)}

data = data['items']['spells']
card_data = {each['name']: each for each in data}

card_data['Golemite'] = {'name': 'Golemite', 'tidType': 'TID_CARD_TYPE_CHARACTER',
                         'summonCharacterData':card_data['Golem']['summonCharacterData']['deathSpawnCharacterData']}

lava_pups = card_data['LavaHound']['summonCharacterData']['deathSpawnCharacterData']
barbarian = card_data['BattleRam']['summonCharacterData']['deathSpawnCharacterData']
card_data['LavaPups'] = {'name': 'LavaPups', 'tidType': 'TID_CARD_TYPE_CHARACTER', 'summonCharacterData':lava_pups} | lava_pups
card_data['Barbarian'] = {'name': 'Barbarian', 'tidType': 'TID_CARD_TYPE_CHARACTER', 'summonCharacterData': barbarian} | barbarian

# The king tower is not defined in `gamedata.json`, have to hard code it here.
# range/sightRange: an earlier pass "corrected" this to 8500 (8.5 tiles) off calibrate.py's
# attribute_tower_hits(), which buckets ALL towers (King + both Princess) under one 'Tower'
# label and never actually identifies which physical tower fired a given hit - so a Princess
# Tower hit can silently get counted as a "King Tower hit". Re-run with hits grouped by tower
# identity (matched against each tower's own recorded position, not just card kind) instead
# of by proximity guesswork: legitimately-identified King Tower hits against stationary ranged
# troops are sparse and noisy (n=4 across all 11 matches: 6.88/7.87/8.51/9.10 tiles, not a tight
# cluster), while properly-identified Princess Tower hits cluster tightly at 6.4-7.5 tiles -
# consistent with the original 7.0/7.5 values, not 8.5. Reverted both back to the pre-"fix"
# values pending a larger sample. See scratchpad verify_king_range.py re-analysis, prompted by
# the user's real-game observation that the King Tower can't reach a Princess Tower under
# attack (e.g. by Minions) - which an 8.5-tile King Tower range contradicts geometrically,
# since it's only ~6.5 tiles from the King Tower to its own Princess Tower's position.
king_tower_stats = {
    'name': 'KingTower',
    'tidType': 'TID_TYPE_TOWER_TROOP',
    'summonCharacterData': {
        'name': 'KingTower',
        'hitpoints': 2100,
        'hitSpeed': 1000,
        'damage': 109,
        'sightRange': 7000,
        'range': 7000,
        'collisionRadius': 2000,
        'tidTarget': 'TID_TARGETS_AIR_AND_GROUND',
        'deployTime': 3300,
        'loadTime': 700,
        'projectileData': {
            'name': 'KingProjectile',
            'speed': 600,
            'damage': 109,
        }
    }
}
card_data['KingTower'] = king_tower_stats
card_data['King_PrincessTowers']['summonCharacterData'] = card_data['King_PrincessTowers']['statCharacterData']
# Reverted alongside the King Tower fix above - see that comment. Back to the datamined 7500
# (7.5 tiles), which is what the correctly-identified Princess Tower hits actually cluster around.
card_data['King_PrincessTowers']['summonCharacterData']['range'] = 7500
card_data['King_PrincessTowers']['summonCharacterData']['sightRange'] = 7500

# The bundled gamedata.json's speed tiers read low against real gameplay - measured from
# telemetry/match_1-3.jsonl via telemetry/calibrate.py (see that folder for methodology:
# both frame-to-frame and 1s-windowed position-delta measurement agree, so it's not a
# capture-rate artifact). Keyed by raw pre-/60 tier value since speed is a shared tier
# constant, not a per-card stat - "Medium" (60) is confirmed via 3 independent cards
# (Knight, Musketeer, Archer) across all 3 matches, consistent to within ~0.05 tiles/s.
_MEASURED_SPEED_TIER_OVERRIDE = {
    60: 1.17,  # "Medium" - Knight/Musketeer/Archer all measured ~1.15-1.20, was 1.0
    # "Very Fast" - Minions (5 matches) and Mini P.E.K.K.A (3 matches) both converge tightly
    # on ~1.75, superseding the earlier Minions-only 1.80 estimate from just match_1-3.
    90: 1.75,
}
# Only one card measured so far for this tier - not generalized further yet.
_MEASURED_SPEED_OVERRIDE_BY_CARD = {
    'Giant': 1.00,     # "Slow" (raw 45), was 0.75 - confirmed tight (0.995-1.030) across all 6 matches
}

# Real spell knockback (Fireball, Arrows) scales by target weight - not a stat that exists in
# gamedata.json at all, so this has to come from measurement. From telemetry/match_4-6.jsonl:
# pooling every "688 flat damage" AOE cluster (= Fireball hitting mixed troop types at once),
# Giant displaced ~0.31 tiles on average afterward vs ~0.83 tiles for Knight/Musketeer/
# MiniPekka/Minions - i.e. Giant gets ~37% of normal knockback. "Heavy" here is currently only
# confirmed for Giant; other classic tanks are extrapolated from general card design knowledge,
# not measured yet. "Light" (small/swarm units get thrown further) is the user's own reported
# observation from actually playing, not yet confirmed numerically - Archers kept dying to the
# Fireball hit before a displacement sample could be taken, so revisit with a survivable hit
# (e.g. Zap or Arrows on Archers) to actually measure it.
_KNOCKBACK_WEIGHT_MULTIPLIER = {
    'Giant': 0.4,          # measured
    'Golem': 0.4, 'Golemite': 0.4, 'Pekka': 0.4, 'GiantSkeleton': 0.4,  # extrapolated, unmeasured
    'RoyalGiant': 0.4, 'LavaHound': 0.4, 'ElectroGiant': 0.4,
}
_DEFAULT_KNOCKBACK_MULTIPLIER = 1.0


class Card:
    def __init__(self, card_name):
        self.data = card_data[card_name]
        self.data.setdefault('summonCharacterData', self.data)
        self.hp = self.data['summonCharacterData'].get('hitpoints', 0)
        self.elixir = self.data.get('manaCost', 0) # princess towers don't have elixir cost
        self.name = self.data['name']
        self.damage = self.data['summonCharacterData'].get('damage', 0)
        self.spawn_number = self.data.get('summonNumber', 1)
        self.spawn_delay = self.data.get('summonDeployDelay', 0) / 1000
        self.spawn_radius = self.data.get('summonRadius', 550) / 1000

        self.area_damage_radius = self.data['summonCharacterData'].get('areaDamageRadius', 0) / 1000
        self.projectile_damage_radius = nested_idx(self.data, 'summonCharacterData', 'projectileData', 'spawnProjectileData', 'radius')
        self.collision_radius = self.data['summonCharacterData'].get('collisionRadius', 1000) / 1000
        self.hit_speed = self.data['summonCharacterData'].get('hitSpeed', 0) / 1000
        self.load_time = self.data['summonCharacterData'].get('loadTime', 0) / 1000
        raw_speed = self.data['summonCharacterData'].get('speed', 0)
        if self.name in _MEASURED_SPEED_OVERRIDE_BY_CARD:
            self.speed = _MEASURED_SPEED_OVERRIDE_BY_CARD[self.name]
        elif raw_speed in _MEASURED_SPEED_TIER_OVERRIDE:
            self.speed = _MEASURED_SPEED_TIER_OVERRIDE[raw_speed]
        else:
            self.speed = raw_speed / 60
        self.target_only_buildings = self.data['summonCharacterData'].get('tidTarget', '') == "TID_TARGETS_BUILDINGS"
        self.is_air_unit = self.name in air_units or self.data['summonCharacterData'].get('name', '') in air_units
        self.attack_air = 'AIR' in self.data['summonCharacterData'].get("tidTarget", '')
        self.attack_ground = ('GROUND' in self.data['summonCharacterData'].get('tidTarget', '')) or self.target_only_buildings
        self.range = self.data['summonCharacterData'].get('range', 0) / 1000
        self.sight_range = self.data['summonCharacterData'].get('sightRange', 0) / 1000
        self.deploy_time = self.data['summonCharacterData'].get('deployTime', 0) / 1000
        self.charge_range = self.data['summonCharacterData'].get('chargeRange', 0) / 1000
        self.knockback_multiplier = _KNOCKBACK_WEIGHT_MULTIPLIER.get(self.name, _DEFAULT_KNOCKBACK_MULTIPLIER)

        self.projectiles = 'projectileData' in self.data['summonCharacterData']
        self.projectile_data = Projectile(self.data['summonCharacterData'].get('projectileData', {}))
        self.projectile_waves = self.data.get('projectileWaves', 1)
        self.wave_interval = self.data.get("projectileWaveInterval", 0) / 1000

        self.charge_damage = self.data['summonCharacterData'].get('damageSpecial', 0)
        self.shield_health = self.data['summonCharacterData'].get('shieldHitpoints', 0)

        self.lifetime = self.data['summonCharacterData'].get('lifeTime', float('inf'))

        self.death_spawn_data = self.data['summonCharacterData'].get('deathSpawnCharacterData', {})
        self.death_area_effect = self.data['summonCharacterData'].get('deathAreaEffectData', {})
        self.death_damage = self.data['summonCharacterData'].get('deathDamage', 0)

        self.jump_height = self.data['summonCharacterData'].get('jumpHeight', 0)
        self.jump_speed = self.data['summonCharacterData'].get('jumpSpeed', 0) / 60

        self.spawn_data = self.data['summonCharacterData'].get("spawnAreaObjectData", {})
        self.kamikaze = self.data['summonCharacterData'].get('kamikaze', False)

        self.tower_damage_mult = 1+self.data['summonCharacterData'].get('crownTowerDamagePercent', 0)/100

        self.type = self.data.get('tidType', '').split('_')[-1].lower()
        self.rarity = self.data.get('rarity', 'Common')

        if self.name == 'King_PrincessTowers':
            self.collision_radius = 1.5

        self.set_level(11)

    def set_level(self, level):
        if self.rarity == 'Common': level_index = level - 1
        elif self.rarity == 'Rare': level_index = level - 3
        elif self.rarity == 'Epic': level_index = level - 6
        elif self.rarity == 'Legendary': level_index = level - 9
        elif self.rarity == 'Champion': level_index = level - 11

        if self.projectiles:
            projectile_name = self.data['summonCharacterData']['projectileData']['name']
            self.projectile_data.damage = projectiles[projectile_name]['damage_per_level'][level_index]

        if self.type == 'troop':
            building_name = self.data['summonCharacterData']['name']
            self.hp = buildings[building_name]["hitpoints_per_level"][level_index]

        if self.type == 'character':
            character_name = self.data['summonCharacterData']['name']
            self.hp = characters[character_name]["hitpoints_per_level"][level_index]
            if self.damage:
                self.damage = characters[character_name]["damage_per_level"][level_index]
        elif self.type == 'spell':
            # For simplicity, just assume that spells are projectiles, which is already handled
            pass
        return level

class Projectile:
    def __init__(self, projectile_data):
        self.data = projectile_data
        self.damage = self.data.get('damage', 0)
        self.speed = self.data.get('speed', 0) / 60
        self.radius = (self.data.get('spawnProjectileData', {}).get('radius', 0) or self.data.get('radius', 0)) / 1000
        self.target_buff = self.data.get('targetBuffData', {})
        self.buff_time = self.data.get('buffTime', 0) / 1000
        self.hits_air = 'AIR' in self.data.get('tidTarget', '')
        self.hits_ground = 'GROUND' in self.data.get('tidTarget', '') or 'BUILDING' in self.data.get('tidTarget', '')
        self.pushback = self.data.get('pushback', 0) / 1000
        self.name = self.data.get('name', 'Unknown')
        self.roll_range = self.data.get('projectileRange', 0) / 1000
        self.crown_tower_percent = (self.data.get("crownTowerDamagePercent", 0) + 100)/100
        if self.data.get('name') == 'TowerPrincessProjectile':
            self.hits_air = True
            self.hits_ground = True

class TimedExplosiveData:
    def __init__(self, death_spawn_data):
        self.data = death_spawn_data
        self.name = self.data['name']
        self.damage = self.data['deathDamage']
        self.deploy_time = self.data['deployTime'] / 1000
        self.collision_radius = self.data['collisionRadius'] / 1000
        self.range = 3.0
        self.crown_tower_damage_percent = self.data.get('crownTowerDamagePercent', 100) / 100

class AreaEffectData:
    def __init__(self, source_card_name):
        # This only works for lumberjack, will modify later.
        self.data = Card(source_card_name)['summonCharacterData'].get('deathSpawnCharacterData', {}).get('deathAreaEffectData', {})
        self.duration = self.data.get('lifeDuration', 0) / 1000
        self.radius = self.data.get('radius', 0) / 1000
        self.buff_time = self.data.get('buffTime', 0)
        self.buff_data = self.data.get('buffData', {})
        self.speed_multiplier = self.buff_data.get('speedMultiplier')
        self.damage = self.data.get('spawnAreaEffectObjectData', {}).get('damage', 0)
        self.crown_tower_damage_percent = self.buff_data.get('crown', 0) or self.data.get('crownTowerDamagePercent', 0)

if __name__ == '__main__':
    deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    for each in deck:
        print(Card(each).type)
