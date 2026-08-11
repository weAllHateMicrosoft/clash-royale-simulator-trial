import battle, player
from new_visualization import Visualizer
from core import Position
from card_utils import Card

import gymnasium as gym
from random import shuffle, randint, uniform
import os
import time
import numpy as np

from stable_baselines3.common.env_checker import check_env

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']

b = battle.BattleState(player.PlayerState(0, player_0_deck, 10),
                       player.PlayerState(1, player_1_deck, 10))

deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']

entity_names = ['None', 'Knight', 'MiniPekka', 'Arrows', 'Minions', 'Archer',
                'Musketeer', 'Fireball', 'Giant', 'King_PrincessTowers',
                'KingTower', 'ArrowsSpell', 'FireballSpell']
# The agent has to learn that it can only deploy fireball and arrows, and the entities that actually appear are
# the arrows/fireball+spells thingy.

card_types = ['troop', 'character', 'spell', 'building']
# Troop mean princess tower, short for tower troop.
# Actual troops are represented as "characters".
speed_types = [0, 0.75, 1.0, 1.5]

# === Reward v2 - see the design conversation this came out of ===
# Tower/crown/win terms are the original baseline weights, not overnight_2_shaped's bumped
# leak weight (0.003) - that run had the weakest eval-vs-random result of the three overnight
# runs, so v2 starts from the clean baseline rather than compounding on an already-questioned
# adjustment. Isolates what the NEW terms below actually contribute.
LEAK_WEIGHT = 0.0012
DEAL_WEIGHT = 0.001
# Spell-efficiency shaping: DISABLED (0.0) by default, deliberately.
# History, so this isn't silently re-enabled without knowing it: these terms were invented
# during an earlier session (not part of the reward design that was actually reasoned
# through), first appeared in `overnight_2_shaped` - the ONLY run of its batch that failed to
# beat random - and then stayed switched on through `parallel_real1` while the docs
# incorrectly claimed they'd been reverted. They also carried a real bug (multi-wave spells
# charged once per wave, so a whiffed Arrows cost 3x a whiffed Fireball); that bug is now
# fixed in step() via per-cast normalization, so if these are re-enabled they at least
# measure what they claim to. Set them non-zero ONLY as a deliberate, isolated experiment
# against an otherwise-identical run.
SPELL_WHIFF_PENALTY = 0.0    # a splash spell that hits zero enemy entities
SPELL_HIT_BONUS = 0.0        # per enemy entity an own splash spell actually hits

# PPO's actual discount factor (SB3 default, unmodified in train.py). Potential-based reward
# shaping's policy-invariance guarantee (Ng, Harada & Russell, 1999 - shaping can only change
# how fast the agent learns, never what it ultimately converges to) requires the exact form
# `reward += GAMMA * Phi(next_state) - Phi(current_state)`, not a naive undiscounted diff.
# This constant MUST match whatever gamma train.py actually trains with, or the guarantee
# doesn't hold as stated.
GAMMA = 0.99

# BASE_RESOURCE_WEIGHT and OVERFLOW_PENALTY are deliberate STARTING GUESSES, not derived
# values - a clean "damage per elixir" constant turned out to be fundamentally confounded
# (depends on whether/how the opponent defends) and not worth chasing. These get calibrated
# against real training behavior instead, same as everything else here.
# Read from the environment so it survives SubprocVecEnv: with 'spawn', each worker is a
# FRESH interpreter that re-imports this module, so a value patched onto the module object in
# the parent process would silently NOT reach the workers - the parent's manifest would claim
# one weight while every worker actually computed rewards with the default. os.environ is
# inherited by spawned children, so this stays consistent across all processes.
BASE_RESOURCE_WEIGHT = float(os.environ.get("CR_RESOURCE_WEIGHT", "0.05"))
OVERFLOW_PENALTY = -0.05
OVERFLOW_THRESHOLD = 9.9  # "at/near cap" - avoids float-precision edge cases at exactly 10.0

Y_TILES, X_TILES = 32, 18


def elixir_phase_weight(t):
    """Reuses the SAME 120s/240s breakpoints as elixir regen timing below (not new/arbitrary
    ones) - elixir advantage matters more when it's scarce (single elixir) than when it's
    abundant (triple), per the design conversation's staircase reasoning."""
    return 2.8 if t < 120 else 1.4 if t < 240 else 2.8 / 3


def troop_value(battle_state, player_id):
    """Coarse proxy for resource value currently on the board: elixir cost x current hp
    fraction, summed over that player's alive troops. Deliberately coarse - precise value
    (position, matchup context, what it might still do) is left to the critic, not hand-
    modeled here. Only 'more troop-value present is better' is confident enough to reward."""
    total = 0.0
    for e in battle_state.entities.values():
        if not e.is_alive or e.player != player_id:
            continue
        if not isinstance(e, battle.Troop):
            continue
        if e.data.hp > 0:
            total += e.data.elixir * (e.hp / e.data.hp)
    return total


def resource_potential(battle_state):
    """Phi(state) for the resource-advantage shaping term, from player 0's perspective:
    banked elixir advantage + board troop-value advantage, phase-weighted. Elixir banked and
    elixir already spent-but-still-alive-as-troops are the same underlying resource in two
    different states, so they're combined into one potential rather than two separate terms."""
    p0, p1 = battle_state.players
    elixir_adv = p0.elixir - p1.elixir
    troop_adv = troop_value(battle_state, 0) - troop_value(battle_state, 1)
    return BASE_RESOURCE_WEIGHT * elixir_phase_weight(battle_state.time) * (elixir_adv + troop_adv)


def decode_action(action):
    """Flat action index (0..2879) -> (slot, y, x). Inverse of encode_action."""
    action = int(action)
    slot, remainder = divmod(action, Y_TILES * X_TILES)
    y, x = divmod(remainder, X_TILES)
    return slot, y, x


def encode_action(slot, y, x):
    """(slot, y, x) -> flat action index. Inverse of decode_action - used by random_strategy
    and anything else that wants to construct an action rather than receive one from the
    policy."""
    return slot * (Y_TILES * X_TILES) + y * X_TILES + x


def compute_action_masks(battle_state, player_id=0):
    """Boolean mask over all 5*32*18 actions: True = the engine will actually accept it.

    Without this, ~91% of the policy's deploy attempts are rejected outright (measured),
    so most of what it "does" is nothing at all - the equivalent of letting a chess
    engine propose any piece to any square and silently discarding illegal moves. It has
    to burn training just discovering which buttons are connected before it can learn
    anything about play. sb3-contrib's MaskablePPO consumes this via the standard
    `action_masks()` name and renormalizes the policy distribution over legal actions
    only.

    Every rule below mirrors `BattleState.deploy_card` exactly. If that function's rules
    change, this MUST change with it - a mask that permits something the engine rejects
    silently reintroduces dead actions, and a mask that forbids something legal
    permanently hides a real move from the policy. There's a test for exactly this drift
    in `test_action_masks.py` - run it after touching either side.

    player_id=1 returns the mask in the OPPONENT's own action frame, including the
    mirrored coordinate transform that `opponent_action()` applies. That exists so the
    random baseline opponent can be masked too: an unmasked random opponent has ~91% of
    its own deploys silently rejected, making it a far weaker baseline than "random play"
    implies, and inflating our win rate for reasons unrelated to learning.
    """
    mask = np.zeros(5 * Y_TILES * X_TILES, dtype=bool)
    mask[encode_action(0, 0, 0)] = True  # slot 0 = deliberate no-op, always available

    me = battle_state.players[player_id]
    enemy = battle_state.players[1 - player_id]
    if me.king_tower_hp <= 0:
        return mask  # can_play_card() rejects everything once the king tower is down

    # Tile centers in the ACTUAL arena frame. For player 0, step() maps action (y,x) to
    # Position(x+0.5, y+0.5). For player 1, opponent_action() mirrors it to
    # Position(18-(x+0.5), 32-(y+0.5)) - so the legality rules must be evaluated at the
    # mirrored point, not the raw one.
    ys = np.arange(Y_TILES) + 0.5
    xs = np.arange(X_TILES) + 0.5
    if player_id == 0:
        pos_y = ys[:, None]
        pos_x = xs[None, :]
    else:
        pos_y = (32.0 - ys)[:, None]
        pos_x = (18.0 - xs)[None, :]
    yy, xx = pos_y, pos_x

    # --- Spatial legality for non-spell cards (identical for every troop/building) ---
    allowed = np.ones((Y_TILES, X_TILES), dtype=bool)
    if player_id == 0:
        allowed &= ~((yy <= 1.0) & ((xx <= 6.0) | (xx > 12.0)))
        allowed &= ~(yy >= 21.5)
        in_band = (yy >= 15.0) & (yy < 21.5)
    else:
        allowed &= ~((yy > 31.0) & ((xx <= 6.0) | (xx > 12.0)))
        allowed &= ~(yy <= 10.5)
        in_band = (yy > 10.5) & (yy <= 17.0)
    # Inside that band the tile is only usable once the enemy princess tower covering
    # that side has fallen (bridge-side expansion after a tower kill).
    if enemy.left_tower_hp > 0:
        allowed &= ~(in_band & (xx <= 9.5))
    if enemy.right_tower_hp > 0:
        allowed &= ~(in_band & (xx > 9.5))

    # Buildings physically block deployment. Computed once here rather than per card,
    # since deploy_card checks it with a fixed mover_radius of 0 regardless of card.
    for entity in battle_state.entities.values():
        if not isinstance(entity, battle.Building) or not entity.is_alive:
            continue
        r = entity.data.collision_radius
        if r <= 0:
            continue
        dy = yy - entity.position.y
        dx = xx - entity.position.x
        allowed &= ~((dy * dy + dx * dx) < (r * r))

    # --- Per-slot affordability, then combine with the spatial mask ---
    for slot in range(1, 5):
        card_name = me.cycle[slot - 1]
        if not me.can_play_card(card_name):
            continue  # unaffordable or not actually in hand - whole slot stays masked off
        if Card(card_name).type == 'spell':
            per_slot = np.ones((Y_TILES, X_TILES), dtype=bool)  # spells ignore deploy zones
        else:
            per_slot = allowed
        base = slot * (Y_TILES * X_TILES)
        mask[base:base + Y_TILES * X_TILES] = per_slot.reshape(-1)

    return mask


class CREnv(gym.Env):
    def __init__(self, opponent_model=None, opponent_checkpoint_path=None, visualize=False, speed=1.0):
        super().__init__()
        self.opponent = opponent_model
        # For SubprocVecEnv-based parallel training: each env runs in its own OS process, so
        # the opponent can't be a live Python object shared from the main process (that
        # in-memory reference doesn't cross process boundaries). Instead, each process's env
        # watches a checkpoint file on disk and reloads it when it changes - see
        # _maybe_reload_opponent(). Only used when opponent_model isn't given directly.
        self.opponent_checkpoint_path = opponent_checkpoint_path
        self._opponent_mtime = None
        # Set by train.py when opponent-mixture training is enabled; None keeps the old
        # single-snapshot self-play behaviour exactly as it was.
        self.opponent_pool = None
        self._pool_opponent = None
        self.battle: battle.BattleState = None
        self.speed = speed
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)
        })
        # Flat joint action space (5 slots x 32 y x 18 x = 2880 choices), not MultiDiscrete.
        # MultiDiscrete([5,32,18]) samples slot/y/x as three independent, simultaneous
        # decisions from the same snapshot - the position logits aren't actually conditioned
        # on which card got sampled in that draw, just on whatever the shared trunk encodes
        # about the hand in general. A flat Discrete forces every (card, position) pair to be
        # its own single choice, so the network can only express "this tile is good" in terms
        # of "this tile is good FOR THIS CARD" - there's no other way to represent it. This
        # uses SB3's standard CategoricalDistribution (same code path as any Discrete action
        # space) - no custom policy, no custom log-prob math, nothing new to get subtly wrong.
        self.action_space = gym.spaces.Discrete(5 * 32 * 18)

        self.visualize = visualize
        self.visualizer = None

    def _maybe_reload_opponent(self):
        """Checked once per episode (not mid-match) - reloads the opponent from disk only if
        the checkpoint file actually changed since we last loaded it."""
        if self.opponent_checkpoint_path is None:
            return
        path = f"{self.opponent_checkpoint_path}.zip"
        if not os.path.exists(path):
            return
        mtime = os.path.getmtime(path)
        if self._opponent_mtime is not None and mtime <= self._opponent_mtime:
            return
        # The self-play opponent must be masked exactly like the learner is, or self-play
        # trains against a crippled version of itself (~91% of its deploys silently
        # rejected) and the learner learns to beat a straw opponent.
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(self.opponent_checkpoint_path, device="cpu")
        self._opponent_mtime = mtime
        self.opponent = lambda obs: model.predict(
            obs, deterministic=False,
            action_masks=self.action_masks(player_id=1))[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._maybe_reload_opponent()
        shuffle(player_0_deck)
        shuffle(player_1_deck)

        # Draw a fresh opponent per episode when a pool is configured. A single fixed
        # opponent makes the enemy board predictable from the clock, which is what let the
        # policy become an open-loop script; a varied opponent makes that fail.
        handicap = 1.0
        if self.opponent_pool is not None:
            self._pool_opponent, handicap = self.opponent_pool.sample()

        # Randomised starting elixir, same anti-script reasoning as shuffling the deck:
        # anything that makes "what time is it" a worse predictor of the right move forces
        # "what is on the board" to carry the load instead. The learner and the opponent
        # draw independently so neither can infer the other's bank from its own.
        p0_elixir = uniform(4.0, 7.0)
        p1_elixir = uniform(4.0, 7.0)
        self.battle = battle.BattleState(
            player.PlayerState(0, player_0_deck[:], p0_elixir),
            player.PlayerState(1, player_1_deck[:], p1_elixir, elixir_rate=handicap))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)
        # Now return initial observation
        return self.observe(0), {}

    def opponent_action(self):
        obs1 = self.observe(1)
        if self._pool_opponent is not None:
            # Scripted bots read the battle state directly; checkpoint bots also need the
            # observation. Dispatch on arity rather than type so new bots need no wiring.
            try:
                opponent_action = self._pool_opponent(self.battle, 1, obs1)
            except TypeError:
                opponent_action = self._pool_opponent(self.battle, 1)
        else:
            opponent_action = self.opponent(obs1)
        slot, y, x = decode_action(opponent_action)
        p1 = self.battle.players[1]
        if slot != 0:
            card_name = p1.cycle[slot - 1]
            self.battle.deploy_card(1, card_name, Position(18-(x+0.5), 32-(y+0.5)))
            # Yes, this transformation seems weird, but it should be correct


    def step(self, action):
        """
        `action` is a single flat integer over 5*32*18 = 2880 choices - decode_action() below
        splits it into (slot, y, x). slot=0 means no action performed. Else deploy the card in
        that hand slot to the corresponding position on the arena.
        A decision is made every 30 frames (which is half a second). The reward is calculated by the damage dealt/taken,
        destroyed tower/lost tower and won game/lose game.
        The opponent is a function that takes in the observation and outputs the action.
        """

        p0, p1 = self.battle.players
        blue_hps_old = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_old = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left = 3-p0.get_crown_count()
        red_left = 3-p1.get_crown_count()
        resource_before = resource_potential(self.battle)

        # Diagnostics, reported through `info` every step. The point of tracking deploy
        # success/failure explicitly: the action space is a flat 2880 with NO legality
        # masking, so most actions the policy can pick are illegal (wrong deploy zone, card
        # unaffordable). Without measuring it, "the agent isn't learning" and "the agent is
        # mostly pressing buttons that do nothing" look identical from the reward curve.
        step_info = {"noop": 0, "deploy_attempted": 0, "deploy_ok": 0, "elixir_capped": 0}

        slot, y, x = decode_action(action)
        if slot != 0:
            card_name = p0.cycle[slot-1]
            step_info["deploy_attempted"] = 1
            step_info["deploy_ok"] = int(bool(
                self.battle.deploy_card(0, card_name, Position(x+0.5, y+0.5))))
        else:
            step_info["noop"] = 1

        self.battle.spell_impact_log.clear()
        self.opponent_action()
        # only make decisions per half second
        for i in range(30):
            if self.battle.game_over:
                break
            for j in range(int(self.speed)):
                self.battle.step(1/60)
            if self.visualizer:
                self.visualizer.render_frame()
                time.sleep(1/60)
        blue_hps_new = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_new = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left_new = 3-p0.get_crown_count()
        red_left_new = 3-p1.get_crown_count()

        reward = 5*(red_left-red_left_new)-5*(blue_left-blue_left_new)+DEAL_WEIGHT*(red_hps_old-red_hps_new)-LEAK_WEIGHT*(blue_hps_old-blue_hps_new)

        # Resource-advantage potential shaping (elixir bank + board troop value). See
        # GAMMA's docstring above for why this exact form, not a naive undiscounted diff.
        resource_after = resource_potential(self.battle)
        reward += GAMMA * resource_after - resource_before

        # Elixir overflow penalty - separate mechanism from the potential term above (this
        # is event-based: "are you at the cap right now", not a function of relative
        # advantage), borrowed from KataCR's reward_builder.py rather than guessed.
        if p0.elixir >= OVERFLOW_THRESHOLD:
            reward += OVERFLOW_PENALTY
            step_info["elixir_capped"] = 1

        # Spell efficiency: only react to OUR OWN casts (player 0) - the opponent's spell
        # choices aren't something this reward should be shaping. A cast that hits nothing
        # is penalized once per cast; each enemy entity actually hit gives a small bonus -
        # this deliberately does not yet distinguish "hit a cheap Skeleton" from "hit a Pekka"
        # (that's the value-weighted version from your original idea list, staged for later
        # once we've seen whether this simpler version already does something reasonable).
        # Known limitation, shipped anyway: multi-wave spells (Arrows fires 3 waves) log one
        # entry per wave, so a fully-whiffed Arrows costs 3x a fully-whiffed Fireball (1
        # wave). Not normalized per-cast tonight - would need tracking which cast a wave
        # belongs to. Watch for whether this visibly distorts Arrows usage before fixing.
        if SPELL_WHIFF_PENALTY or SPELL_HIT_BONUS:
            for caster_player, hits, spell_name in self.battle.spell_impact_log:
                if caster_player != 0:
                    continue
                # Divide by wave count because these terms are defined PER CAST, while
                # spell_impact_log has one entry per wave. Arrows really does strike 3
                # times (3 x 48 dmg vs Fireball's 1 x 269) - that part is correct game
                # behavior. But one wasted Arrows is still one wasted cast, and was being
                # charged 3x a wasted Fireball; likewise it earned 3x the hit bonus for
                # roughly half the total damage.
                waves = max(1, Card(spell_name).projectile_waves)
                per_cast = (SPELL_WHIFF_PENALTY if hits == 0
                            else SPELL_HIT_BONUS * hits)
                reward += per_cast / waves

        if self.battle.game_over:
            #print('Battle over.', self.battle.winner, reward, p0.king_tower_hp, p0.left_tower_hp, p0.right_tower_hp,
            #      p1.king_tower_hp, p1.left_tower_hp, p1.right_tower_hp)
            if self.battle.winner == 0:
                reward += 10
            else:
                reward -= 10
            # End-of-episode outcome facts. These are what actually answer "is it playing
            # better", unlike total reward, which mixes outcome with whatever shaping terms
            # happen to be enabled and so isn't comparable across reward designs at all.
            step_info["episode_end"] = {
                "winner": self.battle.winner,
                "battle_time": self.battle.time,
                # get_crown_count() reports how many of THAT player's own towers are down,
                # so the crowns WE scored are the enemy's count, not ours (same convention
                # step() uses above: `blue_left = 3 - p0.get_crown_count()`).
                "crowns_for": p1.get_crown_count(),
                "crowns_against": p0.get_crown_count(),
                "own_tower_hp": p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp,
                "enemy_tower_hp": p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp,
            }

        return self.observe(0), reward, self.battle.game_over, self.battle.game_over, step_info


    def action_masks(self, player_id=0):
        """Thin delegate to module-level compute_action_masks() - the real implementation
        lives there so tools that hold a raw BattleState (play_vs_ai.py) can produce the
        exact same mask without constructing a CREnv. One implementation, no duplication."""
        return compute_action_masks(self.battle, player_id)

    def observe(self, player_id_observe=0):
        """Gives a representation of game state"""
        obs = np.zeros((32, 18, 15), dtype=np.float32)
        for id, each in self.battle.entities.items():
            if not each.is_alive: continue
            if isinstance(each, battle.Projectile): continue
            entity_id = entity_names.index(each.name)
            card_type = card_types.index(each.data.type)
            player_id = each.player
            elixir = each.data.elixir
            is_air = int(each.data.is_air_unit)
            attacks_ground, attacks_air = int(each.data.attack_ground), int(each.data.attack_air)

            speed = each.data.speed
            hp_left = np.log(each.hp) / 10
            hp_percentage = each.hp / each.data.hp if each.data.hp != 0 else 0
            hit_speed = each.data.hit_speed
            attack_range = each.data.range / 3
            sight_range = each.data.sight_range / 3
            damage = each.data.damage / 200
            projectile_damage = each.data.projectile_data.damage / 200

            x, y = int(each.position.x), int(each.position.y)
            if player_id == 1:
                x = 17-x
                y = 31-y
            obs_arr = np.array([entity_id, player_id, elixir, card_type, speed, is_air, attacks_ground, attacks_air,
                                hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage, projectile_damage])
            obs[y][x] = obs_arr.copy()

        hand = np.array([entity_names.index(each) for each in self.battle.players[player_id_observe].cycle[:5]],
                        dtype=np.int32)

        return {
            'grid': obs,
            'hand': hand,
            'elixir': np.array([self.battle.players[player_id_observe].elixir], dtype=np.float32)
        }


def random_strategy(observation, mask=None):
    """Uniform random action. With `mask`, uniform over LEGAL actions only.

    The unmasked form is kept only for backward compatibility with older checkpoints//runs -
    it is a much weaker baseline than "random play" suggests, because ~91% of the deploys it
    picks are silently rejected by the engine, so it mostly stands still. Any win rate
    measured against the unmasked version is inflated and is NOT comparable to one measured
    against the masked version.
    """
    if mask is not None:
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return encode_action(0, 0, 0)
        return int(legal[randint(0, len(legal) - 1)])
    slot = randint(0, 4)
    y = randint(0, 31)
    x = randint(0, 17)
    return encode_action(slot, y, x)


def masked_random_opponent(env):
    """Random opponent that only picks legal moves - the honest baseline to evaluate against.
    Bound to `env` so it can read live game state, since legality depends on it."""
    return lambda obs: random_strategy(obs, mask=env.action_masks(player_id=1))

if __name__ == '__main__':
    env = CREnv(random_strategy, visualize=True)
    check_env(env)


