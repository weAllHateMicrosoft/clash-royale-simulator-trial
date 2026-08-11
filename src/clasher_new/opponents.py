"""
Opponent pool for self-play, and the scripted bots in it.

WHY THIS EXISTS
---------------
Training previously faced a single frozen snapshot of the learner, refreshed every 50k
steps. That snapshot was itself a script (bridge-spam MiniPekka on a schedule), so the
enemy board state was almost perfectly predictable FROM THE CLOCK. When a feature is
redundant with time, a network learns to ignore it - so the enemy-troop channels never
earned any weight, and the policy became open-loop: it fired Arrows 10-20s after attacking
regardless of whether anything was there, and placed defenders at fixed moments rather than
in response to threats.

Responsiveness is not something a reward term can install. It is a policy that only pays
off against opponents you cannot predict. The fix is therefore an opponent MIXTURE:

- random legal play        - unpredictable in time and place, so clock-scripts fail
- past checkpoints         - prevents over-fitting to the single latest snapshot
- scripted strategy bots   - teach specific lessons random play never assembles

Weak opponents are made threatening with an ELIXIR HANDICAP (see PlayerState.elixir_rate),
never with stat buffs: card stats are calibrated against real telemetry, and altering them
would teach the policy facts about a game that does not exist.

The scripted defender matters most. While bridge-spam goes unpunished it is genuinely the
optimal play, and no amount of reward tuning changes that. Once a defender reliably answers
it, spam stops paying and multi-card pushes (Giant + support) become the best remaining
option - which is the pressure needed for the policy to discover them at all. Random
exploration essentially never stumbles into a two-card combo on its own, because the first
half of a Giant push is worse than doing nothing.
"""
import os
import random

import numpy as np

from environment import (compute_action_masks, decode_action, encode_action,
                          random_strategy)


TILE_BRIDGE_Y = 16          # river row; "at the bridge" in the opponent's own frame
LEFT_BRIDGE_X, RIGHT_BRIDGE_X = 3, 14


def _legal_choice(battle_state, player_id, preferred=None):
    """Pick `preferred` (slot, y, x) if the engine would accept it, else any legal action,
    else the no-op. Keeps every scripted bot honest: a bot that tries an illegal move does
    nothing that tick rather than silently desyncing from the mask."""
    mask = compute_action_masks(battle_state, player_id)
    if preferred is not None:
        idx = encode_action(*preferred)
        if mask[idx]:
            return idx
    legal = np.flatnonzero(mask)
    return int(legal[random.randrange(len(legal))]) if len(legal) else encode_action(0, 0, 0)


LEAK_THRESHOLD = 9.0


def _never_leak(battle_state, player_id, action):
    """Universal anti-leak guard applied to EVERY scripted bot.

    Elixir sitting at the cap is wasted outright, and a handicapped opponent that banks its
    advantage applies no pressure - which defeats the whole purpose of the handicap. Worse,
    a bot that idles cannot cycle its deck: cards only advance when one is played, so a bot
    waiting for a specific card that is not in its opening four waits forever. MEASURED:
    GiantPushBot deadlocked exactly this way, sitting at max elixir 97% of the match with
    zero deploys.

    So: if a bot chose to do nothing while at/near the cap, play SOMETHING legal instead.
    That spends the surplus and rotates the deck toward the card it actually wants.
    """
    slot, _, _ = decode_action(action)
    if slot != 0:
        return action
    if battle_state.players[player_id].elixir < LEAK_THRESHOLD:
        return action
    return _legal_choice(battle_state, player_id, None)


def _slot_of(battle_state, player_id, names):
    """First hand slot (1..4) holding any of `names`, else None."""
    cycle = battle_state.players[player_id].cycle
    for slot in range(1, 5):
        if cycle[slot - 1] in names:
            return slot
    return None


class RandomBot:
    """Uniform over LEGAL actions. The workhorse of the mixture: its deployments are
    unpredictable in both time and place, which is precisely what makes a clock-script
    stop working."""
    name = "random"

    def __call__(self, battle_state, player_id):
        return random_strategy(None, mask=compute_action_masks(battle_state, player_id))


class BridgeSpamBot:
    """Deploys its cheapest heavy hitter at the bridge whenever affordable - the strategy
    the current model discovered. Included so the learner must LEARN TO DEFEND it rather
    than merely out-race it."""
    name = "bridge_spam"
    ATTACKERS = ("MiniPekka", "Knight", "Musketeer")

    def __call__(self, battle_state, player_id):
        slot = _slot_of(battle_state, player_id, self.ATTACKERS)
        if slot is None:
            return encode_action(0, 0, 0)
        x = random.choice((LEFT_BRIDGE_X, RIGHT_BRIDGE_X))
        return _legal_choice(battle_state, player_id, (slot, TILE_BRIDGE_Y - 1, x))


class DefenderBot:
    """Holds elixir and answers threats: when an enemy troop crosses into its half, it drops
    a counter near that troop instead of attacking on a schedule.

    This is the bot that breaks the bridge-spam equilibrium. It is deliberately simple -
    nearest-threat response, no card matchup logic - because its job is to make undefended
    spam stop paying, not to play well."""
    name = "defender"
    DEFENDERS = ("Musketeer", "Knight", "Archer", "Minions", "MiniPekka")

    ATTACKERS = ("MiniPekka", "Knight", "Giant", "Musketeer")

    def __call__(self, battle_state, player_id):
        import battle as battle_mod
        me = battle_state.players[player_id]
        threats = [e for e in battle_state.entities.values()
                    if e.is_alive and isinstance(e, battle_mod.Troop) and e.player != player_id]

        # Spend surplus rather than idle at the cap. MEASURED: without this the bot sat at
        # max elixir 96% of the match and deployed nothing at all, because it only ever
        # reacted to threats - which meant an elixir handicap bought exactly nothing. A
        # handicapped opponent that banks its advantage applies no pressure, and applying
        # pressure is the entire reason the handicap exists.
        if not threats and me.elixir >= 8.0:
            slot = _slot_of(battle_state, player_id, self.ATTACKERS)
            if slot is not None:
                x = random.choice((LEFT_BRIDGE_X, RIGHT_BRIDGE_X))
                return _legal_choice(battle_state, player_id, (slot, TILE_BRIDGE_Y - 1, x))

        if not threats or me.elixir < 4:
            return encode_action(0, 0, 0)
        # Most advanced threat = smallest distance to our side of the board.
        threat = min(threats, key=lambda e: abs(e.position.y - (0 if player_id == 1 else 32)))
        slot = _slot_of(battle_state, player_id, self.DEFENDERS)
        if slot is None:
            return encode_action(0, 0, 0)
        # Convert the threat's arena position into THIS player's action frame.
        tx, ty = threat.position.x, threat.position.y
        if player_id == 1:
            ax, ay = 18 - tx, 32 - ty
        else:
            ax, ay = tx, ty
        ax = int(min(17, max(0, ax)))
        ay = int(min(31, max(0, ay + 2)))   # place slightly behind the threat's path
        return _legal_choice(battle_state, player_id, (slot, ay, ax))


class GiantPushBot:
    """Deploys a Giant at the back, then support behind it. Random play never assembles a
    coherent push, so without this the learner never sees one and cannot learn to defend it."""
    name = "giant_push"
    SUPPORT = ("Musketeer", "Archer", "Minions")

    def __init__(self):
        self.giant_at = None

    def __call__(self, battle_state, player_id):
        me = battle_state.players[player_id]
        giant_slot = _slot_of(battle_state, player_id, ("Giant",))
        if giant_slot is not None and me.elixir >= 9:
            self.giant_at = battle_state.time
            return _legal_choice(battle_state, player_id, (giant_slot, 27, random.choice((5, 12))))
        # Support goes in a few seconds after the Giant, so it walks in behind the tank.
        if self.giant_at is not None and 1.0 < battle_state.time - self.giant_at < 6.0:
            slot = _slot_of(battle_state, player_id, self.SUPPORT)
            if slot is not None:
                return _legal_choice(battle_state, player_id, (slot, 29, random.choice((5, 12))))
        return encode_action(0, 0, 0)


SCRIPTED = (RandomBot, BridgeSpamBot, DefenderBot, GiantPushBot)


class _NoLeak:
    """Wraps a scripted bot so it can never sit on capped elixir. Applied by OpponentPool to
    every scripted opponent, so individual bots stay simple and cannot forget the rule."""

    def __init__(self, bot):
        self.bot = bot
        self.name = bot.name

    def __call__(self, battle_state, player_id):
        return _never_leak(battle_state, player_id, self.bot(battle_state, player_id))


class CheckpointBot:
    """A past checkpoint of the learner, masked exactly as in training."""

    def __init__(self, path, device="cpu"):
        from sb3_contrib import MaskablePPO
        self.name = f"ckpt:{os.path.basename(path)}"
        self.model = MaskablePPO.load(path, device=device)

    def __call__(self, battle_state, player_id, obs=None):
        action, _ = self.model.predict(
            obs, deterministic=False,
            action_masks=compute_action_masks(battle_state, player_id))
        return action


class OpponentPool:
    """Samples a fresh opponent each episode.

    `scripted_prob` is the chance of drawing a scripted bot rather than a past checkpoint.
    Early in training there are no checkpoints yet, so scripted bots carry the whole load;
    the pool falls back to them automatically rather than failing.
    """

    def __init__(self, checkpoint_dir=None, scripted_prob=0.5, max_checkpoints=8,
                 handicap_range=(1.0, 1.4)):
        self.checkpoint_dir = checkpoint_dir
        self.scripted_prob = scripted_prob
        self.max_checkpoints = max_checkpoints
        self.handicap_range = handicap_range
        self._cache = {}

    def _checkpoint_paths(self):
        if not self.checkpoint_dir or not os.path.isdir(self.checkpoint_dir):
            return []
        paths = sorted(os.path.join(self.checkpoint_dir, f)
                        for f in os.listdir(self.checkpoint_dir) if f.endswith(".zip"))
        return paths[-self.max_checkpoints:]

    def sample(self):
        """Returns (callable, handicap). The callable takes (battle_state, player_id[, obs]).

        Handicap applies to SCRIPTED/random bots only; past checkpoints always play at 1.0.
        A checkpoint is already a competent opponent, and an opponent that cannot be beaten
        produces no useful gradient - the learner just loses every episode and learns
        nothing about which of its actions mattered. The handicap exists to make WEAK bots
        threatening, not to make strong ones stronger.

        The handicap is redrawn per episode across a range starting at 1.0, so the learner
        always sees a spread from "fair" to "hard" rather than a single difficulty. That
        keeps some episodes clearly winnable, which is what preserves the learning signal.
        """
        paths = self._checkpoint_paths()
        if paths and random.random() > self.scripted_prob:
            path = random.choice(paths)
            if path not in self._cache:
                try:
                    self._cache[path] = CheckpointBot(path[:-4])
                except Exception:
                    return RandomBot(), random.uniform(*self.handicap_range)
                if len(self._cache) > self.max_checkpoints:
                    self._cache.pop(next(iter(self._cache)))
            return self._cache[path], 1.0        # no handicap for a real policy
        return _NoLeak(random.choice(SCRIPTED)()), random.uniform(*self.handicap_range)
