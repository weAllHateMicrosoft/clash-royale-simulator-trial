from environment import (CREnv, masked_random_opponent, random_strategy, entity_names,
                          SPELL_WHIFF_PENALTY, SPELL_HIT_BONUS)
from eval_diagnostics import N_EVAL_GAMES, append_eval_row, play_eval_games

from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import json
import os
import subprocess
import time

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, net_scale: float = 1.0):
        """net_scale multiplies every conv width. NOTE this is not a free upgrade: each
        SubprocVecEnv worker runs the OPPONENT policy's forward pass on CPU every single
        step, so a wider net slows every worker and costs throughput (games/night), unlike
        batch_size/n_steps which cost nothing per step. Scale up deliberately."""
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        self.in_channels = 13 + self.embedding_dim + 4
        c1, c2, c3 = (max(8, int(round(c * net_scale))) for c in (32, 64, 64))
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1, features_dim)

    def forward(self, observation):
        """
        Gets the observation, use the embedding (dim=8) to expand the channels, then use one-hot to further expand the channels.
        The code is ugly but should do the work.
        """
        grid = observation['grid']  # (B, 32, 18, 15)
        hand = observation['hand'].long()  # (B, 5)
        elixir = observation['elixir']

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_embedding(card_ids)

        rest = grid[..., 1:]  # (B, 32, 18, 14)
        x = torch.cat([rest, card_vecs], dim=-1)  # (B, 32, 18, 14+EMBED)
        card_type = x[..., 0].long()  # (B, 32, 18)
        card_type_oh = F.one_hot(card_type, num_classes=4).float()  # (B, 32, 18, 4)
        rest = x[..., 1:]
        x = torch.cat([rest, card_type_oh], dim=-1)
        x = x.permute(0, 3, 1, 2).float()  # (B, C, 32, 18)

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        combined = torch.cat([grid_feat, hand_feat, elixir.float()], dim=1)
        return torch.relu(self.fc(combined))


class WeightsCopyingCallback(BaseCallback):
    """Refreshes the self-play opponent every ~50k steps. Two modes:
    - single-process (opponent_checkpoint_path=None): copies weights directly in-memory into
      the live `opponent` PPO object, as before.
    - parallel (opponent_checkpoint_path given): each SubprocVecEnv worker runs in its own OS
      process, so there's no shared `opponent` object to update directly - instead this saves
      the learner's current weights to a checkpoint file, and each worker's CREnv notices the
      file changed and reloads it on its next episode reset (see environment.py's
      _maybe_reload_opponent).
    Uses a threshold-crossing check, not `num_timesteps % freq == 0` - with multiple parallel
    envs, timesteps advance in jumps of n_envs per tick, so an exact-modulo check can skip
    right past the boundary forever if n_envs doesn't evenly divide freq."""
    def __init__(self, opponent_checkpoint_path=None, sync_freq=50000, verbose=0):
        super().__init__(verbose)
        self.opponent_checkpoint_path = opponent_checkpoint_path
        self.sync_freq = sync_freq
        self._last_sync = 0

    def _on_step(self):
        if self.num_timesteps - self._last_sync >= self.sync_freq:
            self._last_sync = self.num_timesteps
            if self.opponent_checkpoint_path:
                self.model.save(self.opponent_checkpoint_path)
            else:
                opponent.policy.load_state_dict(self.model.policy.state_dict())
        return True

def make_eval_env():
    """Eval env whose random opponent is ALSO restricted to legal moves. An unmasked random
    opponent has ~91% of its deploys silently rejected, so it barely plays - beating it says
    almost nothing. Win rates from this masked baseline are therefore NOT comparable to any
    number produced before this change; they are lower and mean more."""
    env = CREnv(opponent_model=lambda obs: obs)  # placeholder, replaced below
    env.opponent = masked_random_opponent(env)
    return env


class RandomEvalCallback(BaseCallback):
    """Evaluates against a random opponent and records a full diagnostic row, not one scalar.

    Previously this played 5 games and logged only mean total reward. That sample is far too
    small for a game this variable - the resulting curve swung tens of points between eval
    points with no policy change behind it, so it couldn't support any of the conclusions
    that were drawn from it. It also logged total *reward*, which embeds whatever shaping
    terms are active, making two runs with different reward designs incomparable by
    construction - precisely the comparison it kept being used for.

    Now: N_EVAL_GAMES games, logging win rate (shaping-independent, comparable across any
    reward design) plus margin, behavior and action-legality diagnostics, both to TensorBoard
    and to `runs/<name>/eval_log.csv` for offline analysis without TensorBoard.

    Same threshold-crossing check as WeightsCopyingCallback, same reason.
    """
    def __init__(self, csv_path=None, eval_freq=50000, n_games=N_EVAL_GAMES, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.csv_path = csv_path
        self.n_games = n_games
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            stats = play_eval_games(
                self.model, make_env=make_eval_env, n_games=self.n_games)
            for key, value in stats.items():
                self.logger.record(f"eval/{key}", value)
            if self.csv_path:
                append_eval_row(self.csv_path, {"timesteps": self.num_timesteps, **stats})
            print(f"[eval @ {self.num_timesteps}] win_rate={stats['win_rate']:.2f} "
                  f"illegal_rate={stats['illegal_rate']:.2f} noop_rate={stats['noop_rate']:.2f} "
                  f"mean_reward={stats['mean_reward']:.2f}")
        return True


def make_ppo(env, device="auto", seed=None, n_steps=2048, batch_size=64,
              n_epochs=10, features_dim=256, net_scale=1.0, learning_rate=3e-4,
              ent_coef=0.0):
    """SB3's batch_size default of 64 is a poor fit here: 16 envs x 2048 steps = 32768
    samples per rollout, which at batch 64 is 512 minibatches x n_epochs tiny GPU calls per
    update - almost entirely kernel-launch overhead on a modern GPU, and the main reason
    GPU utilization sat near 33%. Larger batches use the hardware properly and cost nothing
    in per-step simulation time."""
    return MaskablePPO(
        "MultiInputPolicy", env, verbose=1, tensorboard_log="./tb_logs/", device=device,
        seed=seed, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        learning_rate=learning_rate, ent_coef=ent_coef,
        policy_kwargs=dict(features_extractor_class=CRFeatureExtractor,
                            features_extractor_kwargs=dict(features_dim=features_dim,
                                                            net_scale=net_scale)),
    )


def load_or_create(checkpoint_name, env, device="auto"):
    path = f"{checkpoint_name}.zip"
    if os.path.exists(path):
        print(f"Loading existing checkpoint: {path}")
        # Load on CPU first, then move to the target device as a separate step. Asking
        # torch to remap a checkpoint straight onto a different device (esp. cuda) *during*
        # deserialization has caused hard, untraceable native crashes (Windows access
        # violation, no Python traceback) when the checkpoint was saved on a different
        # platform/torch build than it's being loaded on - e.g. a Mac-saved CPU checkpoint
        # loaded straight onto a Windows CUDA build. Loading on CPU (matching how it was
        # saved) then doing an ordinary tensor .to(device) transfer avoids that code path.
        model = MaskablePPO.load(checkpoint_name, env=env, device="cpu")
        target = device
        if target == "auto":
            target = "cuda" if torch.cuda.is_available() else "cpu"
        if target != "cpu":
            print(f"Moving loaded model from cpu to {target}.")
            model.policy.to(target)
            model.device = torch.device(target)
        return model
    print(f"No checkpoint at {path} - starting fresh.")
    return make_ppo(env, device)


def git_info():
    """Best-effort provenance so a run's manifest records exactly what code produced it -
    including whether the working tree had uncommitted changes at the time (a run's results
    are meaningless to compare later if you can't tell which reward/engine tweaks were live)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def write_manifest(run_dir, data):
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def make_env_fn(opponent_checkpoint_path, pool_dir=None, scripted_prob=0.5,
                 handicap_max=1.6):
    """Factory for SubprocVecEnv workers. Must be a real module-level function, not a local
    closure defined inside `if __name__ == '__main__':` - Windows' multiprocessing 'spawn'
    start method re-imports this module fresh in each child process, and needs this name to
    already exist at that point, before the __main__ guard's body would ever run there.

    The OpponentPool is constructed INSIDE _init, i.e. in the worker process. Building it in
    the parent and passing it would require pickling loaded torch models across the process
    boundary; building it here means each worker reads checkpoints off disk itself and picks
    up new ones as training writes them."""
    def _init():
        env = CREnv(opponent_checkpoint_path=opponent_checkpoint_path)
        if pool_dir is not None:
            from opponents import OpponentPool
            env.opponent_pool = OpponentPool(checkpoint_dir=pool_dir,
                                              scripted_prob=scripted_prob,
                                              handicap_range=(1.0, handicap_max))
        return env
    return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                         help="Total training steps for this invocation. Use a small number "
                              "(e.g. 2000) for a smoke test - just to confirm the loop runs.")
    parser.add_argument("--checkpoint-name", default=None,
                         help="Base name (no .zip) of an EXISTING checkpoint to continue "
                              "training from. Omit to start fresh.")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--run-name", default=None,
                         help="Name for this run's output folder under runs/. Defaults to a "
                              "timestamp. Everything about one run (model, periodic "
                              "checkpoints, tensorboard log, manifest.json) lives together "
                              "here so nothing overwrites a previous run and nothing about "
                              "what produced it gets lost.")
    parser.add_argument("--note", default="",
                         help="Free-text note on what this run is testing/changing - the "
                              "single most useful field for comparing runs later.")
    parser.add_argument("--device", default="auto",
                         help="'auto', 'cuda', or 'cpu'. 'auto' should pick cuda when "
                              "available, but pass --device cuda explicitly to be certain - "
                              "watch the 'Using ... device' line SB3 prints on startup to "
                              "confirm which one actually got used.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                         help="Entropy bonus. SB3 defaults to 0.0, i.e. NOTHING keeps the "
                              "policy from committing early: the previous run's entropy_loss "
                              "fell from -0.47 to -0.09, meaning it went nearly deterministic "
                              "and stopped sampling alternatives at all. Once collapsed onto "
                              "one strategy, multi-card plays are never tried again, so they "
                              "cannot be discovered on merit. A small value (0.005-0.02) keeps "
                              "exploration alive WITHOUT teaching any particular pattern.")
    parser.add_argument("--opponent-pool", action="store_true",
                         help="Train against a MIXTURE of opponents (random, scripted bots, "
                              "past checkpoints) resampled every episode, instead of a single "
                              "frozen self-snapshot. A single fixed opponent makes the enemy "
                              "board predictable from the clock, which lets the policy become "
                              "an open-loop script that ignores what the opponent does.")
    parser.add_argument("--scripted-prob", type=float, default=0.5,
                         help="Chance an episode draws a scripted/random bot rather than a "
                              "past checkpoint.")
    parser.add_argument("--handicap-max", type=float, default=1.6,
                         help="Max elixir-regen multiplier for WEAK (scripted/random) "
                              "opponents; drawn per-episode from [1.0, this]. Past "
                              "checkpoints always play at 1.0. Makes weak bots threatening "
                              "without altering any card's real stats.")
    parser.add_argument("--n-steps", type=int, default=2048,
                         help="Rollout length PER ENV before each PPO update. Total samples "
                              "per update = n_steps * n_envs. Costs nothing per simulated "
                              "step; larger means better-conditioned gradient estimates.")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="Minibatch size for PPO updates. SB3's default of 64 is far too "
                              "small for this setup and wastes most of the GPU on kernel "
                              "launch overhead - 1024-4096 is far better here.")
    parser.add_argument("--n-epochs", type=int, default=10,
                         help="Passes over each rollout buffer. More learning per collected "
                              "sample, at some risk of over-fitting that rollout.")
    parser.add_argument("--features-dim", type=int, default=256,
                         help="Width of the final feature layer.")
    parser.add_argument("--net-scale", type=float, default=1.0,
                         help="Multiplies every CNN channel width. UNLIKE batch/n-steps this "
                              "DOES cost throughput: each worker runs the opponent policy on "
                              "CPU every step, so a wider net slows every worker.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resource-weight", type=float, default=None,
                         help="Overrides BASE_RESOURCE_WEIGHT (the elixir+troop advantage "
                              "shaping term). Pass 0 to turn that shaping OFF entirely, "
                              "leaving only tower damage / crowns / win-loss. Running one "
                              "job with the default and one with 0 is a clean A/B of whether "
                              "that shaping actually helps - which no run so far has answered.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for reproducibility. Two runs differing ONLY in seed "
                              "measure run-to-run variance, which is what tells you whether a "
                              "gap between two configurations is real or noise.")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                         help="Timesteps between evaluation points. Lower = more points on "
                              "the curve (better for seeing WHEN something changed) at the "
                              "cost of eval time. With --n-envs high, rollouts are fast and "
                              "eval becomes the bottleneck - 25000 is a reasonable floor.")
    parser.add_argument("--eval-games", type=int, default=N_EVAL_GAMES,
                         help=f"Games per evaluation point (default {N_EVAL_GAMES}). The old "
                              "value of 5 was small enough that the eval curve was mostly "
                              "noise - don't lower this below ~20 and then trust the result.")
    parser.add_argument("--n-envs", type=int, default=1,
                         help="Number of parallel environments for rollout collection. 1 = "
                              "original single-process behavior (opponent lives in-memory in "
                              "this process). >1 spawns real OS processes (SubprocVecEnv), "
                              "each with its own opponent copy loaded from a checkpoint file "
                              "that gets refreshed periodically - see WeightsCopyingCallback. "
                              "Leave some cores free for the OS/other users on a shared "
                              "machine - don't set this to the full core count.")
    args = parser.parse_args()

    # Set BEFORE any env or model is constructed: environment.py reads this at import time,
    # and SubprocVecEnv workers re-import it in fresh interpreters that inherit os.environ.
    # Setting it any later would leave workers on the default value.
    if args.resource_weight is not None:
        os.environ["CR_RESOURCE_WEIGHT"] = str(args.resource_weight)
        import importlib
        import environment
        importlib.reload(environment)
        print(f"Resource shaping weight overridden to {args.resource_weight}")

    ppo_kwargs = dict(n_steps=args.n_steps, batch_size=args.batch_size,
                       n_epochs=args.n_epochs, features_dim=args.features_dim,
                       net_scale=args.net_scale, learning_rate=args.learning_rate,
                       ent_coef=args.ent_coef)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model")
    opponent_checkpoint_path = os.path.join(run_dir, "opponent_checkpoint")

    manifest = {
        "run_name": run_name, "note": args.note, "args": vars(args),
        "git": git_info(), "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "effective_reward": {
            "resource_weight": os.environ.get("CR_RESOURCE_WEIGHT", "0.05 (default)"),
            "spell_whiff_penalty": SPELL_WHIFF_PENALTY,
            "spell_hit_bonus": SPELL_HIT_BONUS,
        },
        "opponent": {
            "pool": args.opponent_pool,
            "scripted_prob": args.scripted_prob,
            "handicap_max": args.handicap_max,
        },
    }
    write_manifest(run_dir, manifest)

    starting_checkpoint = args.checkpoint_name

    if args.n_envs > 1:
        # Parallel mode. Build a single throwaway env just to construct/load the learner
        # model (PPO needs an env to infer observation/action spaces from - any CREnv
        # instance works, spaces don't depend on game state), then reload with the real
        # SubprocVecEnv attached (see the load_or_create call below for why reload, not
        # set_env).
        setup_env = make_eval_env()
        if starting_checkpoint:
            model = load_or_create(starting_checkpoint, setup_env, device=args.device)
        else:
            print("No --checkpoint-name given - starting fresh.")
            model = make_ppo(setup_env, device=args.device, seed=args.seed, **ppo_kwargs)
        model.tensorboard_log = os.path.join(run_dir, "tb")

        # Write the initial opponent checkpoint so worker processes have something to load on
        # their very first episode reset, before any WeightsCopyingCallback update fires. Also
        # reload the model itself from this same file with the real multi-env VecEnv attached -
        # SB3's set_env() requires the new env to have the SAME env count as what the model was
        # built with, it can't be used to change parallelism (learned this from a real crash,
        # not from the docs). PPO.load(path, env=...) has no such restriction.
        #
        # Load DIRECTLY onto the target device here, not via load_or_create's CPU-first path.
        # That path exists specifically to avoid a crash when a checkpoint crosses platforms
        # (e.g. Mac-saved, loaded onto Windows CUDA). This checkpoint was written by THIS
        # process, on THIS machine, moments ago - no cross-platform risk. Loading straight onto
        # the target device also avoids a real bug the CPU-first path has: the rollout buffer's
        # device is fixed when PPO.load() constructs it and does NOT get updated by a later
        # policy.to(device) call, so train() ends up reading CPU-side buffer data against a
        # GPU-side policy the first time it actually runs - real crash, found on real hardware.
        model.save(opponent_checkpoint_path)
        pool_dir = os.path.join(run_dir, "checkpoints") if args.opponent_pool else None
        vec_env = SubprocVecEnv([
            make_env_fn(opponent_checkpoint_path, pool_dir=pool_dir,
                         scripted_prob=args.scripted_prob,
                         handicap_max=args.handicap_max)
            for _ in range(args.n_envs)])
        resolved_device = args.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MaskablePPO.load(opponent_checkpoint_path, env=vec_env, device=resolved_device)
        model.tensorboard_log = os.path.join(run_dir, "tb")
        weights_cb = WeightsCopyingCallback(opponent_checkpoint_path=opponent_checkpoint_path)
    else:
        # env's opponent_model gets replaced below once `opponent` exists - self-play needs
        # the opponent policy wired into the env, not just sitting in a variable it never reads.
        env = make_eval_env()
        if starting_checkpoint:
            model = load_or_create(starting_checkpoint, env, device=args.device)
        else:
            print("No --checkpoint-name given - starting fresh.")
            model = make_ppo(env, device=args.device, seed=args.seed, **ppo_kwargs)
        model.tensorboard_log = os.path.join(run_dir, "tb")

        # self-play opponent starts as a copy of the learner (even a freshly-initialized one) -
        # self-play works by both sides starting equally weak/strong and improving together, it
        # doesn't need a good policy to begin with, just *a* policy.
        # MUST use the same ppo_kwargs as the learner: the opponent receives the
        # learner's weights via load_state_dict, which fails outright if the two
        # networks differ in width (e.g. --net-scale set on one and not the other).
        opponent = make_ppo(env, device=args.device, **ppo_kwargs)
        opponent.policy.load_state_dict(model.policy.state_dict())
        env.opponent = lambda obs: opponent.predict(obs, deterministic=False)[0]
        weights_cb = WeightsCopyingCallback()

    cb = CheckpointCallback(save_freq=args.save_freq, save_path=os.path.join(run_dir, "checkpoints"), name_prefix="cr")
    eval_cb = RandomEvalCallback(csv_path=os.path.join(run_dir, "eval_log.csv"),
                                 eval_freq=args.eval_freq, n_games=args.eval_games)
    try:
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False,
                    callback=[cb, weights_cb, eval_cb])
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        # Deliberate Ctrl+C is a normal way to end an overnight run - record it as such
        # rather than "failed", which would imply a crash. The finally block below still
        # saves the model, so stopping early costs nothing but the remaining steps.
        manifest["status"] = "stopped_early"
        print("\nStopped by user - saving model and manifest.")
    except BaseException as e:
        manifest["status"] = "failed"
        manifest["error"] = repr(e)
        raise
    finally:
        print('Saving model.')
        model.save(model_path)
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["final_timesteps"] = model.num_timesteps
        write_manifest(run_dir, manifest)
