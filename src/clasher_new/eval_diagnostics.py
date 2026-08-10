"""
Evaluation that produces something you can actually analyze, instead of one noisy scalar.

Why this exists: the original eval played 5 games and logged only the mean total reward.
Clash Royale outcomes vary enormously game to game, so 5 games is mostly noise - a run's
eval curve could swing from -32 to +5 without the policy changing at all. Every conclusion
drawn from that curve was unreliable. Worse, *total reward* isn't comparable between runs
that use different reward shaping, because the shaping terms are inside the number - so it
can't answer "did this reward change make the agent play better", which is the entire
question being asked of it.

What this logs instead, per evaluation point, to `runs/<name>/eval_log.csv`:
  - win_rate / loss_rate / draw_rate  <- shaping-independent, comparable across ANY reward
  - mean crowns for/against, mean tower HP remaining for/against  <- margin, not just W/L
  - mean battle duration
  - noop_rate           <- fraction of decisions where it chose "do nothing"
  - illegal_rate        <- fraction of deploy ATTEMPTS the engine rejected (bad zone /
                           unaffordable). With a flat 2880-action space and no legality
                           masking, this separates "isn't learning" from "is mostly pressing
                           buttons that do nothing" - two very different problems.
  - elixir_capped_rate  <- fraction of decisions spent sitting at the elixir cap (wasting it)
  - mean_reward         <- kept for continuity with older runs, no longer the headline

Sample size: N_EVAL_GAMES below. 5 was the old value and was far too small. 30 gives a win
rate with a standard error around +-9 percentage points - still not tight, but it can
distinguish "clearly beats random" from "coin flip", which 5 games genuinely could not.
Raise it if eval cost is acceptable; the cost is linear.
"""
import csv
import os

N_EVAL_GAMES = 30

CSV_FIELDS = [
    "timesteps", "games", "win_rate", "loss_rate", "draw_rate",
    "mean_crowns_for", "mean_crowns_against",
    "mean_own_tower_hp", "mean_enemy_tower_hp",
    "mean_battle_time", "noop_rate", "illegal_rate", "elixir_capped_rate",
    "mean_reward",
]


def play_eval_games(model, make_env, n_games=N_EVAL_GAMES, deterministic=False):
    """Plays n_games and returns an aggregate dict matching CSV_FIELDS (minus 'timesteps').

    deterministic=False matches how the policy behaves during training rollouts. A
    deterministic policy can look artificially better or worse than what training is
    actually optimizing, so the default here deliberately mirrors training.
    """
    env = make_env()
    # Mask-aware only if the env provides masks. Kept conditional so this still works with
    # an unmasked env, rather than hard-failing on one.
    masked = hasattr(env, "action_masks")
    wins = losses = draws = 0
    crowns_for = crowns_against = 0
    own_hp = enemy_hp = 0.0
    battle_time = 0.0
    rewards = []
    noops = deploy_attempts = illegal = capped = decisions = 0

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if masked:
                action, _ = model.predict(obs, deterministic=deterministic,
                                           action_masks=env.action_masks())
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, termination, truncation, info = env.step(action)
            done = termination or truncation
            total_reward += reward

            decisions += 1
            noops += info.get("noop", 0)
            deploy_attempts += info.get("deploy_attempted", 0)
            illegal += info.get("deploy_attempted", 0) - info.get("deploy_ok", 0)
            capped += info.get("elixir_capped", 0)

            end = info.get("episode_end")
            if end:
                if end["winner"] == 0:
                    wins += 1
                elif end["winner"] is None:
                    draws += 1
                else:
                    losses += 1
                crowns_for += end["crowns_for"]
                crowns_against += end["crowns_against"]
                own_hp += end["own_tower_hp"]
                enemy_hp += end["enemy_tower_hp"]
                battle_time += end["battle_time"]

        rewards.append(total_reward)

    n = float(n_games)
    return {
        "games": n_games,
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "mean_crowns_for": crowns_for / n,
        "mean_crowns_against": crowns_against / n,
        "mean_own_tower_hp": own_hp / n,
        "mean_enemy_tower_hp": enemy_hp / n,
        "mean_battle_time": battle_time / n,
        # Guarded denominators: a policy that never attempts a deploy would otherwise
        # divide by zero here, and that's exactly the degenerate case worth seeing, not
        # crashing on.
        "noop_rate": noops / decisions if decisions else 0.0,
        "illegal_rate": illegal / deploy_attempts if deploy_attempts else 0.0,
        "elixir_capped_rate": capped / decisions if decisions else 0.0,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }


def append_eval_row(csv_path, row):
    """One CSV per run, appended to at each eval point - readable with any tool (pandas,
    Excel, or the bundled analyze_run.py) without needing TensorBoard's export UI."""
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
