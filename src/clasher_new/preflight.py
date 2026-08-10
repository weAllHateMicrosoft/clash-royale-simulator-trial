"""
Run this on the TRAINING machine before launching a long run. Takes about a minute and
checks the things that have actually broken before, rather than assuming they work.

    python preflight.py            # checks single-process paths
    python preflight.py --n-envs 6 # ALSO checks parallel, which is where breakage happens

Why it exists: three separate multi-hour runs were launched on configurations that were
never verified on the target machine, and each one wasted the whole run. Parallel training
in particular has broken twice for machine-specific reasons (env-count mismatch on reload,
rollout-buffer device desync), and masking + SubprocVecEnv has NOT been verified on Windows
at the time of writing. A minute here is worth a night.
"""
import argparse
import os
import sys
import tempfile
import traceback


def check(label, fn):
    try:
        detail = fn()
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as exc:
        print(f"  FAIL  {label}")
        print(f"        {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=0,
                         help="If >1, also verifies parallel training end-to-end. Use the "
                              "SAME value you intend to train with.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("PREFLIGHT")
    ok = True

    def _imports():
        import torch
        import sb3_contrib
        import stable_baselines3
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "no CUDA"
        return (f"torch {torch.__version__}, sb3 {stable_baselines3.__version__}, "
                f"sb3-contrib {sb3_contrib.__version__}, {name}")
    ok &= check("imports + CUDA visibility", _imports)

    def _device():
        import torch
        if args.device == "cpu":
            return "cpu requested"
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. A CPU-only torch wheel gets installed silently by "
                "plain `pip install torch` on Windows - reinstall from the CUDA index URL. "
                "Training will otherwise run ~50x slower with no error message.")
        return torch.cuda.get_device_name(0)
    ok &= check("CUDA actually usable", _device)

    def _masks():
        from environment import CREnv, masked_random_opponent
        env = CREnv(opponent_model=lambda obs: obs)
        env.opponent = masked_random_opponent(env)
        env.reset()
        m0 = env.action_masks(player_id=0)
        m1 = env.action_masks(player_id=1)
        if not m0.any() or not m1.any():
            raise RuntimeError("action mask is all-False - policy would have no legal move")
        return f"{int(m0.sum())} legal for p0, {int(m1.sum())} for p1 at reset"
    ok &= check("action masks non-empty for both players", _masks)

    def _mask_matches_engine():
        # The mask duplicates deploy_card's rules; if they drift, masking silently breaks.
        import test_action_masks
        rc = test_action_masks.main()
        if rc != 0:
            raise RuntimeError("mask/engine mismatch - see output above")
        return "0 mismatches"
    ok &= check("mask matches engine exactly", _mask_matches_engine)

    def _single():
        from train import make_ppo, make_eval_env
        model = make_ppo(make_eval_env(), device=args.device)
        model.learn(total_timesteps=256, reset_num_timesteps=False)
        return "256 steps"
    ok &= check("single-process training step", _single)

    def _eval():
        from eval_diagnostics import play_eval_games
        from train import make_ppo, make_eval_env
        model = make_ppo(make_eval_env(), device=args.device)
        stats = play_eval_games(model, make_eval_env, n_games=2)
        if stats["illegal_rate"] > 0:
            raise RuntimeError(
                f"illegal_rate is {stats['illegal_rate']:.2%} but masking should make it 0 - "
                "masks are not reaching the policy")
        return f"illegal_rate={stats['illegal_rate']:.0%} (must be 0%)"
    ok &= check("masked eval produces zero illegal actions", _eval)

    if args.n_envs > 1:
        def _parallel():
            import torch
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import SubprocVecEnv
            from train import make_env_fn, make_ppo, make_eval_env
            tmp = os.path.join(tempfile.gettempdir(), "preflight_opponent")
            model = make_ppo(make_eval_env(), device=args.device)
            model.save(tmp)
            venv = SubprocVecEnv([make_env_fn(tmp) for _ in range(args.n_envs)])
            dev = args.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = MaskablePPO.load(tmp, env=venv, device=dev)
            model.learn(total_timesteps=args.n_envs * 128, reset_num_timesteps=False)
            venv.close()
            return f"{args.n_envs} workers"
        ok &= check(f"parallel training with masking (n_envs={args.n_envs})", _parallel)
    else:
        print("  SKIP  parallel check - pass --n-envs N to include it "
              "(strongly recommended if you plan to train with it)")

    print()
    print("PREFLIGHT PASSED - safe to launch" if ok else
          "PREFLIGHT FAILED - fix the above before starting a long run")
    return 0 if ok else 1


if __name__ == "__main__":
    # Windows 'spawn' re-imports this module in each worker; the guard keeps that safe.
    sys.exit(main())
