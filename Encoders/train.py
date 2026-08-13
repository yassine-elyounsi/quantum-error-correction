"""
train_wandb.py
==============
Train the QEC encoding agent with full Weights & Biases monitoring.

This script runs the curriculum training loop directly (instead of calling
agent.train()) so it can log every metric to W&B at each interval:

  Per-interval scalar metrics:
    - success_rate            fraction of episodes that fully satisfied KL
    - avg_reward              mean total episode reward
    - avg_episode_length      mean number of gates per episode
    - actor_loss              A2C policy loss
    - critic_loss             A2C value loss
    - entropy                 policy entropy (exploration signal)
    - mean_return             mean GAE return
    - mean_advantage          mean GAE advantage
    - psi                     PPR library-reuse probability (if used)
    - curriculum_phase        which phase (0-3) we are in
    - episodes_per_second     training throughput

  Per-code success rates (one line each):
    - success_rate/[[3,1,2]]
    - success_rate/[[5,1,3]]
    - success_rate/[[7,1,3]]
    - success_rate/[[9,1,3]]

  Best circuits found (logged as a W&B Table):
    - code, length, c_Z, circuit string

Usage
-----
    pip install wandb
    wandb login                          # first time only

    python train_wandb.py                                  # full curriculum
    python train_wandb.py --episodes 5000                  # shorter run
    python train_wandb.py --project my-qec --run-name exp1 # custom names
    python train_wandb.py --offline                        # no internet needed

After training the policy is saved to --save (default encoding_policy.pkl).
"""

import os
import time
import argparse
from collections import defaultdict, deque

import numpy as np

from Encoders.Encoding_agent import EncodingAgent
from Encoders.A2C_agent import collect_episode
from Encoders.meta_env import CURRICULUM_PHASES, TOTAL_EPISODES


def code_label(n, k, d):
    return f"[[{n},{k},{d}]]"


def main():
    parser = argparse.ArgumentParser(description="Train QEC encoding agent with W&B")
    # training
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES,
                        help=f"number of episodes (default full curriculum = {TOTAL_EPISODES})")
    parser.add_argument("--log-every", type=int, default=200,
                        help="log to W&B every N episodes")
    parser.add_argument("--save", type=str, default="encoding_policy.pkl",
                        help="path to save the trained policy")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="checkpoint every N episodes")
    # agent hyperparameters
    parser.add_argument("--n-max", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    # wandb
    parser.add_argument("--project", type=str, default="qec-encoding-agent")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--offline", action="store_true",
                        help="run W&B in offline mode (no internet needed)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="disable W&B entirely (plain console logging)")
    args = parser.parse_args()

    # ---- set up Weights & Biases ------------------------------------
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("wandb not installed.  Run:  pip install wandb")
            print("Falling back to console-only logging.\n")
            use_wandb = False

    if use_wandb:
        if args.offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "episodes":      args.episodes,
                "n_max":         args.n_max,
                "hidden":        args.hidden,
                "lr":            args.lr,
                "gamma":         args.gamma,
                "gae_lambda":    args.gae_lambda,
                "ent_coef":      args.ent_coef,
                "seed":          args.seed,
                "curriculum_phases": len(CURRICULUM_PHASES),
                "total_curriculum_episodes": TOTAL_EPISODES,
                "algorithm":     "A2C + PPR + GAE",
            },
        )

    # ---- build the agent --------------------------------------------
    agent_wrapper = EncodingAgent(
        n_max=args.n_max,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )
    env   = agent_wrapper.env
    agent = agent_wrapper.agent

    print("=" * 66)
    print("TRAINING QEC ENCODING AGENT  (with W&B monitoring)")
    print(f"  obs_dim     : {env.obs_dim}")
    print(f"  num_actions : {env.num_actions}")
    print(f"  episodes    : {args.episodes}")
    print(f"  wandb       : {'on' if use_wandb else 'off'}"
          f"{' (offline)' if args.offline else ''}")
    print("=" * 66)

    # ---- rolling windows for metrics --------------------------------
    W = args.log_every
    win_reward  = deque(maxlen=W)
    win_success = deque(maxlen=W)
    win_length  = deque(maxlen=W)
    win_actor   = deque(maxlen=W)
    win_critic  = deque(maxlen=W)
    win_entropy = deque(maxlen=W)
    win_return  = deque(maxlen=W)
    win_adv     = deque(maxlen=W)
    # per-code success tracking
    per_code_success = defaultdict(lambda: deque(maxlen=W))

    t_start = time.time()
    t_interval = t_start

    for ep in range(1, args.episodes + 1):
        # one episode
        batch, info = collect_episode(env, agent)
        diag = agent.update(batch)

        # collect stats
        win_reward.append(info["total_reward"])
        win_success.append(1.0 if info["success"] else 0.0)
        win_length.append(info["length"])
        win_actor.append(diag["actor_loss"])
        win_critic.append(diag["critic_loss"])
        win_entropy.append(diag["entropy"])
        win_return.append(diag["mean_return"])
        win_adv.append(diag["mean_advantage"])

        code = code_label(env.n, env.k, env.d)
        per_code_success[code].append(1.0 if info["success"] else 0.0)

        # track best circuits
        if info["success"]:
            key = (env.n, env.k, env.d)
            prev = agent_wrapper.best_circuits.get(key, {}).get("length", 1e9)
            if info["length"] < prev:
                agent_wrapper.best_circuits[key] = {
                    "circuit": list(info["history"]),
                    "length":  info["length"],
                    "c_Z":     env.c_Z,
                    "p_I":     env.p_I,
                }

        # ---- log every log_every episodes --------------------------
        if ep % args.log_every == 0:
            now = time.time()
            eps_per_sec = args.log_every / (now - t_interval)
            t_interval = now

            metrics = {
                "success_rate":      float(np.mean(win_success)),
                "avg_reward":        float(np.mean(win_reward)),
                "avg_episode_length": float(np.mean(win_length)),
                "actor_loss":        float(np.mean(win_actor)),
                "critic_loss":       float(np.mean(win_critic)),
                "entropy":           float(np.mean(win_entropy)),
                "mean_return":       float(np.mean(win_return)),
                "mean_advantage":    float(np.mean(win_adv)),
                "curriculum_phase":  env._phase_idx,
                "psi":               agent._psi(),
                "episodes_per_second": eps_per_sec,
                "episode":           ep,
            }
            # per-code success rates
            for c, dq in per_code_success.items():
                if len(dq) > 0:
                    metrics[f"success_rate/{c}"] = float(np.mean(dq))

            if use_wandb:
                wandb.log(metrics, step=ep)

            # console line
            print(
                f"  ep {ep:6d}/{args.episodes}"
                f"  phase {env._phase_idx}"
                f"  SR {metrics['success_rate']:.2f}"
                f"  reward {metrics['avg_reward']:7.2f}"
                f"  len {metrics['avg_episode_length']:5.1f}"
                f"  entropy {metrics['entropy']:.2f}"
                f"  {eps_per_sec:5.1f} eps/s"
            )

        # ---- checkpoint --------------------------------------------
        if ep % args.save_every == 0:
            agent_wrapper.save(args.save)

    # ---- final save -------------------------------------------------
    agent_wrapper.save(args.save)
    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Policy saved to {args.save}")

    # ---- log best circuits as a W&B table ---------------------------
    print("\nBest circuits discovered:")
    rows = []
    for (n, k, d), binfo in sorted(agent_wrapper.best_circuits.items()):
        circuit_str = "  ".join(
            f"CNOT({g[1]}->{g[2]})" if g[0] == "CNOT" else f"{g[0]}({g[1]})"
            for g in binfo["circuit"]
        )
        print(f"  {code_label(n,k,d)}  {binfo['length']} gates  c_Z={binfo['c_Z']:.2f}")
        rows.append([code_label(n, k, d), binfo["length"],
                     round(binfo["c_Z"], 2), circuit_str])

    if use_wandb and rows:
        table = wandb.Table(
            columns=["code", "length", "c_Z", "circuit"],
            data=rows,
        )
        wandb.log({"best_circuits": table})

        # also log final summary metrics
        for (n, k, d), binfo in agent_wrapper.best_circuits.items():
            wandb.run.summary[f"best_length/{code_label(n,k,d)}"] = binfo["length"]

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()