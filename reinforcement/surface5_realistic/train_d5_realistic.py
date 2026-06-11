"""
Training Loop — Realistic d=5 Surface Code (CONTINUING task)
==============================================================
Multi-discrete DDQN on the realistic, continuing surface-code environment.

Continuing-task done handling (CRITICAL)
----------------------------------------
This is a continuing task: new noise arrives every round and the agent must
protect the logical qubit indefinitely. The episode ends in two distinct ways:

  • terminated = logical failure → the qubit truly DIED → no future value
        done_for_buffer = 1.0   → bootstrap CUT:   td = r
  • truncated  = T rounds reached → the qubit is ALIVE, we just stopped
        done_for_buffer = 0.0   → bootstrap KEPT:  td = r + γ·Q(s')

The training loop therefore stores  done = terminated  (NOT terminated|truncated).
Loop control uses  episode_over = terminated or truncated.

Features
--------
  • Dedicated checkpoint folder: checkpoints_d5_continuous/
  • Crash-safe resume with --resume (continues same wandb run)
  • Separate wandb project: surface-code-rl-d5-realistic

Usage
-----
    python -m reinforcement.surface.train_d5_realistic
    python -m reinforcement.surface.train_d5_realistic --resume
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import wandb

from reinforcement.surface5_realistic.Ddqn_agent_multi import MultiDiscreteDDQNAgent
from src.environments.surface5_realistic_env         import SurfaceCodeRealisticEnv

# ── Checkpoint dir ────────────────────────────────────────────────────────────
CKPT_DIR    = os.path.join(_ROOT, "checkpoints_d5_continuous_new")
LATEST_CKPT = os.path.join(CKPT_DIR, "d5_latest.pt")
META_FILE   = os.path.join(CKPT_DIR, "d5_meta.json")
os.makedirs(CKPT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # Environment
    distance   = 5,
    k          = 5,
    T          = 500,
    p_data     = 0.001,
    p_meas     = 0.0,
    R_success  = 25.0,
    R_failure  = 25.0,

    # Agent
    lr              = 1e-5,
    lr_conv1        = None,       # set to e.g. 1e-5 if warm-starting conv
    grad_clip       = 1.0,
    gamma           = 0.99,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    epsilon_decay   = 500_000,    # in agent steps (= rounds), long horizon
    batch_size      = 64,
    target_update   = 5_000,
    buffer_capacity = 100_000,
    identity_bias   = 0.97,        # exploration: P(Identity) per head when exploring

    # Training
    n_episodes      = 60_000,
    warmup_episodes = 50,         # episodes can be long → fewer warmup eps
    train_every     = 1,          # train_step every N env steps
    log_interval    = 50,
    save_interval   = 1_000,
    eval_interval   = 500,
    eval_episodes   = 50,

    # Wandb
    project   = "surface-code-rl-d5-realistic",
    run_name  = "d5-continuous-multidiscrete",
    device    = "cpu",
)


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            return json.load(f)
    return {}

def save_meta(episode, run_id, best_survival, global_step):
    with open(META_FILE, "w") as f:
        json.dump({
            "episode": episode, "wandb_run_id": run_id,
            "best_survival": best_survival, "global_step": global_step,
        }, f)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION  (greedy)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent, env, n_eval: int = 50) -> dict:
    survivals      = 0       # reached T (truncated, alive)
    logical_deaths = 0       # terminated (logical failure)
    total_len      = 0
    total_reward   = 0.0

    for _ in range(n_eval):
        obs, _ = env.reset()
        episode_over = False
        ep_len = 0
        ep_r   = 0.0
        while not episode_over:
            action = agent.select_action(obs, greedy=True)
            obs, r, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            ep_len += 1
            ep_r   += float(np.sum(r))
        total_len    += ep_len
        total_reward += ep_r
        if truncated and not terminated:
            survivals += 1
        if terminated:
            logical_deaths += 1

    return {
        "eval/survival_rate":      survivals      / n_eval,
        "eval/logical_death_rate": logical_deaths / n_eval,
        "eval/avg_episode_length": total_len      / n_eval,
        "eval/avg_reward":         total_reward   / n_eval,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(config: dict = None, resume: bool = False):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── Resume detection ──────────────────────────────────────────────────────
    start_episode = 1
    best_survival = 0.0
    global_step   = 0
    meta          = {}

    if resume and os.path.exists(LATEST_CKPT) and os.path.exists(META_FILE):
        meta = load_meta()
        start_episode = meta.get("episode", 0) + 1
        best_survival = meta.get("best_survival", 0.0)
        global_step   = meta.get("global_step", 0)
        print(f"\n[Resume] Continuing from episode {start_episode}")
        print(f"[Resume] Best survival: {best_survival:.4f}")
        print(f"[Resume] Global step:   {global_step:,}")
    elif resume:
        print("\n[Resume] No checkpoint found — starting fresh")
        resume = False

    # ── Wandb (separate realistic project) ────────────────────────────────────
    wandb_kwargs = dict(project=cfg["project"], name=cfg["run_name"], config=cfg)
    if resume and "wandb_run_id" in meta:
        wandb_kwargs["id"]     = meta["wandb_run_id"]
        wandb_kwargs["resume"] = "must"
        print(f"[Resume] Resuming wandb run: {meta['wandb_run_id']}\n")
    run = wandb.init(**wandb_kwargs)

    print(f"\n{'═'*66}")
    print(f"  Realistic d=5 Surface Code — Continuing Task")
    print(f"{'═'*66}")
    print(f"  noise          : p_data={cfg['p_data']}  p_meas={cfg['p_meas']}")
    print(f"  horizon T      : {cfg['T']}  rounds/episode")
    print(f"  obs window k   : {cfg['k']}")
    print(f"  episodes       : {start_episode} → {cfg['n_episodes']}")
    print(f"  reward         : ±{cfg['R_success']}  (split per qubit)")
    print(f"  checkpoint dir : checkpoints_d5_continuous/")
    print(f"  wandb project  : {cfg['project']}")
    print(f"{'═'*66}\n")

    # ── Environment ───────────────────────────────────────────────────────────
    env = SurfaceCodeRealisticEnv(
        distance=cfg["distance"], k=cfg["k"], T=cfg["T"],
        p_data=cfg["p_data"], p_meas=cfg["p_meas"],
        R_success=cfg["R_success"], R_failure=cfg["R_failure"],
    )
    env_eval = SurfaceCodeRealisticEnv(
        distance=cfg["distance"], k=cfg["k"], T=cfg["T"],
        p_data=cfg["p_data"], p_meas=cfg["p_meas"],
        R_success=cfg["R_success"], R_failure=cfg["R_failure"],
    )

    in_channels = env.observation_space.shape[0]   # 15
    grid_size   = env.observation_space.shape[1]   # 11
    n_qubits    = env.n_qubits                      # 25

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = MultiDiscreteDDQNAgent(
        in_channels=in_channels, grid_size=grid_size, n_qubits=n_qubits,
        n_per_qubit=4,
        lr=cfg["lr"], lr_conv1=cfg["lr_conv1"], grad_clip=cfg["grad_clip"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"], epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"], batch_size=cfg["batch_size"],
        target_update=cfg["target_update"], buffer_capacity=cfg["buffer_capacity"],
        identity_bias=cfg["identity_bias"],
        device=cfg["device"],
    )

    if resume and os.path.exists(LATEST_CKPT):
        agent.load(LATEST_CKPT)
        print(f"[Resume] Agent loaded — ε={agent.epsilon:.4f}  steps={agent._steps_done:,}\n")

    agent.summary()
    wandb.watch(agent.online_net, log="all", log_freq=1000)

    # ── Warmup buffer (Identity-biased random policy) ─────────────────────────
    warmup_eps = cfg["warmup_episodes"] if not resume else max(5, cfg["warmup_episodes"]//2)
    print(f"[Warmup] Running {warmup_eps} Identity-biased random episodes...")
    saved_eps = agent.epsilon
    agent.epsilon = 1.0   # force full exploration (but Identity-biased)
    for _ in range(warmup_eps):
        obs, _ = env.reset()
        episode_over = False
        while not episode_over:
            action = agent.select_action(obs)   # uses Identity-biased exploration
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            # CONTINUING TASK: store done = terminated ONLY (not truncated)
            agent.push(obs, action, reward, next_obs, float(terminated))
            obs = next_obs
    agent.epsilon = saved_eps
    # Reset the step counter so warmup doesn't consume the ε schedule
    agent._steps_done = 0
    print(f"[Warmup] Buffer: {len(agent.buffer):,}\n")

    # ── Rolling windows ───────────────────────────────────────────────────────
    W = cfg["log_interval"]
    survival_window = deque(maxlen=W)
    death_window    = deque(maxlen=W)
    length_window   = deque(maxlen=W)
    reward_window   = deque(maxlen=W)
    loss_window     = deque(maxlen=W)

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode in range(start_episode, cfg["n_episodes"] + 1):

        obs, _ = env.reset()
        episode_over = False
        ep_len    = 0
        ep_reward = 0.0
        ep_losses = []

        while not episode_over:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

            # ── CONTINUING TASK done handling ─────────────────────────────────
            #   terminated (logical death) → done=1.0 → bootstrap cut
            #   truncated  (T reached)     → done=0.0 → bootstrap kept
            done_for_buffer = float(terminated)      # NOT (terminated or truncated)

            agent.push(obs, action, reward, next_obs, done_for_buffer)

            if global_step % cfg["train_every"] == 0:
                loss = agent.train_step()
                if loss is not None: ep_losses.append(loss)

            ep_reward += float(np.sum(reward))
            ep_len    += 1
            obs        = next_obs
            global_step += 1

        # ── Episode stats ─────────────────────────────────────────────────────
        survived    = bool(truncated and not terminated)
        died        = bool(terminated)
        avg_loss    = float(np.mean(ep_losses)) if ep_losses else 0.0

        survival_window.append(float(survived))
        death_window.append(float(died))
        length_window.append(ep_len)
        reward_window.append(ep_reward)
        loss_window.append(avg_loss)

        # ── Wandb log ─────────────────────────────────────────────────────────
        wandb.log({
            "train/episode_reward":   ep_reward,
            "train/episode_length":   ep_len,
            "train/survived":         float(survived),
            "train/logical_death":    float(died),
            "train/loss":             avg_loss,
            "agent/epsilon":          agent.epsilon,
            "agent/buffer_size":      len(agent.buffer),
            "agent/train_steps":      agent._train_steps,
            "rolling/survival_rate":  np.mean(survival_window),
            "rolling/death_rate":     np.mean(death_window),
            "rolling/avg_length":     np.mean(length_window),
            "rolling/avg_reward":     np.mean(reward_window),
            "rolling/avg_loss":       np.mean(loss_window),
            "episode":     episode,
            "global_step": global_step,
        }, step=episode)

        # ── Crash-safe save every episode ─────────────────────────────────────
        agent.save(LATEST_CKPT)
        save_meta(episode, run.id, best_survival, global_step)

        # ── Console log ───────────────────────────────────────────────────────
        if episode % cfg["log_interval"] == 0:
            sr  = np.mean(survival_window)
            dr  = np.mean(death_window)
            al  = np.mean(length_window)
            avr = np.mean(reward_window)
            print(
                f"Ep {episode:>6}/{cfg['n_episodes']} | "
                f"Survive={sr:.3f} | Death={dr:.3f} | "
                f"Len={al:6.1f} | Reward={avr:+8.2f} | "
                f"Loss={np.mean(loss_window):.5f} | ε={agent.epsilon:.3f}"
            )

        # ── Eval ──────────────────────────────────────────────────────────────
        if episode % cfg["eval_interval"] == 0:
            em = evaluate(agent, env_eval, n_eval=cfg["eval_episodes"])
            wandb.log(em, step=episode)
            esr = em["eval/survival_rate"]
            print(f"         └─ Eval({cfg['eval_episodes']}): "
                  f"survival={esr:.3f}  death={em['eval/logical_death_rate']:.3f}  "
                  f"len={em['eval/avg_episode_length']:.1f}")
            if esr > best_survival:
                best_survival = esr
                best_path = os.path.join(CKPT_DIR, "d5_best.pt")
                agent.save(best_path)
                wandb.run.summary["best_survival_rate"] = best_survival
                wandb.run.summary["best_episode"]       = episode
                save_meta(episode, run.id, best_survival, global_step)
                print(f"         ★ New best survival: {best_survival:.4f} → {best_path}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % cfg["save_interval"] == 0:
            ckpt = os.path.join(CKPT_DIR, f"d5_ep{episode}.pt")
            agent.save(ckpt)
            print(f"[Checkpoint] Saved → {ckpt}")

    # ── Final save + eval ─────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR, "d5_final.pt")
    agent.save(final_path)

    print("\n[Eval] Final evaluation — 200 episodes (greedy)...")
    fm = evaluate(agent, env_eval, n_eval=200)
    wandb.log({**fm, "episode": cfg["n_episodes"]})

    print(f"\n{'═'*66}")
    print(f"  REALISTIC d=5 TRAINING COMPLETE")
    print(f"{'═'*66}")
    print(f"  Final survival rate : {fm['eval/survival_rate']:.4f}")
    print(f"  Logical death rate  : {fm['eval/logical_death_rate']:.4f}")
    print(f"  Avg episode length  : {fm['eval/avg_episode_length']:.1f} / {cfg['T']}")
    print(f"  Best survival rate  : {best_survival:.4f}")
    print(f"  Final checkpoint    : {final_path}")
    print(f"{'═'*66}\n")

    if os.path.exists(LATEST_CKPT): os.remove(LATEST_CKPT)
    if os.path.exists(META_FILE):   os.remove(META_FILE)

    wandb.finish()
    return agent, fm


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train realistic d=5 continuing DDQN")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--episodes",        type=int,   default=DEFAULT_CONFIG["n_episodes"])
    parser.add_argument("--p_data",          type=float, default=DEFAULT_CONFIG["p_data"])
    parser.add_argument("--p_meas",          type=float, default=DEFAULT_CONFIG["p_meas"])
    parser.add_argument("--T",               type=int,   default=DEFAULT_CONFIG["T"])
    parser.add_argument("--lr",              type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--lr_conv1",        type=float, default=DEFAULT_CONFIG["lr_conv1"])
    parser.add_argument("--grad_clip",       type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--gamma",           type=float, default=DEFAULT_CONFIG["gamma"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--buffer_capacity", type=int,   default=DEFAULT_CONFIG["buffer_capacity"])
    parser.add_argument("--epsilon_decay",   type=int,   default=DEFAULT_CONFIG["epsilon_decay"])
    parser.add_argument("--target_update",   type=int,   default=DEFAULT_CONFIG["target_update"])
    parser.add_argument("--warmup",          type=int,   default=DEFAULT_CONFIG["warmup_episodes"])
    parser.add_argument("--log_interval",    type=int,   default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--save_interval",   type=int,   default=DEFAULT_CONFIG["save_interval"])
    parser.add_argument("--eval_interval",   type=int,   default=DEFAULT_CONFIG["eval_interval"])
    parser.add_argument("--device",          type=str,   default=DEFAULT_CONFIG["device"])
    parser.add_argument("--run_name",        type=str,   default=DEFAULT_CONFIG["run_name"])
    parser.add_argument("--project",         type=str,   default=DEFAULT_CONFIG["project"])
    args = parser.parse_args()

    train(
        config={
            "n_episodes":      args.episodes,
            "p_data":          args.p_data,
            "p_meas":          args.p_meas,
            "T":               args.T,
            "lr":              args.lr,
            "lr_conv1":        args.lr_conv1,
            "grad_clip":       args.grad_clip,
            "gamma":           args.gamma,
            "batch_size":      args.batch_size,
            "buffer_capacity": args.buffer_capacity,
            "epsilon_decay":   args.epsilon_decay,
            "target_update":   args.target_update,
            "warmup_episodes": args.warmup,
            "log_interval":    args.log_interval,
            "save_interval":   args.save_interval,
            "eval_interval":   args.eval_interval,
            "device":          args.device,
            "run_name":        args.run_name,
            "project":         args.project,
        },
        resume=args.resume,
    )