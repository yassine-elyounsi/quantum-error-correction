"""
Training Loop — d=5 Surface Code DDQN  (warm start from d=3)
==============================================================
Transfer learning via CONVOLUTIONAL WEIGHT WARM START:
  d=3 conv filters learn LOCAL syndrome→error patterns.
  These patterns are distance-independent, so they transfer to d=5
  across the WHOLE grid (not just the centre).

Two-level stabilization for the transfer:
  ① Warm start  — copy d=3 conv weights (zero-pad extra channels)
  ② Differential lr — Conv1 trains slow (1e-5), rest normal (3e-5)
                       + gradient clip 1.0

Features:
  • Dedicated checkpoint folder: checkpoints_stim_final_d5/
  • Crash-safe resume with --resume (continues same wandb run)
  • Separate wandb project: surface-code-rl-d5

Usage
-----
    python -m reinforcement.surface.train_d5
    python -m reinforcement.surface.train_d5 --resume
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
import torch

from reinforcement.surface5.ddqn_agent_d5 import DDQNAgentD5
from src.environments.surface5_env       import SurfaceCodeEnv

# ── Checkpoint dirs ───────────────────────────────────────────────────────────
CKPT_DIR_D3 = os.path.join(_ROOT, "checkpoints_stim_final")      # d=3 source
CKPT_DIR_D5 = os.path.join(_ROOT, "checkpoints_stim_warm_d5")   # d=5 output
LATEST_CKPT = os.path.join(CKPT_DIR_D5, "d5_latest.pt")
META_FILE   = os.path.join(CKPT_DIR_D5, "d5_meta.json")
os.makedirs(CKPT_DIR_D5, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # Environment
    distance        = 5,
    noise           = 0.01,
    syndrome_rounds = 5,
    max_steps       = 150,

    # DDQN agent (d=5) — differential lr for stability
    lr              = 3e-5,      # all layers except Conv1
    lr_conv1        = 1e-5,      # Conv1 only (slow → preserve d=3)
    grad_clip       = 1.0,       # tight clip for d=5 stability
    gamma           = 0.99,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    epsilon_decay   = 150_000,
    batch_size      = 64,
    target_update   = 1_000,
    buffer_capacity = 100_000,

    # Warm start
    d3_checkpoint   = os.path.join(CKPT_DIR_D3, "d3_best.pt"),
    use_warm_start  = True,

    # Training
    n_episodes      = 30_000,
    warmup_episodes = 500,
    log_interval    = 100,
    save_interval   = 2_000,

    # Wandb
    project         = "surface-code-rl-d5",
    run_name        = "d5-ddqn-warmstart",
    device          = "cpu",
)


# ══════════════════════════════════════════════════════════════════════════════
#  WARM START — copy d=3 conv weights into d=5 network
# ══════════════════════════════════════════════════════════════════════════════

def warm_start_d5_from_d3(agent_d5: DDQNAgentD5, d3_checkpoint_path: str):
    """
    Transfer conv weights from trained d=3 network into d=5 network.

    conv.0.weight  (32,6,3,3) → (32,10,3,3)  copy first 6 channels, zero-pad rest
    conv.0.bias, conv.1.* (BN1)              direct copy
    conv.3.* (Conv2), conv.4.* (BN2)         direct copy
    conv.6.* (Conv3), conv.7.* (BN3)         direct copy
    value_stream.*, advantage_stream.*       NOT copied (different sizes)
    """
    ckpt     = torch.load(d3_checkpoint_path, map_location="cpu")
    d3_state = ckpt["online_net"]
    d5_state = agent_d5.online_net.state_dict()

    copied = skipped = 0
    for key in d3_state:
        if key.startswith("value_stream") or key.startswith("advantage_stream"):
            skipped += 1
            continue
        if key == "conv.0.weight":
            d3_w = d3_state[key]                 # (32, 6, 3, 3)
            d5_w = d5_state[key].clone()         # (32, 10, 3, 3)
            d5_w[:, :6, :, :] = d3_w             # copy 6 known channels
            d5_w[:, 6:, :, :] = 0.0              # zero-pad extra 4
            d5_state[key] = d5_w
            copied += 1
        elif key in d5_state and d3_state[key].shape == d5_state[key].shape:
            d5_state[key] = d3_state[key]
            copied += 1
        else:
            skipped += 1

    agent_d5.online_net.load_state_dict(d5_state)
    agent_d5.target_net.load_state_dict(d5_state)
    print(f"[WarmStart] Copied {copied} conv/BN layers from d=3 → d=5")
    print(f"[WarmStart] Skipped {skipped} FC layers (random init)")
    print(f"[WarmStart] Conv1: channels 0:6 = d=3 weights, channels 6:10 = zeros")


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}

def save_meta(episode, wandb_run_id, best_success, global_step):
    with open(META_FILE, "w") as f:
        json.dump({
            "episode": episode, "wandb_run_id": wandb_run_id,
            "best_success": best_success, "global_step": global_step,
        }, f)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent, env, n_eval: int = 200) -> dict:
    successes = logical_errors = 0
    total_reward = 0.0
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += r
        total_reward += ep_r
        if info["corrected"]:     successes      += 1
        if info["logical_error"]: logical_errors += 1
    return {
        "eval/success_rate":       successes      / n_eval,
        "eval/logical_error_rate": logical_errors / n_eval,
        "eval/avg_reward":         total_reward   / n_eval,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(config: dict = None, resume: bool = False):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── Verify d=3 checkpoint if warm starting ────────────────────────────────
    d3_path = cfg["d3_checkpoint"]
    if not os.path.isabs(d3_path):
        d3_path = os.path.join(_ROOT, d3_path)
    if cfg["use_warm_start"]:
        assert os.path.exists(d3_path), \
            f"d=3 checkpoint not found: {d3_path}\nFinish d=3 training first " \
            f"or set use_warm_start=False."

    # ── Resume detection ──────────────────────────────────────────────────────
    start_episode     = 1
    best_success_rate = 0.0
    global_step       = 0
    meta              = {}

    if resume and os.path.exists(LATEST_CKPT) and os.path.exists(META_FILE):
        meta = load_meta()
        start_episode     = meta.get("episode", 0) + 1
        best_success_rate = meta.get("best_success", 0.0)
        global_step       = meta.get("global_step", 0)
        print(f"\n[Resume] Continuing from episode {start_episode}")
        print(f"[Resume] Best success: {best_success_rate:.4f}")
        print(f"[Resume] Global step: {global_step:,}")
    elif resume:
        print("\n[Resume] No checkpoint found — starting fresh")
        resume = False

    # ── Init wandb (separate d=5 project) ─────────────────────────────────────
    wandb_kwargs = dict(project=cfg["project"], name=cfg["run_name"], config=cfg)
    if resume and "wandb_run_id" in meta:
        wandb_kwargs["id"]     = meta["wandb_run_id"]
        wandb_kwargs["resume"] = "must"
        print(f"[Resume] Resuming wandb run: {meta['wandb_run_id']}\n")

    run = wandb.init(**wandb_kwargs)

    print(f"\n{'═'*65}")
    print(f"  d=5 DDQN Training (warm start from d=3)")
    print(f"{'═'*65}")
    print(f"  noise          : {cfg['noise']}")
    print(f"  episodes       : {start_episode} → {cfg['n_episodes']}")
    print(f"  lr / lr_conv1  : {cfg['lr']:.1e} / {cfg['lr_conv1']:.1e}")
    print(f"  grad_clip      : {cfg['grad_clip']}")
    print(f"  warm start     : {cfg['use_warm_start']}  ({d3_path if cfg['use_warm_start'] else '—'})")
    print(f"  checkpoint dir : checkpoints_stim_final_d5/")
    print(f"  wandb project  : {cfg['project']}")
    print(f"{'═'*65}\n")

    # ── Environment ───────────────────────────────────────────────────────────
    env = SurfaceCodeEnv(
        distance=cfg["distance"], noise=cfg["noise"],
        syndrome_rounds=cfg["syndrome_rounds"], max_steps=cfg["max_steps"],
    )
    env_eval = SurfaceCodeEnv(
        distance=cfg["distance"], noise=cfg["noise"],
        syndrome_rounds=cfg["syndrome_rounds"], max_steps=cfg["max_steps"],
    )

    in_channels = env.observation_space.shape[0]   # 10
    grid_size   = env.observation_space.shape[1]   # 11
    n_actions   = env.action_space.n               # 51

    # ── d=5 agent (differential lr) ───────────────────────────────────────────
    agent = DDQNAgentD5(
        in_channels=in_channels, grid_size=grid_size, n_actions=n_actions,
        lr=cfg["lr"], lr_conv1=cfg["lr_conv1"], grad_clip=cfg["grad_clip"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"], epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"], batch_size=cfg["batch_size"],
        target_update=cfg["target_update"], buffer_capacity=cfg["buffer_capacity"],
        device=cfg["device"],
    )

    # ── Warm start OR resume ──────────────────────────────────────────────────
    if resume and os.path.exists(LATEST_CKPT):
        agent.load(LATEST_CKPT)
        print(f"[Resume] Agent loaded — ε={agent.epsilon:.4f}  steps={agent._steps_done:,}\n")
    elif cfg["use_warm_start"]:
        warm_start_d5_from_d3(agent, d3_path)

    agent.summary()
    wandb.watch(agent.online_net, log="all", log_freq=500)

    # ── Warmup buffer ─────────────────────────────────────────────────────────
    warmup_eps = cfg["warmup_episodes"] if not resume else cfg["warmup_episodes"] // 2
    print(f"[Warmup] Running {warmup_eps} random episodes...")
    for _ in range(warmup_eps):
        obs, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            nobs, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.push(obs, a, r, nobs, done)
            obs = nobs
    print(f"[Warmup] Buffer: {len(agent.buffer):,}\n")

    # ── Rolling windows ───────────────────────────────────────────────────────
    window = cfg["log_interval"]
    reward_window     = deque(maxlen=window)
    success_window    = deque(maxlen=window)
    loss_window       = deque(maxlen=window)
    logicalerr_window = deque(maxlen=window)
    weight_window     = deque(maxlen=window)

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode in range(start_episode, cfg["n_episodes"] + 1):

        obs, _ = env.reset()
        done = False
        ep_reward  = 0.0
        ep_losses  = []
        ep_weights = []

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            agent.push(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None: ep_losses.append(loss)

            ep_reward += reward
            ep_weights.append(info["syndrome_weight"])
            obs = next_obs
            global_step += 1

        # ── Episode stats ─────────────────────────────────────────────────────
        corrected   = info["corrected"]
        logical_err = info["logical_error"]
        avg_loss    = float(np.mean(ep_losses))  if ep_losses  else 0.0
        avg_weight  = float(np.mean(ep_weights)) if ep_weights else 0.0

        reward_window.append(ep_reward)
        success_window.append(float(corrected))
        loss_window.append(avg_loss)
        logicalerr_window.append(float(logical_err))
        weight_window.append(avg_weight)

        # ── Wandb log ─────────────────────────────────────────────────────────
        wandb.log({
            "train/episode_reward":      ep_reward,
            "train/corrected":           float(corrected),
            "train/logical_error":       float(logical_err),
            "train/avg_syndrome_weight": avg_weight,
            "train/loss":                avg_loss,
            "agent/epsilon":             agent.epsilon,
            "agent/buffer_size":         len(agent.buffer),
            "agent/train_steps":         agent._train_steps,
            "rolling/success_rate":      np.mean(success_window),
            "rolling/avg_reward":        np.mean(reward_window),
            "rolling/avg_loss":          np.mean(loss_window),
            "rolling/logical_error_rate":np.mean(logicalerr_window),
            "rolling/avg_weight":        np.mean(weight_window),
            "episode": episode,
            "global_step": global_step,
        }, step=episode)

        # ── Crash-safe save every episode ─────────────────────────────────────
        agent.save(LATEST_CKPT)
        save_meta(episode, run.id, best_success_rate, global_step)

        # ── Console + eval ────────────────────────────────────────────────────
        if episode % cfg["log_interval"] == 0:
            sr  = np.mean(success_window)
            ler = np.mean(logicalerr_window)
            avr = np.mean(reward_window)
            avl = np.mean(loss_window)
            print(
                f"Ep {episode:>6}/{cfg['n_episodes']} | "
                f"Success={sr:.3f} | LogErr={ler:.3f} | "
                f"Reward={avr:+.2f} | Loss={avl:.5f} | ε={agent.epsilon:.3f}"
            )
            eval_metrics = evaluate(agent, env_eval, n_eval=200)
            wandb.log(eval_metrics, step=episode)
            eval_sr = eval_metrics["eval/success_rate"]
            print(f"         └─ Eval(200): success={eval_sr:.3f}  "
                  f"logical={eval_metrics['eval/logical_error_rate']:.3f}")

            if eval_sr > best_success_rate:
                best_success_rate = eval_sr
                best_path = os.path.join(CKPT_DIR_D5, "d5_best.pt")
                agent.save(best_path)
                wandb.run.summary["best_success_rate"] = best_success_rate
                wandb.run.summary["best_episode"]      = episode
                save_meta(episode, run.id, best_success_rate, global_step)
                print(f"         ★ New best: {best_success_rate:.4f} → {best_path}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % cfg["save_interval"] == 0:
            ckpt = os.path.join(CKPT_DIR_D5, f"d5_ep{episode}.pt")
            agent.save(ckpt)
            print(f"[Checkpoint] Saved → {ckpt}")

    # ── Final save + eval ─────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR_D5, "d5_final.pt")
    agent.save(final_path)

    print("\n[Eval] Final evaluation — 500 episodes (greedy)...")
    final_metrics = evaluate(agent, env_eval, n_eval=500)
    wandb.log({**final_metrics, "episode": cfg["n_episodes"]})

    print(f"\n{'═'*65}")
    print(f"  d=5 TRAINING COMPLETE")
    print(f"{'═'*65}")
    print(f"  Final success rate : {final_metrics['eval/success_rate']:.4f}")
    print(f"  Logical error rate : {final_metrics['eval/logical_error_rate']:.4f}")
    print(f"  Best success rate  : {best_success_rate:.4f}")
    print(f"  Final checkpoint   : {final_path}")
    print(f"{'═'*65}\n")

    if os.path.exists(LATEST_CKPT): os.remove(LATEST_CKPT)
    if os.path.exists(META_FILE):   os.remove(META_FILE)

    wandb.finish()
    return agent, final_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train d=5 DDQN with warm start")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--episodes",        type=int,   default=DEFAULT_CONFIG["n_episodes"])
    parser.add_argument("--noise",           type=float, default=DEFAULT_CONFIG["noise"])
    parser.add_argument("--lr",              type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--lr_conv1",        type=float, default=DEFAULT_CONFIG["lr_conv1"])
    parser.add_argument("--grad_clip",       type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--buffer_capacity", type=int,   default=DEFAULT_CONFIG["buffer_capacity"])
    parser.add_argument("--epsilon_decay",   type=int,   default=DEFAULT_CONFIG["epsilon_decay"])
    parser.add_argument("--max_steps",       type=int,   default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--warmup",          type=int,   default=DEFAULT_CONFIG["warmup_episodes"])
    parser.add_argument("--d3_checkpoint",   type=str,   default=DEFAULT_CONFIG["d3_checkpoint"])
    parser.add_argument("--no_warm_start",   action="store_true",
                        help="Train from scratch (no d=3 transfer)")
    parser.add_argument("--log_interval",    type=int,   default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--save_interval",   type=int,   default=DEFAULT_CONFIG["save_interval"])
    parser.add_argument("--device",          type=str,   default=DEFAULT_CONFIG["device"])
    parser.add_argument("--run_name",        type=str,   default=DEFAULT_CONFIG["run_name"])
    parser.add_argument("--project",         type=str,   default=DEFAULT_CONFIG["project"])
    args = parser.parse_args()

    train(
        config={
            "n_episodes":      args.episodes,
            "noise":           args.noise,
            "lr":              args.lr,
            "lr_conv1":        args.lr_conv1,
            "grad_clip":       args.grad_clip,
            "batch_size":      args.batch_size,
            "buffer_capacity": args.buffer_capacity,
            "epsilon_decay":   args.epsilon_decay,
            "max_steps":       args.max_steps,
            "warmup_episodes": args.warmup,
            "d3_checkpoint":   args.d3_checkpoint,
            "use_warm_start":  not args.no_warm_start,
            "log_interval":    args.log_interval,
            "save_interval":   args.save_interval,
            "device":          args.device,
            "run_name":        args.run_name,
            "project":         args.project,
        },
        resume=args.resume,
    )