# """
# Training Loop — d=3 Surface Code DDQN
# =======================================
# Trains a DDQN agent to decode the distance-3 surface code.
# Logs all metrics to Weights & Biases for live visualization.

# Usage
# -----
# From quantum_rl root:
#     python -m reinforcement.surface.train

# Or with custom args:
#     python -m reinforcement.surface.train --episodes 20000 --noise 0.01

# After training the agent is saved to:
#     checkpoints/d3_final.pt

# This checkpoint is later loaded into the PPR policy library
# when training the d=5 agent.
# """

# import os
# import sys
# import argparse
# import numpy as np
# from collections import deque

# # ── Path setup ────────────────────────────────────────────────────────────────
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)

# import wandb

# from reinforcement.surface.ddqn_agent import DDQNAgent
# from src.environments.surface3_env   import SurfaceCodeEnv

# # ── Checkpoint directory ──────────────────────────────────────────────────────
# CKPT_DIR = os.path.join(_ROOT, "checkpoints")
# os.makedirs(CKPT_DIR, exist_ok=True)


# # ══════════════════════════════════════════════════════════════════════════════
# #  DEFAULT HYPERPARAMETERS
# # ══════════════════════════════════════════════════════════════════════════════

# DEFAULT_CONFIG = dict(
#     # Environment
#     distance        = 3,
#     noise           = 0.01,
#     syndrome_rounds = 3,        # k = d
#     max_steps       = 80,       # max correction steps per episode

#     # Agent
#     lr              = 1e-4,
#     gamma           = 0.99,
#     epsilon_start   = 1.0,
#     epsilon_end     = 0.05,
#     epsilon_decay   = 100_000,   # env steps to decay ε from 1.0 → 0.05
#     batch_size      = 64,
#     target_update   = 1_000,    # train steps between target net syncs
#     buffer_capacity = 50_000,

#     # Training
#     n_episodes      = 20_000,
#     warmup_episodes = 500,      # random episodes to pre-fill buffer
#     log_interval    = 100,      # episodes between console + wandb logs
#     save_interval   = 2_000,    # episodes between periodic checkpoints

#     # Wandb
#     project         = "surface-code-rl",
#     run_name        = "d3-ddqn-stim-v3",
#     device          = "cpu",
# )


# # ══════════════════════════════════════════════════════════════════════════════
# #  EVALUATION
# # ══════════════════════════════════════════════════════════════════════════════

# def evaluate(agent: DDQNAgent, env: SurfaceCodeEnv, n_eval: int = 200) -> dict:
#     """
#     Run n_eval episodes with greedy policy (no exploration).
#     Returns dict of evaluation metrics ready to log to wandb.
#     """
#     successes      = 0
#     logical_errors = 0
#     total_reward   = 0.0
#     final_weights  = []

#     for _ in range(n_eval):
#         obs, _    = env.reset()
#         done      = False
#         ep_reward = 0.0

#         while not done:
#             action                         = agent.select_action(obs, greedy=True)
#             obs, reward, term, trunc, info = env.step(action)
#             done                           = term or trunc
#             ep_reward                     += reward

#         total_reward  += ep_reward
#         final_weights.append(info["syndrome_weight"])

#         if info["corrected"]:
#             successes += 1
#         if info["logical_error"]:
#             logical_errors += 1

#     return {
#         "eval/success_rate":       successes      / n_eval,
#         "eval/logical_error_rate": logical_errors / n_eval,
#         "eval/avg_reward":         total_reward   / n_eval,
#         "eval/avg_final_weight":   float(np.mean(final_weights)),
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN TRAINING FUNCTION
# # ══════════════════════════════════════════════════════════════════════════════

# def train(config: dict = None):
#     cfg = {**DEFAULT_CONFIG, **(config or {})}

#     # ── Init wandb ────────────────────────────────────────────────────────────
#     wandb.init(
#         project = cfg["project"],
#         name    = cfg["run_name"],
#         config  = cfg,
#     )

#     print(f"\n{'═'*60}")
#     print(f"  DDQN Training — distance={cfg['distance']}  noise={cfg['noise']}")
#     print(f"  episodes={cfg['n_episodes']}  device={cfg['device']}")
#     print(f"  wandb project : {cfg['project']}")
#     print(f"  wandb run     : {cfg['run_name']}")
#     print(f"{'═'*60}\n")

#     # ── Environment ───────────────────────────────────────────────────────────
#     env = SurfaceCodeEnv(
#         distance        = cfg["distance"],
#         noise           = cfg["noise"],
#         syndrome_rounds = cfg["syndrome_rounds"],
#         max_steps       = cfg["max_steps"],
#     )
#     env_eval = SurfaceCodeEnv(
#         distance        = cfg["distance"],
#         noise           = cfg["noise"],
#         syndrome_rounds = cfg["syndrome_rounds"],
#         max_steps       = cfg["max_steps"],
#     )

#     # ── Agent ─────────────────────────────────────────────────────────────────
#     in_channels = env.observation_space.shape[0]   # 2k = 6
#     grid_size   = env.observation_space.shape[1]   # 2d+1 = 7
#     n_actions   = env.action_space.n               # 19

#     agent = DDQNAgent(
#         in_channels     = in_channels,
#         grid_size       = grid_size,
#         n_actions       = n_actions,
#         lr              = cfg["lr"],
#         gamma           = cfg["gamma"],
#         epsilon_start   = cfg["epsilon_start"],
#         epsilon_end     = cfg["epsilon_end"],
#         epsilon_decay   = cfg["epsilon_decay"],
#         batch_size      = cfg["batch_size"],
#         target_update   = cfg["target_update"],
#         buffer_capacity = cfg["buffer_capacity"],
#         device          = cfg["device"],
#     )
#     agent.summary()

#     # Watch gradients and weights in wandb
#     wandb.watch(agent.online_net, log="all", log_freq=500)

#     # ── Warmup — fill buffer with random transitions ───────────────────────────
#     print(f"[Warmup] Running {cfg['warmup_episodes']} random episodes...")
#     for _ in range(cfg["warmup_episodes"]):
#         obs, _ = env.reset()
#         done   = False
#         while not done:
#             action                         = env.action_space.sample()
#             next_obs, r, term, trunc, info = env.step(action)
#             done                           = term or trunc
#             agent.push(obs, action, r, next_obs, done)
#             obs = next_obs

#     print(f"[Warmup] Buffer: {len(agent.buffer):,} transitions  "
#           f"Ready: {agent.buffer.is_ready_for(cfg['batch_size'])}\n")

#     # ── Training loop ─────────────────────────────────────────────────────────
#     window            = cfg["log_interval"]
#     reward_window     = deque(maxlen=window)
#     success_window    = deque(maxlen=window)
#     loss_window       = deque(maxlen=window)
#     weight_window     = deque(maxlen=window)
#     logicalerr_window = deque(maxlen=window)

#     best_success_rate = 0.0
#     global_step       = 0

#     for episode in range(1, cfg["n_episodes"] + 1):

#         obs, _ = env.reset()
#         done   = False

#         ep_reward  = 0.0
#         ep_losses  = []
#         ep_weights = []

#         # ── Episode ───────────────────────────────────────────────────────────
#         while not done:
#             action                              = agent.select_action(obs)
#             next_obs, reward, term, trunc, info = env.step(action)
#             done                                = term or trunc

#             agent.push(obs, action, reward, next_obs, done)
#             loss = agent.train_step()

#             if loss is not None:
#                 ep_losses.append(loss)

#             ep_reward  += reward
#             ep_weights.append(info["syndrome_weight"])
#             obs         = next_obs
#             global_step += 1

#         # ── Episode stats ─────────────────────────────────────────────────────
#         corrected   = info["corrected"]
#         logical_err = info["logical_error"]
#         avg_loss    = float(np.mean(ep_losses))  if ep_losses  else 0.0
#         avg_weight  = float(np.mean(ep_weights)) if ep_weights else 0.0

#         reward_window.append(ep_reward)
#         success_window.append(float(corrected))
#         loss_window.append(avg_loss)
#         weight_window.append(avg_weight)
#         logicalerr_window.append(float(logical_err))

#         # ── Log every episode to wandb ─────────────────────────────────────────
#         wandb.log({
#             # Raw episode metrics
#             "train/episode_reward":      ep_reward,
#             "train/corrected":           float(corrected),
#             "train/logical_error":       float(logical_err),
#             "train/avg_syndrome_weight": avg_weight,
#             "train/loss":                avg_loss,

#             # Agent state
#             "agent/epsilon":             agent.epsilon,
#             "agent/buffer_size":         len(agent.buffer),
#             "agent/train_steps":         agent._train_steps,

#             # Rolling averages (smooth curves in wandb)
#             "rolling/success_rate":      np.mean(success_window),
#             "rolling/avg_reward":        np.mean(reward_window),
#             "rolling/avg_loss":          np.mean(loss_window),
#             "rolling/logical_error_rate":np.mean(logicalerr_window),
#             "rolling/avg_weight":        np.mean(weight_window),

#             "episode":     episode,
#             "global_step": global_step,
#         }, step=episode)

#         # ── Console + evaluation every log_interval episodes ──────────────────
#         if episode % cfg["log_interval"] == 0:
#             sr  = np.mean(success_window)
#             ler = np.mean(logicalerr_window)
#             avr = np.mean(reward_window)
#             avl = np.mean(loss_window)

#             print(
#                 f"Ep {episode:>6}/{cfg['n_episodes']} | "
#                 f"Success={sr:.3f} | "
#                 f"LogErr={ler:.3f} | "
#                 f"Reward={avr:+.2f} | "
#                 f"Loss={avl:.5f} | "
#                 f"ε={agent.epsilon:.3f} | "
#                 f"Buffer={len(agent.buffer):,}"
#             )

#             # Evaluation run (greedy policy)
#             eval_metrics = evaluate(agent, env_eval, n_eval=200)
#             wandb.log(eval_metrics, step=episode)

#             eval_sr = eval_metrics["eval/success_rate"]
#             print(f"         └─ Eval(200): "
#                   f"success={eval_sr:.3f}  "
#                   f"logical_err={eval_metrics['eval/logical_error_rate']:.3f}  "
#                   f"reward={eval_metrics['eval/avg_reward']:+.2f}")

#             # Save best model
#             if eval_sr > best_success_rate:
#                 best_success_rate = eval_sr
#                 best_path = os.path.join(CKPT_DIR, "d3_best.pt")
#                 agent.save(best_path)
#                 wandb.run.summary["best_success_rate"] = best_success_rate
#                 wandb.run.summary["best_episode"]      = episode
#                 print(f"         ★ New best: {best_success_rate:.4f} → {best_path}")

#         # ── Periodic checkpoint ───────────────────────────────────────────────
#         if episode % cfg["save_interval"] == 0:
#             ckpt_path = os.path.join(CKPT_DIR, f"d3_ep{episode}.pt")
#             agent.save(ckpt_path)

#     # ── Final save ────────────────────────────────────────────────────────────
#     final_path = os.path.join(CKPT_DIR, "d3_final.pt")
#     agent.save(final_path)

#     # ── Final evaluation ──────────────────────────────────────────────────────
#     print("\n[Eval] Final evaluation — 500 episodes (greedy policy)...")
#     final_metrics = evaluate(agent, env_eval, n_eval=500)
#     wandb.log({**final_metrics, "episode": cfg["n_episodes"]})

#     print(f"\n{'═'*60}")
#     print(f"  TRAINING COMPLETE")
#     print(f"{'═'*60}")
#     print(f"  Final success rate   : {final_metrics['eval/success_rate']:.4f}")
#     print(f"  Logical error rate   : {final_metrics['eval/logical_error_rate']:.4f}")
#     print(f"  Avg reward           : {final_metrics['eval/avg_reward']:+.4f}")
#     print(f"  Best success rate    : {best_success_rate:.4f}")
#     print(f"  Final checkpoint     : {final_path}")
#     print(f"{'═'*60}\n")

#     wandb.finish()
#     return agent, final_metrics


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLI
# # ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train DDQN agent for d=3 surface code decoding"
#     )
#     parser.add_argument("--episodes",        type=int,   default=DEFAULT_CONFIG["n_episodes"])
#     parser.add_argument("--noise",           type=float, default=DEFAULT_CONFIG["noise"])
#     parser.add_argument("--lr",              type=float, default=DEFAULT_CONFIG["lr"])
#     parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
#     parser.add_argument("--buffer_capacity", type=int,   default=DEFAULT_CONFIG["buffer_capacity"])
#     parser.add_argument("--epsilon_decay",   type=int,   default=DEFAULT_CONFIG["epsilon_decay"])
#     parser.add_argument("--target_update",   type=int,   default=DEFAULT_CONFIG["target_update"])
#     parser.add_argument("--max_steps",       type=int,   default=DEFAULT_CONFIG["max_steps"])
#     parser.add_argument("--warmup",          type=int,   default=DEFAULT_CONFIG["warmup_episodes"])
#     parser.add_argument("--log_interval",    type=int,   default=DEFAULT_CONFIG["log_interval"])
#     parser.add_argument("--save_interval",   type=int,   default=DEFAULT_CONFIG["save_interval"])
#     parser.add_argument("--device",          type=str,   default=DEFAULT_CONFIG["device"])
#     parser.add_argument("--run_name",        type=str,   default=DEFAULT_CONFIG["run_name"])
#     parser.add_argument("--project",         type=str,   default=DEFAULT_CONFIG["project"])
#     args = parser.parse_args()

#     train({
#         "n_episodes":       args.episodes,
#         "noise":            args.noise,
#         "lr":               args.lr,
#         "batch_size":       args.batch_size,
#         "buffer_capacity":  args.buffer_capacity,
#         "epsilon_decay":    args.epsilon_decay,
#         "target_update":    args.target_update,
#         "max_steps":        args.max_steps,
#         "warmup_episodes":  args.warmup,
#         "log_interval":     args.log_interval,
#         "save_interval":    args.save_interval,
#         "device":           args.device,
#         "run_name":         args.run_name,
#         "project":          args.project,
#     })
"""
Training Loop — d=3 Surface Code DDQN  (run: stim_final)
==========================================================
Self-contained training run with:
  • Dedicated checkpoint folder: checkpoints_stim_final/
  • Crash-safe resume: continues from the last episode if interrupted
  • Continuous wandb logging: same run resumed → full plot from episode 0

Usage
-----
    # Fresh start
    python -m reinforcement.surface.train

    # Resume after a crash (auto-detects last checkpoint)
    python -m reinforcement.surface.train --resume

Checkpoints saved to:
    checkpoints_stim_final/
        d3_best.pt          ← best eval success so far
        d3_final.pt         ← end of training
        d3_ep{N}.pt         ← periodic (every save_interval)
        d3_latest.pt        ← overwritten every episode (crash recovery)
        d3_meta.json        ← episode + wandb run id + best score
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import deque

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import wandb

from reinforcement.surface.ddqn_agent import DDQNAgent
from src.environments.surface3_env   import SurfaceCodeEnv

# ── DEDICATED CHECKPOINT DIRECTORY for this run ───────────────────────────────
CKPT_DIR     = os.path.join(_ROOT, "checkpoints_stim_final")
LATEST_CKPT  = os.path.join(CKPT_DIR, "d3_latest.pt")     # overwritten every episode
META_FILE    = os.path.join(CKPT_DIR, "d3_meta.json")     # resume metadata
os.makedirs(CKPT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    distance        = 3,
    noise           = 0.01,
    syndrome_rounds = 3,
    max_steps       = 80,

    lr              = 1e-4,
    gamma           = 0.99,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    epsilon_decay   = 100_000,
    batch_size      = 64,
    target_update   = 1_000,
    buffer_capacity = 50_000,

    n_episodes      = 20_000,
    warmup_episodes = 500,
    log_interval    = 100,
    save_interval   = 2_000,

    project         = "surface-code-rl",
    run_name        = "d3-stim-final",
    device          = "cpu",
)


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}


def save_meta(episode: int, wandb_run_id: str, best_success: float, global_step: int):
    with open(META_FILE, "w") as f:
        json.dump({
            "episode":       episode,
            "wandb_run_id":  wandb_run_id,
            "best_success":  best_success,
            "global_step":   global_step,
        }, f)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent: DDQNAgent, env: SurfaceCodeEnv, n_eval: int = 200) -> dict:
    successes      = 0
    logical_errors = 0
    total_reward   = 0.0
    final_weights  = []

    for _ in range(n_eval):
        obs, _    = env.reset()
        done      = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
        total_reward  += ep_reward
        final_weights.append(info["syndrome_weight"])
        if info["corrected"]:     successes      += 1
        if info["logical_error"]: logical_errors += 1

    return {
        "eval/success_rate":       successes      / n_eval,
        "eval/logical_error_rate": logical_errors / n_eval,
        "eval/avg_reward":         total_reward   / n_eval,
        "eval/avg_final_weight":   float(np.mean(final_weights)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(config: dict = None, resume: bool = False):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

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
        print(f"\n[Resume] Found checkpoint: {LATEST_CKPT}")
        print(f"[Resume] Continuing from episode {start_episode}")
        print(f"[Resume] Best success so far: {best_success_rate:.4f}")
        print(f"[Resume] Global step: {global_step:,}")
    elif resume:
        print("\n[Resume] No checkpoint found in checkpoints_stim_final/ — starting fresh")
        resume = False

    # ── Init wandb (resume same run for continuous plot) ──────────────────────
    wandb_kwargs = dict(project=cfg["project"], name=cfg["run_name"], config=cfg)
    if resume and "wandb_run_id" in meta:
        wandb_kwargs["id"]     = meta["wandb_run_id"]
        wandb_kwargs["resume"] = "must"
        print(f"[Resume] Resuming wandb run: {meta['wandb_run_id']}\n")

    run = wandb.init(**wandb_kwargs)

    print(f"\n{'═'*60}")
    print(f"  DDQN Training — d={cfg['distance']}  noise={cfg['noise']}")
    print(f"  checkpoint dir : checkpoints_stim_final/")
    print(f"  episodes       : {start_episode} → {cfg['n_episodes']}")
    print(f"  wandb run      : {cfg['run_name']}  (id={run.id})")
    print(f"{'═'*60}\n")

    # ── Environment ───────────────────────────────────────────────────────────
    env = SurfaceCodeEnv(
        distance=cfg["distance"], noise=cfg["noise"],
        syndrome_rounds=cfg["syndrome_rounds"], max_steps=cfg["max_steps"],
    )
    env_eval = SurfaceCodeEnv(
        distance=cfg["distance"], noise=cfg["noise"],
        syndrome_rounds=cfg["syndrome_rounds"], max_steps=cfg["max_steps"],
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    in_channels = env.observation_space.shape[0]
    grid_size   = env.observation_space.shape[1]
    n_actions   = env.action_space.n

    agent = DDQNAgent(
        in_channels=in_channels, grid_size=grid_size, n_actions=n_actions,
        lr=cfg["lr"], gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"], epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"], batch_size=cfg["batch_size"],
        target_update=cfg["target_update"], buffer_capacity=cfg["buffer_capacity"],
        device=cfg["device"],
    )

    # Load weights if resuming
    if resume and os.path.exists(LATEST_CKPT):
        agent.load(LATEST_CKPT)
        print(f"[Resume] Agent loaded — ε={agent.epsilon:.4f}  steps={agent._steps_done:,}\n")

    agent.summary()
    wandb.watch(agent.online_net, log="all", log_freq=500)

    # ── Warmup buffer (always — buffer is not saved) ──────────────────────────
    warmup_eps = cfg["warmup_episodes"] if not resume else cfg["warmup_episodes"] // 2
    print(f"[Warmup] Running {warmup_eps} random episodes to fill buffer...")
    for _ in range(warmup_eps):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            agent.push(obs, action, r, next_obs, done)
            obs = next_obs
    print(f"[Warmup] Buffer: {len(agent.buffer):,}  "
          f"Ready: {agent.buffer.is_ready_for(cfg['batch_size'])}\n")

    # ── Rolling windows ───────────────────────────────────────────────────────
    window            = cfg["log_interval"]
    reward_window     = deque(maxlen=window)
    success_window    = deque(maxlen=window)
    loss_window       = deque(maxlen=window)
    weight_window     = deque(maxlen=window)
    logicalerr_window = deque(maxlen=window)

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode in range(start_episode, cfg["n_episodes"] + 1):

        obs, _ = env.reset()
        done   = False
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

            ep_reward  += reward
            ep_weights.append(info["syndrome_weight"])
            obs         = next_obs
            global_step += 1

        # ── Episode stats ─────────────────────────────────────────────────────
        corrected   = info["corrected"]
        logical_err = info["logical_error"]
        avg_loss    = float(np.mean(ep_losses))  if ep_losses  else 0.0
        avg_weight  = float(np.mean(ep_weights)) if ep_weights else 0.0

        reward_window.append(ep_reward)
        success_window.append(float(corrected))
        loss_window.append(avg_loss)
        weight_window.append(avg_weight)
        logicalerr_window.append(float(logical_err))

        # ── Wandb log (step=episode → continuous plot from 0) ─────────────────
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
            "episode":     episode,
            "global_step": global_step,
        }, step=episode)

        # ── Crash-safe save EVERY episode ─────────────────────────────────────
        agent.save(LATEST_CKPT)
        save_meta(episode, run.id, best_success_rate, global_step)

        # ── Console + eval every log_interval ─────────────────────────────────
        if episode % cfg["log_interval"] == 0:
            sr  = np.mean(success_window)
            ler = np.mean(logicalerr_window)
            avr = np.mean(reward_window)
            avl = np.mean(loss_window)
            print(
                f"Ep {episode:>6}/{cfg['n_episodes']} | "
                f"Success={sr:.3f} | LogErr={ler:.3f} | "
                f"Reward={avr:+.2f} | Loss={avl:.5f} | "
                f"ε={agent.epsilon:.3f} | Buffer={len(agent.buffer):,}"
            )
            eval_metrics = evaluate(agent, env_eval, n_eval=200)
            wandb.log(eval_metrics, step=episode)
            eval_sr = eval_metrics["eval/success_rate"]
            print(f"         └─ Eval(200): success={eval_sr:.3f}  "
                  f"logical={eval_metrics['eval/logical_error_rate']:.3f}  "
                  f"reward={eval_metrics['eval/avg_reward']:+.2f}")

            if eval_sr > best_success_rate:
                best_success_rate = eval_sr
                best_path = os.path.join(CKPT_DIR, "d3_best.pt")
                agent.save(best_path)
                wandb.run.summary["best_success_rate"] = best_success_rate
                wandb.run.summary["best_episode"]      = episode
                save_meta(episode, run.id, best_success_rate, global_step)
                print(f"         ★ New best: {best_success_rate:.4f} → {best_path}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % cfg["save_interval"] == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"d3_ep{episode}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] Saved → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR, "d3_final.pt")
    agent.save(final_path)

    print("\n[Eval] Final evaluation — 500 episodes (greedy)...")
    final_metrics = evaluate(agent, env_eval, n_eval=500)
    wandb.log({**final_metrics, "episode": cfg["n_episodes"]})

    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═'*60}")
    print(f"  Final success rate : {final_metrics['eval/success_rate']:.4f}")
    print(f"  Logical error rate : {final_metrics['eval/logical_error_rate']:.4f}")
    print(f"  Best success rate  : {best_success_rate:.4f}")
    print(f"  Final checkpoint   : {final_path}")
    print(f"{'═'*60}\n")

    # Clean up crash-recovery files after successful completion
    if os.path.exists(LATEST_CKPT): os.remove(LATEST_CKPT)
    if os.path.exists(META_FILE):   os.remove(META_FILE)

    wandb.finish()
    return agent, final_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train d=3 DDQN (stim_final run)")
    parser.add_argument("--resume",          action="store_true",
                        help="Resume from last checkpoint in checkpoints_stim_final/")
    parser.add_argument("--episodes",        type=int,   default=DEFAULT_CONFIG["n_episodes"])
    parser.add_argument("--noise",           type=float, default=DEFAULT_CONFIG["noise"])
    parser.add_argument("--lr",              type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--buffer_capacity", type=int,   default=DEFAULT_CONFIG["buffer_capacity"])
    parser.add_argument("--epsilon_decay",   type=int,   default=DEFAULT_CONFIG["epsilon_decay"])
    parser.add_argument("--target_update",   type=int,   default=DEFAULT_CONFIG["target_update"])
    parser.add_argument("--max_steps",       type=int,   default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--warmup",          type=int,   default=DEFAULT_CONFIG["warmup_episodes"])
    parser.add_argument("--log_interval",    type=int,   default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--save_interval",   type=int,   default=DEFAULT_CONFIG["save_interval"])
    parser.add_argument("--device",          type=str,   default=DEFAULT_CONFIG["device"])
    parser.add_argument("--run_name",        type=str,   default=DEFAULT_CONFIG["run_name"])
    parser.add_argument("--project",         type=str,   default=DEFAULT_CONFIG["project"])
    args = parser.parse_args()

    train(
        config = {
            "n_episodes":       args.episodes,
            "noise":            args.noise,
            "lr":               args.lr,
            "batch_size":       args.batch_size,
            "buffer_capacity":  args.buffer_capacity,
            "epsilon_decay":    args.epsilon_decay,
            "target_update":    args.target_update,
            "max_steps":        args.max_steps,
            "warmup_episodes":  args.warmup,
            "log_interval":     args.log_interval,
            "save_interval":    args.save_interval,
            "device":           args.device,
            "run_name":         args.run_name,
            "project":          args.project,
        },
        resume = args.resume,
    )