"""
PPR Curriculum Trainer — Realistic d=5 Surface Code
=====================================================
Transfer a trained decoder ACROSS NOISE LEVELS at fixed distance d=5.

This is the CORRECT use of Probabilistic Policy Reuse (PPR): the source and
target tasks share identical state (15,11,11) and action MultiDiscrete([4]*25)
shapes, so the reused policy applies to the full lattice with no cropping.

Two transfer mechanisms, used together
--------------------------------------
  1. WARM START — the target agent's network is initialized by copying ALL
     weights from the source checkpoint. The decoder already knows how to
     map syndromes to corrections; it only needs to adapt to a higher error
     rate.

  2. PROBABILISTIC POLICY REUSE (PPR) — a frozen copy of the source policy
     (the "library") guides exploration. At each step, with probability ψ the
     action comes from the library (greedy); otherwise from the target agent
     (ε-greedy). ψ decays over training so the agent weans off the library:

        ψ(stage_step) = max(ψ_end, ψ_start · (1 − stage_step / ψ_horizon))

Usage
-----
    python -m reinforcement.surface5_realistic.train_d5_ppr_curriculum \
        --source_ckpt checkpoints_d5_continuous_new/d5_final.pt \
        --p_data 0.005 --episodes 40000 --run_name d5-ppr-p005

    # resume:
    python -m reinforcement.surface5_realistic.train_d5_ppr_curriculum --resume
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import wandb

from reinforcement.surface5_realistic.Ddqn_agent_multi import MultiDiscreteDDQNAgent
from src.environments.surface5_realistic_env         import SurfaceCodeRealisticEnv

# ── Checkpoint dir (separate from the p=0.001 stage) ──────────────────────────
CKPT_DIR    = os.path.join(_ROOT, "checkpoints_d5_ppr_p005_latest")
LATEST_CKPT = os.path.join(CKPT_DIR, "d5_latest.pt")
META_FILE   = os.path.join(CKPT_DIR, "d5_meta.json")
os.makedirs(CKPT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # ── Source (PPR library + warm start) ──
    source_ckpt = os.path.join(_ROOT, "checkpoints_d5_continuous_new1", "d5_final.pt"),

    # ── Environment (target noise level) ──
    distance   = 5,
    k          = 5,
    T          = 500,
    p_data     = 0.005,           # TARGET noise (higher than source 0.001)
    p_meas     = 0.0,
    R_success  = 25.0,
    R_failure  = 25.0,

    # ── Agent ──
    lr              = 1e-5,
    lr_conv1        = None,
    grad_clip       = 1.0,
    gamma           = 0.99,
    epsilon_start   = 0.7,        # lower than 1.0: warm-started agent already knows a lot
    epsilon_end     = 0.05,
    epsilon_decay   = 500_000,
    batch_size      = 64,
    target_update   = 5_000,
    buffer_capacity = 100_000,
    identity_bias   = 0.95,

    # ── PPR schedule ──
    psi_start   = 0.6,            # initial probability of following the library
    psi_end     = 0.00,           # final probability
    psi_horizon = 200_000,        # steps over which ψ decays start→end

    # ── Training ──
    n_episodes      = 80_000,
    warmup_episodes = 25,
    train_every     = 1,
    log_interval    = 50,
    save_interval   = 1_000,
    eval_interval   = 500,
    eval_episodes   = 50,

    # ── Wandb ──
    project   = "surface-code-rl-d5-realistic",
    run_name  = "d5-ppr-p005",
    device    = "cpu",
)


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME HELPERS (crash-safe)
# ══════════════════════════════════════════════════════════════════════════════

def load_meta() -> dict:
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print("[Resume] meta file empty/corrupted — starting fresh")
            return {}
    return {}

def save_meta(episode, run_id, best_len, global_step, stage_step):
    tmp = META_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "episode": episode, "wandb_run_id": run_id,
            "best_len": best_len, "global_step": global_step,
            "stage_step": stage_step,
        }, f)
    os.replace(tmp, META_FILE)


# ══════════════════════════════════════════════════════════════════════════════
#  PPR ψ SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def psi_value(stage_step, cfg):
    """Linear decay of the policy-reuse probability."""
    frac = stage_step / cfg["psi_horizon"]
    return max(cfg["psi_end"], cfg["psi_start"] * (1.0 - frac))


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION (greedy, target agent only — no PPR, no exploration)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent, env, n_eval=50):
    survivals = 0
    deaths    = 0
    total_len = 0
    total_r   = 0.0
    for _ in range(n_eval):
        obs, _ = env.reset()
        over = False
        L = 0; R = 0.0
        while not over:
            a = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, info = env.step(a)
            over = term or trunc
            L += 1; R += float(np.sum(r))
        total_len += L; total_r += R
        if trunc and not term: survivals += 1
        if term: deaths += 1
    return {
        "eval/survival_rate":      survivals / n_eval,
        "eval/logical_death_rate": deaths    / n_eval,
        "eval/avg_episode_length": total_len / n_eval,
        "eval/avg_reward":         total_r   / n_eval,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(config=None, resume=False):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── Resume detection ──
    start_episode = 1
    best_len      = 0.0
    global_step   = 0
    stage_step    = 0        # counts target-agent steps for the ψ schedule
    meta          = {}

    if resume and os.path.exists(LATEST_CKPT) and os.path.exists(META_FILE):
        meta = load_meta()
        if meta:
            start_episode = meta.get("episode", 0) + 1
            best_len      = meta.get("best_len", 0.0)
            global_step   = meta.get("global_step", 0)
            stage_step    = meta.get("stage_step", 0)
            print(f"\n[Resume] From episode {start_episode}, stage_step={stage_step:,}")
        else:
            resume = False
    elif resume:
        print("\n[Resume] No checkpoint — starting fresh")
        resume = False

    # ── Wandb ──
    wk = dict(project=cfg["project"], name=cfg["run_name"], config=cfg)
    if resume and "wandb_run_id" in meta:
        wk["id"] = meta["wandb_run_id"]; wk["resume"] = "must"
    run = wandb.init(**wk)

    print(f"\n{'═'*66}")
    print(f"  PPR Curriculum — d=5  (warm start + policy reuse)")
    print(f"{'═'*66}")
    print(f"  source ckpt    : {cfg['source_ckpt']}")
    print(f"  target noise   : p_data={cfg['p_data']}  p_meas={cfg['p_meas']}")
    print(f"  ψ schedule     : {cfg['psi_start']} → {cfg['psi_end']} over {cfg['psi_horizon']:,} steps")
    print(f"  episodes       : {start_episode} → {cfg['n_episodes']}")
    print(f"  checkpoint dir : checkpoints_d5_ppr_p005/")
    print(f"{'═'*66}\n")

    # ── Environments (target noise) ──
    env = SurfaceCodeRealisticEnv(distance=cfg["distance"], k=cfg["k"], T=cfg["T"],
        p_data=cfg["p_data"], p_meas=cfg["p_meas"],
        R_success=cfg["R_success"], R_failure=cfg["R_failure"])
    env_eval = SurfaceCodeRealisticEnv(distance=cfg["distance"], k=cfg["k"], T=cfg["T"],
        p_data=cfg["p_data"], p_meas=cfg["p_meas"],
        R_success=cfg["R_success"], R_failure=cfg["R_failure"])

    in_ch = env.observation_space.shape[0]
    gs    = env.observation_space.shape[1]
    nq    = env.n_qubits

    # ── Target agent ──
    agent = MultiDiscreteDDQNAgent(
        in_channels=in_ch, grid_size=gs, n_qubits=nq, n_per_qubit=4,
        lr=cfg["lr"], lr_conv1=cfg["lr_conv1"], grad_clip=cfg["grad_clip"],
        gamma=cfg["gamma"], epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"], epsilon_decay=cfg["epsilon_decay"],
        batch_size=cfg["batch_size"], target_update=cfg["target_update"],
        buffer_capacity=cfg["buffer_capacity"], identity_bias=cfg["identity_bias"],
        device=cfg["device"])

    # ── Frozen LIBRARY policy (source) ──
    library = MultiDiscreteDDQNAgent(
        in_channels=in_ch, grid_size=gs, n_qubits=nq, n_per_qubit=4,
        lr=cfg["lr"], device=cfg["device"])

    if resume and os.path.exists(LATEST_CKPT):
        # Resuming: target weights come from our own latest checkpoint,
        # library still loads from the source.
        agent.load(LATEST_CKPT)
        library.load(cfg["source_ckpt"])
        print(f"[Resume] target ← {LATEST_CKPT}")
        print(f"[Resume] library ← {cfg['source_ckpt']}")
    else:
        # Fresh PPR stage: WARM START target from source, library = same source.
        if not os.path.exists(cfg["source_ckpt"]):
            raise FileNotFoundError(
                f"Source checkpoint not found: {cfg['source_ckpt']}\n"
                f"Finish the p=0.001 run first (produces d5_final.pt), "
                f"or pass --source_ckpt with a valid path.")
        agent.load(cfg["source_ckpt"])      # WARM START (all weights)
        library.load(cfg["source_ckpt"])    # LIBRARY (frozen)
        # Reset exploration for the new (harder) noise level
        agent.epsilon     = cfg["epsilon_start"]
        agent._steps_done = 0               # ε schedule restarts for this stage
        print(f"[WarmStart] target ← {cfg['source_ckpt']}")
        print(f"[Library]   frozen ← {cfg['source_ckpt']}")

    # Freeze the library: eval mode, never trained, always greedy
    library.online_net.eval()
    for p in library.online_net.parameters():
        p.requires_grad_(False)
    library.epsilon = 0.0

    agent.summary()
    wandb.watch(agent.online_net, log="all", log_freq=1000)

    # ── Warmup buffer (target agent, Identity-biased) ──
    warm = cfg["warmup_episodes"] if not resume else max(5, cfg["warmup_episodes"]//2)
    print(f"[Warmup] {warm} Identity-biased episodes (target agent)...")
    saved_eps = agent.epsilon
    agent.epsilon = max(saved_eps, 0.3)
    for _ in range(warm):
        obs, _ = env.reset(); over = False
        while not over:
            a = agent.select_action(obs)
            nobs, r, term, trunc, info = env.step(a)
            over = term or trunc
            agent.push(obs, a, r, nobs, float(term))
            obs = nobs
    agent.epsilon = saved_eps
    if not resume:
        agent._steps_done = 0
    print(f"[Warmup] Buffer: {len(agent.buffer):,}\n")

    # ── Rolling windows ──
    W = cfg["log_interval"]
    win = {k: deque(maxlen=W) for k in
           ["surv","death","len","rew","loss","psi","src"]}

    # ── Training loop ──
    for episode in range(start_episode, cfg["n_episodes"] + 1):
        obs, _ = env.reset()
        over = False
        ep_len = 0; ep_r = 0.0; ep_losses = []
        ep_src = 0     # how many actions came from the library this episode

        while not over:
            psi = psi_value(stage_step, cfg)

            # ── PPR action selection ──
            if np.random.random() < psi:
                action = library.select_action(obs, greedy=True)   # frozen source
                ep_src += 1
            else:
                action = agent.select_action(obs)                  # target ε-greedy

            nobs, reward, term, trunc, info = env.step(action)
            over = term or trunc

            # CONTINUING TASK: done = terminated only
            agent.push(obs, action, reward, nobs, float(term))

            if global_step % cfg["train_every"] == 0:
                loss = agent.train_step()
                if loss is not None: ep_losses.append(loss)

            ep_r += float(np.sum(reward)); ep_len += 1
            obs = nobs
            global_step += 1
            stage_step  += 1

        # ── Stats ──
        survived = bool(trunc and not term)
        died     = bool(term)
        aloss    = float(np.mean(ep_losses)) if ep_losses else 0.0
        psi_now  = psi_value(stage_step, cfg)
        src_frac = ep_src / max(1, ep_len)

        win["surv"].append(float(survived)); win["death"].append(float(died))
        win["len"].append(ep_len); win["rew"].append(ep_r)
        win["loss"].append(aloss); win["psi"].append(psi_now); win["src"].append(src_frac)

        wandb.log({
            "train/episode_reward": ep_r, "train/episode_length": ep_len,
            "train/survived": float(survived), "train/logical_death": float(died),
            "train/loss": aloss,
            "ppr/psi": psi_now, "ppr/library_action_frac": src_frac,
            "agent/epsilon": agent.epsilon, "agent/buffer_size": len(agent.buffer),
            "rolling/survival_rate": np.mean(win["surv"]),
            "rolling/death_rate":    np.mean(win["death"]),
            "rolling/avg_length":    np.mean(win["len"]),
            "rolling/avg_reward":    np.mean(win["rew"]),
            "rolling/avg_loss":      np.mean(win["loss"]),
            "rolling/psi":           np.mean(win["psi"]),
            "rolling/library_frac":  np.mean(win["src"]),
            "episode": episode, "global_step": global_step, "stage_step": stage_step,
        }, step=episode)

        # ── Crash-safe save (every 25 episodes; atomic) ──
        if episode % 25 == 0:
            agent.save(LATEST_CKPT)
            save_meta(episode, run.id, best_len, global_step, stage_step)

        # ── Console ──
        if episode % cfg["log_interval"] == 0:
            print(f"Ep {episode:>6}/{cfg['n_episodes']} | "
                  f"Len={np.mean(win['len']):6.1f} | "
                  f"Death={np.mean(win['death']):.3f} | "
                  f"Reward={np.mean(win['rew']):+8.2f} | "
                  f"ψ={psi_now:.3f} | libFrac={np.mean(win['src']):.2f} | "
                  f"ε={agent.epsilon:.3f}")

        # ── Eval ──
        if episode % cfg["eval_interval"] == 0:
            em = evaluate(agent, env_eval, n_eval=cfg["eval_episodes"])
            wandb.log(em, step=episode)
            el = em["eval/avg_episode_length"]
            print(f"         └─ Eval: len={el:.1f}  "
                  f"survival={em['eval/survival_rate']:.3f}  "
                  f"death={em['eval/logical_death_rate']:.3f}")
            # Select best on episode LENGTH (the right metric at this noise)
            if el > best_len:
                best_len = el
                bp = os.path.join(CKPT_DIR, "d5_best.pt")
                agent.save(bp)
                wandb.run.summary["best_avg_length"] = best_len
                wandb.run.summary["best_episode"]    = episode
                save_meta(episode, run.id, best_len, global_step, stage_step)
                print(f"         ★ New best length: {best_len:.1f} → {bp}")

        # ── Periodic checkpoint ──
        if episode % cfg["save_interval"] == 0:
            cp = os.path.join(CKPT_DIR, f"d5_ep{episode}.pt")
            agent.save(cp)
            print(f"[Checkpoint] → {cp}")

    # ── Final ──
    fp = os.path.join(CKPT_DIR, "d5_final.pt")
    agent.save(fp)
    print("\n[Eval] Final — 200 greedy episodes...")
    fm = evaluate(agent, env_eval, n_eval=200)
    wandb.log({**fm, "episode": cfg["n_episodes"]})
    print(f"\n{'═'*66}")
    print(f"  PPR STAGE COMPLETE  (p_data={cfg['p_data']})")
    print(f"{'═'*66}")
    print(f"  Final avg length   : {fm['eval/avg_episode_length']:.1f} / {cfg['T']}")
    print(f"  Final survival     : {fm['eval/survival_rate']:.4f}")
    print(f"  Best avg length    : {best_len:.1f}")
    print(f"  Final checkpoint   : {fp}")
    print(f"{'═'*66}\n")

    wandb.finish()
    return agent, fm


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PPR curriculum trainer (across noise levels)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--source_ckpt", type=str, default=DEFAULT_CONFIG["source_ckpt"])
    ap.add_argument("--p_data", type=float, default=DEFAULT_CONFIG["p_data"])
    ap.add_argument("--p_meas", type=float, default=DEFAULT_CONFIG["p_meas"])
    ap.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["n_episodes"])
    ap.add_argument("--psi_start", type=float, default=DEFAULT_CONFIG["psi_start"])
    ap.add_argument("--psi_end", type=float, default=DEFAULT_CONFIG["psi_end"])
    ap.add_argument("--psi_horizon", type=int, default=DEFAULT_CONFIG["psi_horizon"])
    ap.add_argument("--epsilon_start", type=float, default=DEFAULT_CONFIG["epsilon_start"])
    ap.add_argument("--epsilon_decay", type=int, default=DEFAULT_CONFIG["epsilon_decay"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    ap.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"])
    ap.add_argument("--run_name", type=str, default=DEFAULT_CONFIG["run_name"])
    ap.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    args = ap.parse_args()

    train(config={
        "source_ckpt": args.source_ckpt, "p_data": args.p_data, "p_meas": args.p_meas,
        "n_episodes": args.episodes, "psi_start": args.psi_start, "psi_end": args.psi_end,
        "psi_horizon": args.psi_horizon, "epsilon_start": args.epsilon_start,
        "epsilon_decay": args.epsilon_decay, "lr": args.lr, "gamma": args.gamma,
        "run_name": args.run_name, "device": args.device,
    }, resume=args.resume)