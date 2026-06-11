# """
# Benchmark — PER vs LER for MWPM and trained RL agent
# ======================================================
# Compares decoder performance across a range of physical error rates:

#   PER values: [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

# For each PER:
#     1. Build a fresh Stim circuit (surface_code:rotated_memory_z, d=3)
#     2. Sample N noise traces from Stim
#     3. MWPM:      decode each trace → logical error rate (LER)
#     4. RL agent:  run an episode for each trace → logical error rate (LER)
#        (the RL agent applies multiple corrections per episode)
#     5. Record both LERs

# Output:
#     - PNG plot (log-log: PER on x, LER on y)
#     - JSON results
#     - Wandb chart (optional)

# Usage
# -----
# From quantum_rl root:
#     python -m reinforcement.surface.benchmark
#     python -m reinforcement.surface.benchmark --checkpoint checkpoints/d3_best.pt
#     python -m reinforcement.surface.benchmark --shots 5000 --no-wandb
# """

# import os
# import sys
# import argparse
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# # ── Path setup ────────────────────────────────────────────────────────────────
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)

# import stim
# import pymatching

# from reinforcement.surface.ddqn_agent import DDQNAgent
# from src.environments.surface3_env    import SurfaceCodeEnv


# # ══════════════════════════════════════════════════════════════════════════════
# #  HYPERPARAMETERS
# # ══════════════════════════════════════════════════════════════════════════════

# DEFAULT_PER = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]


# # ══════════════════════════════════════════════════════════════════════════════
# #  MWPM DECODER
# # ══════════════════════════════════════════════════════════════════════════════

# def build_stim_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
#     """Identical circuit to SurfaceCodeEnv — same noise model."""
#     return stim.Circuit.generated(
#         "surface_code:rotated_memory_z",
#         distance=distance,
#         rounds=rounds,
#         after_clifford_depolarization=noise,
#         after_reset_flip_probability=noise,
#         before_measure_flip_probability=noise,
#         before_round_data_depolarization=noise,
#     )


# def benchmark_mwpm(distance: int, noise: float, n_shots: int, rounds: int = None) -> float:
#     """
#     Run MWPM on `n_shots` independent Stim noise samples.
#     Returns the logical error rate.
#     """
#     if rounds is None:
#         rounds = distance

#     circuit  = build_stim_circuit(distance, rounds, noise)
#     dem      = circuit.detector_error_model(decompose_errors=True)
#     matcher  = pymatching.Matching.from_detector_error_model(dem)
#     sampler  = circuit.compile_detector_sampler()

#     detector_events, true_obs = sampler.sample(
#         shots=n_shots, separate_observables=True
#     )
#     predictions = matcher.decode_batch(detector_events)
#     n_errors    = int(np.sum(predictions != true_obs))
#     return n_errors / n_shots


# # ══════════════════════════════════════════════════════════════════════════════
# #  RL DECODER
# # ══════════════════════════════════════════════════════════════════════════════

# def benchmark_rl(
#     agent:     DDQNAgent,
#     distance:  int,
#     noise:     float,
#     n_shots:   int,
#     max_steps: int = 50,
#     rounds:    int = None,
# ) -> dict:
#     """
#     Run the trained RL agent on `n_shots` independent episodes at this noise level.

#     The agent applies MULTIPLE corrections per episode (one per step)
#     until either:
#         - syndrome weight = 0 → check logical error
#         - max_steps reached    → timeout

#     Returns dict with logical_error_rate, success_rate, timeout_rate, avg_steps.
#     """
#     if rounds is None:
#         rounds = distance

#     env = SurfaceCodeEnv(
#         distance        = distance,
#         noise           = noise,
#         syndrome_rounds = rounds,
#         max_steps       = max_steps,
#     )

#     n_logical_errors = 0
#     n_successes     = 0
#     n_timeouts      = 0
#     total_steps     = 0

#     for _ in range(n_shots):
#         obs, _ = env.reset()
#         done   = False
#         steps  = 0

#         while not done:
#             action                              = agent.select_action(obs, greedy=True)
#             obs, reward, term, trunc, info     = env.step(action)
#             done                                = term or trunc
#             steps                              += 1

#         total_steps += steps
#         if info["corrected"]:
#             n_successes += 1
#         elif info["logical_error"]:
#             n_logical_errors += 1
#         else:
#             n_timeouts += 1

#     return {
#         "logical_error_rate": n_logical_errors / n_shots,
#         "success_rate":       n_successes      / n_shots,
#         "timeout_rate":       n_timeouts       / n_shots,
#         "avg_steps":          total_steps      / n_shots,
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  PLOTTING
# # ══════════════════════════════════════════════════════════════════════════════

# def plot_per_vs_ler(per_values, mwpm_ler, rl_ler, save_path: str, distance: int):
#     """
#     Plot PER vs LER on log-log axes with both MWPM and RL curves.
#     """
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

#     # MWPM curve
#     ax.plot(
#         per_values, mwpm_ler,
#         marker="s", markersize=10, linewidth=2.2,
#         color="#0077B6", label="MWPM (classical baseline)",
#         markeredgecolor="white", markeredgewidth=1.5,
#     )

#     # RL curve
#     ax.plot(
#         per_values, rl_ler,
#         marker="o", markersize=10, linewidth=2.2,
#         color="#EF4444", label="RL Agent (DDQN)",
#         markeredgecolor="white", markeredgewidth=1.5,
#     )

#     # Pseudo-threshold reference line  (LER = PER means no protection)
#     pers = np.array(per_values)
#     ax.plot(pers, pers, linestyle="--", color="#94A3B8", linewidth=1.0,
#             label="LER = PER (no protection)")

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlabel("Physical Error Rate (PER)",  fontsize=13)
#     ax.set_ylabel("Logical Error Rate (LER)",   fontsize=13)
#     ax.set_title(f"Surface Code Decoding Benchmark  d={distance}",
#                  fontsize=14, fontweight="bold")
#     ax.grid(True, which="both", alpha=0.3)
#     ax.legend(loc="upper left", fontsize=11)

#     # Annotate each point with the value
#     for x, y in zip(per_values, mwpm_ler):
#         ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
#                     xytext=(0, 10), fontsize=8, color="#0077B6",
#                     ha="center")
#     for x, y in zip(per_values, rl_ler):
#         ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
#                     xytext=(0, -16), fontsize=8, color="#EF4444",
#                     ha="center")

#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()
#     print(f"[Plot] Saved → {save_path}")


# def plot_separate(per_values, mwpm_ler, rl_ler, save_path_mwpm: str,
#                   save_path_rl: str, distance: int):
#     """Two separate plots — one per decoder — for clarity."""
#     # MWPM only
#     fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
#     ax.plot(per_values, mwpm_ler, marker="s", markersize=10,
#             linewidth=2.2, color="#0077B6", markeredgecolor="white")
#     ax.set_xscale("log"); ax.set_yscale("log")
#     ax.set_xlabel("Physical Error Rate (PER)", fontsize=13)
#     ax.set_ylabel("Logical Error Rate (LER)",   fontsize=13)
#     ax.set_title(f"MWPM Decoder  d={distance}", fontsize=14, fontweight="bold")
#     ax.grid(True, which="both", alpha=0.3)
#     for x, y in zip(per_values, mwpm_ler):
#         ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
#                     xytext=(0, 10), fontsize=9, color="#0077B6", ha="center")
#     plt.tight_layout()
#     plt.savefig(save_path_mwpm, bbox_inches="tight")
#     plt.close()

#     # RL only
#     fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
#     ax.plot(per_values, rl_ler, marker="o", markersize=10,
#             linewidth=2.2, color="#EF4444", markeredgecolor="white")
#     ax.set_xscale("log"); ax.set_yscale("log")
#     ax.set_xlabel("Physical Error Rate (PER)", fontsize=13)
#     ax.set_ylabel("Logical Error Rate (LER)",   fontsize=13)
#     ax.set_title(f"RL Decoder (DDQN)  d={distance}", fontsize=14, fontweight="bold")
#     ax.grid(True, which="both", alpha=0.3)
#     for x, y in zip(per_values, rl_ler):
#         ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
#                     xytext=(0, 10), fontsize=9, color="#EF4444", ha="center")
#     plt.tight_layout()
#     plt.savefig(save_path_rl, bbox_inches="tight")
#     plt.close()

#     print(f"[Plot] Saved → {save_path_mwpm}")
#     print(f"[Plot] Saved → {save_path_rl}")


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def run_benchmark(
#     checkpoint_path: str,
#     per_values:      list  = DEFAULT_PER,
#     distance:        int   = 3,
#     rounds:          int   = None,
#     n_shots:         int   = 5000,
#     max_steps:       int   = 50,
#     out_dir:         str   = None,
#     use_wandb:       bool  = True,
# ):
#     """
#     Run the full PER vs LER benchmark.
#     """
#     if rounds is None:
#         rounds = distance
#     if out_dir is None:
#         out_dir = os.path.join(_ROOT, "benchmark_results")
#     os.makedirs(out_dir, exist_ok=True)

#     # ── Init wandb ────────────────────────────────────────────────────────────
#     wandb_run = None
#     if use_wandb:
#         try:
#             import wandb
#             wandb_run = wandb.init(
#                 project = "surface-code-rl",
#                 name    = f"benchmark-d{distance}",
#                 config  = {
#                     "distance":  distance,
#                     "rounds":    rounds,
#                     "n_shots":   n_shots,
#                     "max_steps": max_steps,
#                     "per_values": per_values,
#                     "checkpoint": checkpoint_path,
#                 },
#             )
#         except Exception as e:
#             print(f"[Warn] wandb init failed: {e}. Continuing without wandb.")
#             use_wandb = False

#     # ── Build temporary env to read dims for agent ───────────────────────────
#     tmp_env = SurfaceCodeEnv(distance=distance, noise=per_values[0],
#                              syndrome_rounds=rounds, max_steps=max_steps)
#     in_channels = tmp_env.observation_space.shape[0]
#     grid_size   = tmp_env.observation_space.shape[1]
#     n_actions   = tmp_env.action_space.n

#     # ── Load trained RL agent ─────────────────────────────────────────────────
#     agent = DDQNAgent(
#         in_channels=in_channels, grid_size=grid_size, n_actions=n_actions,
#         device="cpu",
#     )
#     agent.load(checkpoint_path)

#     # ── Run benchmark for each PER ────────────────────────────────────────────
#     print(f"\n{'═'*70}")
#     print(f"  BENCHMARK — d={distance}  shots/PER={n_shots:,}  max_steps={max_steps}")
#     print(f"  PER values: {per_values}")
#     print(f"  Checkpoint: {checkpoint_path}")
#     print(f"{'═'*70}\n")

#     results = {
#         "distance":  distance,
#         "rounds":    rounds,
#         "n_shots":   n_shots,
#         "max_steps": max_steps,
#         "per_values": per_values,
#         "mwpm_ler":  [],
#         "rl_ler":    [],
#         "rl_success":[],
#         "rl_timeout":[],
#         "rl_avg_steps":[],
#     }

#     for per in per_values:
#         print(f"┌─ PER = {per}")

#         # MWPM
#         mwpm_ler = benchmark_mwpm(distance, per, n_shots, rounds)
#         results["mwpm_ler"].append(mwpm_ler)
#         print(f"│   MWPM LER      : {mwpm_ler:.4f}")

#         # RL
#         rl_stats = benchmark_rl(agent, distance, per, n_shots, max_steps, rounds)
#         results["rl_ler"].append(rl_stats["logical_error_rate"])
#         results["rl_success"].append(rl_stats["success_rate"])
#         results["rl_timeout"].append(rl_stats["timeout_rate"])
#         results["rl_avg_steps"].append(rl_stats["avg_steps"])

#         print(f"│   RL  LER       : {rl_stats['logical_error_rate']:.4f}")
#         print(f"│   RL  success   : {rl_stats['success_rate']:.4f}")
#         print(f"│   RL  timeout   : {rl_stats['timeout_rate']:.4f}")
#         print(f"│   RL  avg steps : {rl_stats['avg_steps']:.2f}")
#         print(f"└─")

#         if use_wandb:
#             wandb_run.log({
#                 "per":          per,
#                 "mwpm_ler":     mwpm_ler,
#                 "rl_ler":       rl_stats["logical_error_rate"],
#                 "rl_success":   rl_stats["success_rate"],
#                 "rl_timeout":   rl_stats["timeout_rate"],
#                 "rl_avg_steps": rl_stats["avg_steps"],
#             })

#     # ── Save results JSON ─────────────────────────────────────────────────────
#     json_path = os.path.join(out_dir, f"benchmark_d{distance}.json")
#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n[Results] Saved → {json_path}")

#     # ── Plots ─────────────────────────────────────────────────────────────────
#     combined_path = os.path.join(out_dir, f"per_vs_ler_d{distance}.png")
#     plot_per_vs_ler(per_values, results["mwpm_ler"], results["rl_ler"],
#                     combined_path, distance)

#     mwpm_path = os.path.join(out_dir, f"per_vs_ler_d{distance}_mwpm.png")
#     rl_path   = os.path.join(out_dir, f"per_vs_ler_d{distance}_rl.png")
#     plot_separate(per_values, results["mwpm_ler"], results["rl_ler"],
#                   mwpm_path, rl_path, distance)

#     if use_wandb:
#         import wandb
#         wandb_run.log({
#             "plot/combined": wandb.Image(combined_path),
#             "plot/mwpm":     wandb.Image(mwpm_path),
#             "plot/rl":       wandb.Image(rl_path),
#         })
#         wandb_run.finish()

#     # ── Print summary ─────────────────────────────────────────────────────────
#     print(f"\n{'═'*70}")
#     print(f"  SUMMARY")
#     print(f"{'═'*70}")
#     print(f"  {'PER':>10} | {'MWPM LER':>10} | {'RL LER':>10} | {'RL Success':>10}")
#     print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
#     for per, m, r, s in zip(per_values, results["mwpm_ler"], results["rl_ler"],
#                             results["rl_success"]):
#         print(f"  {per:>10.3f} | {m:>10.4f} | {r:>10.4f} | {s:>10.4f}")
#     print(f"{'═'*70}\n")

#     return results


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLI
# # ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="PER vs LER benchmark — MWPM vs RL")
#     parser.add_argument("--checkpoint", type=str,
#                         default=os.path.join(_ROOT, "checkpoints", "d3_best.pt"),
#                         help="Path to trained agent checkpoint")
#     parser.add_argument("--distance",  type=int,   default=3)
#     parser.add_argument("--rounds",    type=int,   default=None)
#     parser.add_argument("--shots",     type=int,   default=5_000)
#     parser.add_argument("--max_steps", type=int,   default=50)
#     parser.add_argument("--per",       type=float, nargs="+", default=DEFAULT_PER,
#                         help="Physical error rates to test")
#     parser.add_argument("--out_dir",   type=str,   default=None)
#     parser.add_argument("--no-wandb",  action="store_true", help="Disable wandb")
#     args = parser.parse_args()

#     run_benchmark(
#         checkpoint_path = args.checkpoint,
#         per_values      = args.per,
#         distance        = args.distance,
#         rounds          = args.rounds,
#         n_shots         = args.shots,
#         max_steps       = args.max_steps,
#         out_dir         = args.out_dir,
#         use_wandb       = not args.no_wandb,
#     )
"""
Benchmark — PER vs LER for MWPM, Old RL, and New RL
=====================================================
Three-way comparison across physical error rates:

  PER values: [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

For each PER:
    1. Build a fresh Stim circuit (surface_code:rotated_memory_z, d=3)
    2. MWPM    → decode → logical error rate
    3. Old RL  → run episodes → logical error rate
    4. New RL  → run episodes → logical error rate

Output:
    - Combined PNG (all three curves)
    - Separate PNGs (one per decoder)
    - JSON results
    - Wandb charts (optional)

Usage
-----
From quantum_rl root:
    python -m reinforcement.surface.benchmark3 \
        --old_checkpoint checkpoints/d3_final.pt \
        --new_checkpoint checkpoints_stim_final/d3_best.pt

    # Disable wandb
    python -m reinforcement.surface.benchmark3 --no-wandb
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import stim
import pymatching

from reinforcement.surface.ddqn_agent import DDQNAgent
from src.environments.surface3_env    import SurfaceCodeEnv


DEFAULT_PER = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

# Colors for the three decoders
COL_MWPM   = "#0077B6"   # blue
COL_OLD_RL = "#F59E0B"   # amber
COL_NEW_RL = "#EF4444"   # red
COL_REF    = "#94A3B8"   # grey reference line


# ══════════════════════════════════════════════════════════════════════════════
#  MWPM DECODER
# ══════════════════════════════════════════════════════════════════════════════

def build_stim_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance, rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )


def benchmark_mwpm(distance: int, noise: float, n_shots: int, rounds: int = None) -> float:
    if rounds is None:
        rounds = distance
    circuit = build_stim_circuit(distance, rounds, noise)
    dem     = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    det, true_obs = sampler.sample(shots=n_shots, separate_observables=True)
    preds = matcher.decode_batch(det)
    return int(np.sum(preds != true_obs)) / n_shots


# ══════════════════════════════════════════════════════════════════════════════
#  RL DECODER
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_rl(agent, distance, noise, n_shots, max_steps=80, rounds=None) -> dict:
    if rounds is None:
        rounds = distance
    env = SurfaceCodeEnv(distance=distance, noise=noise,
                         syndrome_rounds=rounds, max_steps=max_steps)
    n_logical = n_success = n_timeout = total_steps = 0
    for _ in range(n_shots):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        total_steps += steps
        if info["corrected"]:     n_success += 1
        elif info["logical_error"]: n_logical += 1
        else:                       n_timeout += 1
    return {
        "logical_error_rate": n_logical / n_shots,
        "success_rate":       n_success / n_shots,
        "timeout_rate":       n_timeout / n_shots,
        "avg_steps":          total_steps / n_shots,
    }


def load_agent(checkpoint_path, distance, rounds, max_steps):
    """Load a DDQN agent from a checkpoint, inferring dims from the env."""
    tmp = SurfaceCodeEnv(distance=distance, noise=0.01,
                         syndrome_rounds=rounds, max_steps=max_steps)
    agent = DDQNAgent(
        in_channels=tmp.observation_space.shape[0],
        grid_size=tmp.observation_space.shape[1],
        n_actions=tmp.action_space.n,
        device="cpu",
    )
    agent.load(checkpoint_path)
    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING — combined 3-way
# ══════════════════════════════════════════════════════════════════════════════

def plot_three(per_values, mwpm, old_rl, new_rl, save_path, distance):
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=120)

    ax.plot(per_values, mwpm, marker="s", markersize=10, linewidth=2.2,
            color=COL_MWPM, label="MWPM (classical baseline)",
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(per_values, old_rl, marker="^", markersize=10, linewidth=2.2,
            color=COL_OLD_RL, label="Old RL (DDQN)",
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(per_values, new_rl, marker="o", markersize=10, linewidth=2.2,
            color=COL_NEW_RL, label="New RL (DDQN)",
            markeredgecolor="white", markeredgewidth=1.5)

    pers = np.array(per_values)
    ax.plot(pers, pers, linestyle="--", color=COL_REF, linewidth=1.0,
            label="LER = PER (no protection)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (PER)", fontsize=13)
    ax.set_ylabel("Logical Error Rate (LER)",   fontsize=13)
    ax.set_title(f"Surface Code Decoding — 3-way Benchmark  d={distance}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_each(per_values, series, save_dir, distance):
    """One plot per decoder for clarity."""
    decoders = [
        ("mwpm",   "MWPM Decoder",        series["mwpm_ler"],  COL_MWPM,   "s"),
        ("old_rl", "Old RL Decoder",      series["old_rl_ler"],COL_OLD_RL, "^"),
        ("new_rl", "New RL Decoder",      series["new_rl_ler"],COL_NEW_RL, "o"),
    ]
    for key, title, data, col, marker in decoders:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        ax.plot(per_values, data, marker=marker, markersize=10, linewidth=2.2,
                color=col, markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate (PER)", fontsize=13)
        ax.set_ylabel("Logical Error Rate (LER)",   fontsize=13)
        ax.set_title(f"{title}  d={distance}", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        for x, y in zip(per_values, data):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), fontsize=9, color=col, ha="center")
        path = os.path.join(save_dir, f"per_vs_ler_d{distance}_{key}.png")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    old_checkpoint: str,
    new_checkpoint: str,
    per_values:     list  = DEFAULT_PER,
    distance:       int   = 3,
    rounds:         int   = None,
    n_shots:        int   = 5000,
    max_steps:      int   = 80,
    out_dir:        str   = None,
    use_wandb:      bool  = True,
):
    if rounds is None:
        rounds = distance
    if out_dir is None:
        out_dir = os.path.join(_ROOT, "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)

    # ── Init wandb ────────────────────────────────────────────────────────────
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="surface-code-rl",
                name=f"benchmark3-d{distance}",
                config={
                    "distance": distance, "rounds": rounds, "n_shots": n_shots,
                    "max_steps": max_steps, "per_values": per_values,
                    "old_checkpoint": old_checkpoint,
                    "new_checkpoint": new_checkpoint,
                },
            )
        except Exception as e:
            print(f"[Warn] wandb init failed: {e}. Continuing without wandb.")
            use_wandb = False

    # ── Load both agents ──────────────────────────────────────────────────────
    print(f"[Load] Old RL: {old_checkpoint}")
    old_agent = load_agent(old_checkpoint, distance, rounds, max_steps)
    print(f"[Load] New RL: {new_checkpoint}")
    new_agent = load_agent(new_checkpoint, distance, rounds, max_steps)

    # ── Run benchmark for each PER ────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  3-WAY BENCHMARK — d={distance}  shots/PER={n_shots:,}")
    print(f"  MWPM  vs  Old RL  vs  New RL")
    print(f"{'═'*72}\n")

    results = {
        "distance": distance, "rounds": rounds, "n_shots": n_shots,
        "max_steps": max_steps, "per_values": per_values,
        "mwpm_ler": [], "old_rl_ler": [], "new_rl_ler": [],
        "old_rl_success": [], "new_rl_success": [],
    }

    for per in per_values:
        print(f"┌─ PER = {per}")

        mwpm_ler = benchmark_mwpm(distance, per, n_shots, rounds)
        results["mwpm_ler"].append(mwpm_ler)
        print(f"│   MWPM   LER     : {mwpm_ler:.4f}")

        old_stats = benchmark_rl(old_agent, distance, per, n_shots, max_steps, rounds)
        results["old_rl_ler"].append(old_stats["logical_error_rate"])
        results["old_rl_success"].append(old_stats["success_rate"])
        print(f"│   Old RL LER     : {old_stats['logical_error_rate']:.4f}  "
              f"(success {old_stats['success_rate']:.3f})")

        new_stats = benchmark_rl(new_agent, distance, per, n_shots, max_steps, rounds)
        results["new_rl_ler"].append(new_stats["logical_error_rate"])
        results["new_rl_success"].append(new_stats["success_rate"])
        print(f"│   New RL LER     : {new_stats['logical_error_rate']:.4f}  "
              f"(success {new_stats['success_rate']:.3f})")
        print(f"└─")

        if use_wandb:
            wandb_run.log({
                "per": per,
                "mwpm_ler":       mwpm_ler,
                "old_rl_ler":     old_stats["logical_error_rate"],
                "new_rl_ler":     new_stats["logical_error_rate"],
                "old_rl_success": old_stats["success_rate"],
                "new_rl_success": new_stats["success_rate"],
            })

    # ── Save results JSON ─────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, f"benchmark3_d{distance}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved → {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    combined = os.path.join(out_dir, f"per_vs_ler_d{distance}_3way.png")
    plot_three(per_values, results["mwpm_ler"], results["old_rl_ler"],
               results["new_rl_ler"], combined, distance)
    plot_each(per_values, results, out_dir, distance)

    if use_wandb:
        import wandb
        wandb_run.log({
            "plot/combined": wandb.Image(combined),
            "plot/mwpm":     wandb.Image(os.path.join(out_dir, f"per_vs_ler_d{distance}_mwpm.png")),
            "plot/old_rl":   wandb.Image(os.path.join(out_dir, f"per_vs_ler_d{distance}_old_rl.png")),
            "plot/new_rl":   wandb.Image(os.path.join(out_dir, f"per_vs_ler_d{distance}_new_rl.png")),
        })
        wandb_run.finish()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  SUMMARY")
    print(f"{'═'*72}")
    print(f"  {'PER':>8} | {'MWPM':>9} | {'Old RL':>9} | {'New RL':>9}")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for per, m, o, n in zip(per_values, results["mwpm_ler"],
                            results["old_rl_ler"], results["new_rl_ler"]):
        print(f"  {per:>8.3f} | {m:>9.4f} | {o:>9.4f} | {n:>9.4f}")
    print(f"{'═'*72}\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-way PER vs LER benchmark")
    parser.add_argument("--old_checkpoint", type=str,
                        default=os.path.join(_ROOT, "checkpoints", "d3_final.pt"),
                        help="Old RL checkpoint")
    parser.add_argument("--new_checkpoint", type=str,
                        default=os.path.join(_ROOT, "checkpoints_stim_final", "d3_best.pt"),
                        help="New RL checkpoint")
    parser.add_argument("--distance",  type=int,   default=3)
    parser.add_argument("--rounds",    type=int,   default=None)
    parser.add_argument("--shots",     type=int,   default=5_000)
    parser.add_argument("--max_steps", type=int,   default=80)
    parser.add_argument("--per",       type=float, nargs="+", default=DEFAULT_PER)
    parser.add_argument("--out_dir",   type=str,   default=None)
    parser.add_argument("--no-wandb",  action="store_true")
    args = parser.parse_args()

    run_benchmark(
        old_checkpoint = args.old_checkpoint,
        new_checkpoint = args.new_checkpoint,
        per_values     = args.per,
        distance       = args.distance,
        rounds         = args.rounds,
        n_shots        = args.shots,
        max_steps      = args.max_steps,
        out_dir        = args.out_dir,
        use_wandb      = not args.no_wandb,
    )