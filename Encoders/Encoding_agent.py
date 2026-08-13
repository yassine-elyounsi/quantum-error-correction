# """
# encoding_agent.py
# =================
# The QEC encoding agent — the single entry point for the full system.

# What this file does
# -------------------
#   1. Defines EncodingAgent — a wrapper around A2CAgent + MetaQECEnv
#      that handles training, saving, loading, and circuit discovery.

#   2. Provides a clean API:

#         agent = EncodingAgent()
#         agent.train()                          # curriculum training
#         circuit = agent.encode(n, k, d, c_Z, p_I)  # discover a circuit
#         agent.save("policy.pkl")               # save the trained policy
#         agent = EncodingAgent.load("policy.pkl")    # reload it later
#         agent.validate(n, k, d, c_Z, p_I)     # verify the circuit works

#   3. Can be run directly:
#         python encoding_agent.py               # trains and saves

# How the RL works (plain language)
# ----------------------------------
#   INPUT  (every step):
#     - The current circuit state as a flat binary vector
#       (the check matrix G, flattened — all 0s and 1s)
#     - 5 context numbers: n, k, d, c_Z, p_I  (normalized to [0,1])

#   OUTPUT (every step):
#     - One gate to apply: H(i), S(i), or CNOT(i→j)
#     - The actor network produces a probability for each available gate
#     - The agent samples one gate from those probabilities

#   REWARD (every step):
#     - r = −Σ λ_μ K_μ   (Eq. 10 of the paper)
#     - K_μ = 1 if error E_μ is NOT detected, else 0
#     - λ_μ = p_μ / max(p_μ)  (error weight from the noise model)
#     - reward = 0 when ALL errors are detected  →  success
#     - +20 bonus on success, −10 penalty on timeout

#   LEARNING:
#     - A2C with GAE: after each episode, compute advantages and
#       update actor + critic weights
#     - Curriculum: phases 0→3, each adding harder codes and wider
#       noise ranges. The agent builds on what it learned earlier.

#   AFTER TRAINING:
#     - Call agent.encode(n, k, d, c_Z, p_I)
#     - The agent runs one greedy episode and returns the gate sequence
#     - That sequence is a real encoding circuit for a real quantum chip
# """

# import os
# import time
# import pickle
# import numpy as np
# from typing import Optional

# from Encoders.Clifford_sim import (
#     all_pauli_strings_up_to_weight,
#     pauli_strings_to_binary,
#     kl_undetected_mask,
#     validate_circuit,
# )
# from Encoders.meta_env import MetaQECEnv, CURRICULUM_PHASES, TOTAL_EPISODES
# from Encoders.A2C_agent import A2CAgent, collect_episode


# # =====================================================================
# # The encoding agent
# # =====================================================================

# class EncodingAgent:
#     """
#     One agent that discovers encoding circuits for many codes and noise
#     models. Trained once via a curriculum, usable forever after.

#     Parameters
#     ----------
#     n_max        : largest number of physical qubits to support.
#     hidden       : hidden layer size of the actor and critic networks.
#     lr           : learning rate.
#     gamma        : discount factor for future rewards.
#     gae_lambda   : GAE lambda (0 = TD, 1 = Monte Carlo, 0.95 = paper).
#     ent_coef     : entropy bonus coefficient (keeps the policy exploring).
#     vf_coef      : critic loss coefficient.
#     max_grad_norm: gradient clipping threshold.
#     p_I          : base noise rate for the depolarizing channel.
#     max_steps    : maximum gates the agent can apply per episode.
#     success_bonus: reward added when all KL conditions are satisfied.
#     fail_penalty : reward added when the step budget is exhausted.
#     seed         : random seed.
#     """

#     def __init__(
#         self,
#         n_max=9,
#         hidden=128,
#         lr=2e-3,
#         gamma=0.95,
#         gae_lambda=0.95,
#         ent_coef=0.03,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         p_I=0.9,
#         max_steps=35,
#         success_bonus=20.0,
#         fail_penalty=-10.0,
#         seed=0,
#     ):
#         self.n_max        = n_max
#         self.p_I          = p_I
#         self.max_steps    = max_steps
#         self.success_bonus = success_bonus
#         self.fail_penalty  = fail_penalty
#         self.seed          = seed

#         # build the environment — this defines obs_dim and num_actions
#         self.env = MetaQECEnv(
#             n_max=n_max,
#             p_I=p_I,
#             max_steps=max_steps,
#             success_bonus=success_bonus,
#             fail_penalty=fail_penalty,
#             seed=seed,
#         )

#         # build the agent
#         self.agent = A2CAgent(
#             obs_dim=self.env.obs_dim,
#             n_actions=self.env.num_actions,
#             hidden=hidden,
#             lr=lr,
#             gamma=gamma,
#             gae_lambda=gae_lambda,
#             ent_coef=ent_coef,
#             vf_coef=vf_coef,
#             max_grad_norm=max_grad_norm,
#             seed=seed,
#         )

#         # training history
#         self.history = []
#         self.best_circuits = {}   # key: (n, k, d)  value: shortest circuit found

#     # ── training ────────────────────────────────────────────────────

#     def train(
#         self,
#         num_episodes=None,
#         log_every=200,
#         save_path=None,
#         save_every=1000,
#         verbose=True,
#     ):
#         """
#         Run the full curriculum training.

#         Parameters
#         ----------
#         num_episodes : total episodes to train (default = full curriculum).
#         log_every    : print a progress line every N episodes.
#         save_path    : if given, save the policy to this path.
#         save_every   : save a checkpoint every N episodes.
#         verbose      : print progress.
#         """
#         if num_episodes is None:
#             num_episodes = TOTAL_EPISODES

#         if verbose:
#             print("=" * 65)
#             print("ENCODING AGENT — curriculum training")
#             print(f"  obs_dim    : {self.env.obs_dim}")
#             print(f"  num_actions: {self.env.num_actions}")
#             print(f"  n_max      : {self.n_max}")
#             print(f"  episodes   : {num_episodes}")
#             print("=" * 65)

#         recent_rewards  = []
#         recent_success  = []
#         recent_lengths  = []
#         t_start = time.time()

#         for ep in range(1, num_episodes + 1):

#             # collect one episode
#             batch, info = collect_episode(self.env, self.agent)
#             self.agent.update(batch)

#             # track stats
#             recent_rewards.append(info["total_reward"])
#             recent_success.append(float(info["success"]))
#             recent_lengths.append(info["length"])

#             # track best circuits per code
#             if info["success"]:
#                 key = (self.env.n, self.env.k, self.env.d)
#                 prev_best = self.best_circuits.get(key, {}).get("length", 999)
#                 if info["length"] < prev_best:
#                     self.best_circuits[key] = {
#                         "circuit": list(info["history"]),
#                         "length":  info["length"],
#                         "c_Z":     self.env.c_Z,
#                         "p_I":     self.p_I,
#                     }

#             self.history.append({
#                 "episode":       ep,
#                 "total_reward":  info["total_reward"],
#                 "success":       info["success"],
#                 "length":        info["length"],
#                 "n":             self.env.n,
#                 "d":             self.env.d,
#                 "c_Z":           self.env.c_Z,
#                 "phase":         self.env._phase_idx,
#             })

#             # print progress
#             if verbose and ep % log_every == 0:
#                 w = min(log_every, len(recent_success))
#                 sr  = float(np.mean(recent_success[-w:]))
#                 avg_r = float(np.mean(recent_rewards[-w:]))
#                 avg_l = float(np.mean(recent_lengths[-w:]))
#                 elapsed = time.time() - t_start
#                 phase_name = self.env.phase_name
#                 print(
#                     f"  ep {ep:6d}/{num_episodes}"
#                     f"  phase {self.env._phase_idx}"
#                     f"  SR {sr:.2f}"
#                     f"  avg_reward {avg_r:7.2f}"
#                     f"  avg_len {avg_l:5.1f}"
#                     f"  elapsed {elapsed:6.0f}s"
#                     f"  [{phase_name}]"
#                 )

#             # save checkpoint
#             if save_path and ep % save_every == 0:
#                 self.save(save_path)
#                 if verbose:
#                     print(f"  → checkpoint saved to {save_path}")

#         if verbose:
#             elapsed = time.time() - t_start
#             print(f"\nTraining complete in {elapsed:.0f}s")
#             print(f"Best circuits discovered:")
#             for (n, k, d), info in sorted(self.best_circuits.items()):
#                 print(f"  [[{n},{k},{d}]]  {info['length']} gates"
#                       f"  c_Z={info['c_Z']:.2f}")

#         if save_path:
#             self.save(save_path)

#     # ── circuit discovery ────────────────────────────────────────────

#     def encode(
#         self,
#         n: int,
#         k: int = 1,
#         d: int = 3,
#         c_Z: float = 1.0,
#         p_I: float = 0.9,
#         n_attempts: int = 10,
#         greedy: bool = True,
#     ):
#         """
#         Discover an encoding circuit for the given code and noise model.

#         The agent runs up to `n_attempts` episodes and returns the
#         shortest successful circuit found. If no attempt succeeds,
#         returns the best partial circuit (fewest undetected errors).

#         Parameters
#         ----------
#         n          : number of physical qubits.
#         k          : number of logical qubits (default 1).
#         d          : target distance (default 3).
#         c_Z        : noise bias parameter (1.0 = symmetric).
#         p_I        : noise rate (0.9 = standard).
#         n_attempts : how many tries before giving up.
#         greedy     : if True use argmax (deterministic), else sample.

#         Returns
#         -------
#         dict with keys:
#             circuit   : list of gate tuples  [('H',2), ('CNOT',0,1), ...]
#             length    : number of gates
#             success   : bool — did it satisfy all KL conditions?
#             n_errors_detected : int
#             n_errors_total    : int
#         """
#         # temporarily override the env to fix the config we want
#         best_result = None
#         best_detected = -1

#         for attempt in range(n_attempts):
#             # cleanly reset the environment to the requested config
#             obs = self.env.reset_to(n, k, d, c_Z, p_I)

#             done = False
#             history = []
#             for _ in range(self.max_steps):
#                 mask = self.env.action_mask()
#                 action, _ = self.agent.select_action(
#                     obs, action_mask=mask, greedy=greedy)
#                 obs, reward, done, info = self.env.step(action)
#                 if not info.get("illegal", False):
#                     history.append(self.env.actions[action])
#                 if done:
#                     break

#             # count detected errors
#             from Encoders.Clifford_sim import kl_undetected_mask
#             mask_undet = kl_undetected_mask(
#                 self.env.errors, self.env.tab.G, n, exact=True)
#             n_detected = int((~mask_undet).sum())
#             n_total    = len(self.env.errors)
#             success    = bool(not mask_undet.any())

#             result = {
#                 "circuit":            list(history),
#                 "length":             len(history),
#                 "success":            success,
#                 "n_errors_detected":  n_detected,
#                 "n_errors_total":     n_total,
#                 "attempt":            attempt + 1,
#                 "generators":         self.env.tab.generators_str(),
#             }

#             if success:
#                 if best_result is None or len(history) < best_result["length"]:
#                     best_result = result
#             elif n_detected > best_detected:
#                 best_detected = n_detected
#                 if best_result is None or not best_result["success"]:
#                     best_result = result

#         return best_result

#     # ── validation ───────────────────────────────────────────────────

#     def validate(self, n, k=1, d=3, c_Z=1.0, p_I=0.9, verbose=True):
#         """
#         Discover a circuit and verify it using the exact KL check.
#         Prints a readable report.
#         """
#         if verbose:
#             print(f"\nValidating encoding for [[{n},{k},{d}]]"
#                   f"  c_Z={c_Z}  p_I={p_I}")
#             print("-" * 55)

#         result = self.encode(n, k, d, c_Z, p_I)

#         if verbose:
#             status = "✓ SUCCESS" if result["success"] else "✗ PARTIAL"
#             print(f"  {status}")
#             print(f"  Gates     : {len(result['circuit'])}")
#             circuit_str = "  →  ".join(
#                 f"CNOT({g[1]}→{g[2]})" if g[0] == "CNOT"
#                 else f"{g[0]}({g[1]})"
#                 for g in result["circuit"]
#             )
#             print(f"  Circuit   : {circuit_str}")
#             print(f"  Generators: {result['generators']}")
#             print(f"  Detected  : {result['n_errors_detected']}"
#                   f"/{result['n_errors_total']} errors")

#         return result

#     # ── save / load ──────────────────────────────────────────────────

#     def save(self, path: str):
#         """Save the trained policy and training history."""
#         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#         payload = {
#             "agent_state":    {
#                 "actor":  self.agent.actor.state_dict(),
#                 "critic": self.agent.critic.state_dict(),
#                 "train_step": self.agent.train_step,
#                 "env_steps":  self.agent.env_steps,
#             },
#             "config": {
#                 "n_max":         self.n_max,
#                 "obs_dim":       self.env.obs_dim,
#                 "num_actions":   self.env.num_actions,
#                 "p_I":           self.p_I,
#                 "max_steps":     self.max_steps,
#                 "success_bonus": self.success_bonus,
#                 "fail_penalty":  self.fail_penalty,
#                 "seed":          self.seed,
#             },
#             "best_circuits": self.best_circuits,
#             "history_len":   len(self.history),
#         }
#         with open(path, "wb") as f:
#             pickle.dump(payload, f)

#     @classmethod
#     def load(cls, path: str, **kwargs):
#         """Load a saved policy. kwargs override saved config values."""
#         with open(path, "rb") as f:
#             payload = pickle.load(f)
#         cfg = {**payload["config"], **kwargs}
#         agent = cls(
#             n_max=cfg["n_max"],
#             p_I=cfg["p_I"],
#             max_steps=cfg["max_steps"],
#             success_bonus=cfg["success_bonus"],
#             fail_penalty=cfg["fail_penalty"],
#             seed=cfg["seed"],
#         )
#         agent.agent.actor.load_state_dict(payload["agent_state"]["actor"])
#         agent.agent.critic.load_state_dict(payload["agent_state"]["critic"])
#         agent.agent.train_step = payload["agent_state"]["train_step"]
#         agent.agent.env_steps  = payload["agent_state"]["env_steps"]
#         agent.best_circuits    = payload.get("best_circuits", {})
#         return agent

#     # ── quick info ───────────────────────────────────────────────────

#     def info(self):
#         """Print a summary of what this agent knows."""
#         print("\nEncoding agent summary")
#         print(f"  n_max        : {self.n_max} qubits")
#         print(f"  obs_dim      : {self.env.obs_dim}")
#         print(f"  num_actions  : {self.env.num_actions}")
#         print(f"  train_steps  : {self.agent.train_step}")
#         print(f"  env_steps    : {self.agent.env_steps}")
#         print(f"  best circuits discovered so far:")
#         if not self.best_circuits:
#             print("    none yet — run agent.train() first")
#         for (n, k, d), info in sorted(self.best_circuits.items()):
#             print(f"    [[{n},{k},{d}]]  {info['length']} gates"
#                   f"  c_Z={info['c_Z']:.2f}")


# # =====================================================================
# # Run directly — quick demo
# # =====================================================================

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="QEC Encoding Agent")
#     parser.add_argument("--episodes", type=int, default=3000,
#                         help="number of training episodes (default 3000 for a quick demo)")
#     parser.add_argument("--save",     type=str, default="encoding_policy_new1.pkl",
#                         help="path to save the trained policy")
#     parser.add_argument("--load",     type=str, default=None,
#                         help="load a saved policy instead of training")
#     parser.add_argument("--n_max",    type=int, default=9,
#                         help="maximum number of physical qubits")
#     args = parser.parse_args()

#     if args.load:
#         print(f"Loading policy from {args.load}")
#         agent = EncodingAgent.load(args.load)
#         agent.info()
#     else:
#         agent = EncodingAgent(n_max=args.n_max, seed=0)
#         agent.train(
#             num_episodes=args.episodes,
#             log_every=200,
#             save_path=args.save,
#             save_every=1000,
#             verbose=True,
#         )

#     print("\n" + "=" * 65)
#     print("ENCODING EXAMPLES")
#     print("=" * 65)

#     configs = [
#         (3, 1, 2, 1.0, 0.9,  "3-qubit bit-flip (warmup)"),
#         (5, 1, 3, 1.0, 0.9,  "[[5,1,3]] perfect code"),
#         (5, 1, 3, 0.5, 0.9,  "[[5,1,3]] Z-biased noise"),
#         (7, 1, 3, 1.0, 0.9,  "[[7,1,3]] Steane-like"),
#         (9, 1, 3, 2.0, 0.9,  "[[9,1,3]] X-biased noise"),
#     ]

#     for n, k, d, c_Z, p_I, label in configs:
#         print(f"\n{label}")
#         result = agent.validate(n, k, d, c_Z, p_I, verbose=True)
"""
encoding_agent.py
=================
The QEC encoding agent — the single entry point for the full system.

What this file does
-------------------
  1. Defines EncodingAgent — a wrapper around A2CAgent + MetaQECEnv
     that handles training, saving, loading, and circuit discovery.

  2. Provides a clean API:

        agent = EncodingAgent()
        agent.train()                          # curriculum training
        circuit = agent.encode(n, k, d, c_Z, p_I)  # discover a circuit
        agent.save("policy.pkl")               # save the trained policy
        agent = EncodingAgent.load("policy.pkl")    # reload it later
        agent.validate(n, k, d, c_Z, p_I)     # verify the circuit works

  3. Can be run directly:
        python encoding_agent.py               # trains and saves

How the RL works (plain language)
----------------------------------
  INPUT  (every step):
    - The current circuit state as a flat binary vector
      (the check matrix G, flattened — all 0s and 1s)
    - 5 context numbers: n, k, d, c_Z, p_I  (normalized to [0,1])

  OUTPUT (every step):
    - One gate to apply: H(i), S(i), or CNOT(i→j)
    - The actor network produces a probability for each available gate
    - The agent samples one gate from those probabilities

  REWARD (every step):
    - r = −Σ λ_μ K_μ   (Eq. 10 of the paper)
    - K_μ = 1 if error E_μ is NOT detected, else 0
    - λ_μ = p_μ / max(p_μ)  (error weight from the noise model)
    - reward = 0 when ALL errors are detected  →  success
    - +20 bonus on success, −10 penalty on timeout

  LEARNING:
    - A2C with GAE: after each episode, compute advantages and
      update actor + critic weights
    - Curriculum: phases 0→3, each adding harder codes and wider
      noise ranges. The agent builds on what it learned earlier.

  AFTER TRAINING:
    - Call agent.encode(n, k, d, c_Z, p_I)
    - The agent runs one greedy episode and returns the gate sequence
    - That sequence is a real encoding circuit for a real quantum chip
"""

import os
import time
import pickle
import numpy as np
from typing import Optional

from Encoders.Clifford_sim import (
    all_pauli_strings_up_to_weight,
    pauli_strings_to_binary,
    kl_undetected_mask,
    validate_circuit,
)
from Encoders.meta_env import MetaQECEnv, CURRICULUM_PHASES, TOTAL_EPISODES
from Encoders.A2C_agent import A2CAgent, collect_episode


# =====================================================================
# The encoding agent
# =====================================================================

class EncodingAgent:
    """
    One agent that discovers encoding circuits for many codes and noise
    models. Trained once via a curriculum, usable forever after.

    Parameters
    ----------
    n_max        : largest number of physical qubits to support.
    hidden       : hidden layer size of the actor and critic networks.
    lr           : learning rate.
    gamma        : discount factor for future rewards.
    gae_lambda   : GAE lambda (0 = TD, 1 = Monte Carlo, 0.95 = paper).
    ent_coef     : entropy bonus coefficient (keeps the policy exploring).
    vf_coef      : critic loss coefficient.
    max_grad_norm: gradient clipping threshold.
    p_I          : base noise rate for the depolarizing channel.
    max_steps    : maximum gates the agent can apply per episode.
    success_bonus: reward added when all KL conditions are satisfied.
    fail_penalty : reward added when the step budget is exhausted.
    seed         : random seed.
    """

    def __init__(
        self,
        n_max=9,
        hidden=128,
        lr=2e-3,
        gamma=0.95,
        gae_lambda=0.95,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        p_I=0.9,
        max_steps=35,
        success_bonus=20.0,
        fail_penalty=-10.0,
        seed=0,
    ):
        self.n_max        = n_max
        self.p_I          = p_I
        self.max_steps    = max_steps
        self.success_bonus = success_bonus
        self.fail_penalty  = fail_penalty
        self.seed          = seed

        # build the environment — this defines obs_dim and num_actions
        self.env = MetaQECEnv(
            n_max=n_max,
            p_I=p_I,
            max_steps=max_steps,
            success_bonus=success_bonus,
            fail_penalty=fail_penalty,
            seed=seed,
        )

        # build the agent
        self.agent = A2CAgent(
            obs_dim=self.env.obs_dim,
            n_actions=self.env.num_actions,
            hidden=hidden,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            seed=seed,
        )

        # training history
        self.history = []
        self.best_circuits = {}   # key: (n, k, d)  value: shortest circuit found

    # ── training ────────────────────────────────────────────────────

    def train(
        self,
        num_episodes=None,
        log_every=200,
        save_path=None,
        save_every=1000,
        verbose=True,
    ):
        """
        Run the full curriculum training.

        Parameters
        ----------
        num_episodes : total episodes to train (default = full curriculum).
        log_every    : print a progress line every N episodes.
        save_path    : if given, save the policy to this path.
        save_every   : save a checkpoint every N episodes.
        verbose      : print progress.
        """
        if num_episodes is None:
            num_episodes = TOTAL_EPISODES

        if verbose:
            print("=" * 65)
            print("ENCODING AGENT — curriculum training")
            print(f"  obs_dim    : {self.env.obs_dim}")
            print(f"  num_actions: {self.env.num_actions}")
            print(f"  n_max      : {self.n_max}")
            print(f"  episodes   : {num_episodes}")
            print("=" * 65)

        recent_rewards  = []
        recent_success  = []
        recent_lengths  = []
        t_start = time.time()

        for ep in range(1, num_episodes + 1):

            # collect one episode
            batch, info = collect_episode(self.env, self.agent)
            self.agent.update(batch)

            # track stats
            recent_rewards.append(info["total_reward"])
            recent_success.append(float(info["success"]))
            recent_lengths.append(info["length"])

            # track best circuits per code
            if info["success"]:
                key = (self.env.n, self.env.k, self.env.d)
                prev_best = self.best_circuits.get(key, {}).get("length", 999)
                if info["length"] < prev_best:
                    self.best_circuits[key] = {
                        "circuit": list(info["history"]),
                        "length":  info["length"],
                        "c_Z":     self.env.c_Z,
                        "p_I":     self.p_I,
                    }

            self.history.append({
                "episode":       ep,
                "total_reward":  info["total_reward"],
                "success":       info["success"],
                "length":        info["length"],
                "n":             self.env.n,
                "d":             self.env.d,
                "c_Z":           self.env.c_Z,
                "phase":         self.env._phase_idx,
            })

            # print progress
            if verbose and ep % log_every == 0:
                w = min(log_every, len(recent_success))
                sr  = float(np.mean(recent_success[-w:]))
                avg_r = float(np.mean(recent_rewards[-w:]))
                avg_l = float(np.mean(recent_lengths[-w:]))
                elapsed = time.time() - t_start
                phase_name = self.env.phase_name
                print(
                    f"  ep {ep:6d}/{num_episodes}"
                    f"  phase {self.env._phase_idx}"
                    f"  SR {sr:.2f}"
                    f"  avg_reward {avg_r:7.2f}"
                    f"  avg_len {avg_l:5.1f}"
                    f"  elapsed {elapsed:6.0f}s"
                    f"  [{phase_name}]"
                )

            # save checkpoint
            if save_path and ep % save_every == 0:
                self.save(save_path)
                if verbose:
                    print(f"  → checkpoint saved to {save_path}")

        if verbose:
            elapsed = time.time() - t_start
            print(f"\nTraining complete in {elapsed:.0f}s")
            print(f"Best circuits discovered:")
            for (n, k, d), info in sorted(self.best_circuits.items()):
                print(f"  [[{n},{k},{d}]]  {info['length']} gates"
                      f"  c_Z={info['c_Z']:.2f}")

        if save_path:
            self.save(save_path)

    # ── circuit discovery ────────────────────────────────────────────

    def encode(
        self,
        n: int,
        k: int = 1,
        d: int = 3,
        c_Z: float = 1.0,
        p_I: float = 0.9,
        n_attempts: int = 10,
        greedy: bool = True,
        prune: bool = True,
    ):
        """
        Discover an encoding circuit for the given code and noise model.

        The agent runs up to `n_attempts` episodes and returns the
        shortest successful circuit found. If no attempt succeeds,
        returns the best partial circuit (fewest undetected errors).

        Parameters
        ----------
        n          : number of physical qubits.
        k          : number of logical qubits (default 1).
        d          : target distance (default 3).
        c_Z        : noise bias parameter (1.0 = symmetric).
        p_I        : noise rate (0.9 = standard).
        n_attempts : how many tries before giving up.
        greedy     : if True use argmax (deterministic), else sample.
        prune      : if True, remove redundant gates from a successful
                     circuit (post-processing, does not affect the policy).

        Returns
        -------
        dict with keys:
            circuit   : list of gate tuples  [('H',2), ('CNOT',0,1), ...]
                        (pruned if prune=True and the circuit succeeded)
            length    : number of gates in `circuit`
            success   : bool — did it satisfy all KL conditions?
            n_errors_detected : int
            n_errors_total    : int
            circuit_raw : the unpruned circuit (only if pruning happened)
            length_raw  : length before pruning (only if pruning happened)
        """
        # temporarily override the env to fix the config we want
        best_result = None
        best_detected = -1

        for attempt in range(n_attempts):
            # cleanly reset the environment to the requested config
            obs = self.env.reset_to(n, k, d, c_Z, p_I)

            done = False
            history = []
            for _ in range(self.max_steps):
                mask = self.env.action_mask()
                action, _ = self.agent.select_action(
                    obs, action_mask=mask, greedy=greedy)
                obs, reward, done, info = self.env.step(action)
                if not info.get("illegal", False):
                    history.append(self.env.actions[action])
                if done:
                    break

            # count detected errors
            from Encoders.Clifford_sim import kl_undetected_mask
            mask_undet = kl_undetected_mask(
                self.env.errors, self.env.tab.G, n, exact=True)
            n_detected = int((~mask_undet).sum())
            n_total    = len(self.env.errors)
            success    = bool(not mask_undet.any())

            result = {
                "circuit":            list(history),
                "length":             len(history),
                "success":            success,
                "n_errors_detected":  n_detected,
                "n_errors_total":     n_total,
                "attempt":            attempt + 1,
                "generators":         self.env.tab.generators_str(),
            }

            if success:
                if best_result is None or len(history) < best_result["length"]:
                    best_result = result
            elif n_detected > best_detected:
                best_detected = n_detected
                if best_result is None or not best_result["success"]:
                    best_result = result

        # ── post-processing: prune redundant gates from a valid circuit ──
        if prune and best_result is not None and best_result["success"]:
            from Encoders.circuit_pruner import prune_circuit
            from Encoders.Clifford_sim import validate_circuit
            error_strings = self.env.error_strings
            raw_circuit = best_result["circuit"]
            pruned = prune_circuit(n, k, raw_circuit, error_strings)
            if len(pruned) < len(raw_circuit):
                # recompute generators for the pruned circuit
                res = validate_circuit(n, k, pruned, error_strings)
                best_result["circuit_raw"] = raw_circuit
                best_result["length_raw"]  = len(raw_circuit)
                best_result["circuit"]     = pruned
                best_result["length"]      = len(pruned)
                best_result["generators"]  = res["generators"]

        return best_result

    # ── validation ───────────────────────────────────────────────────

    def validate(self, n, k=1, d=3, c_Z=1.0, p_I=0.9, verbose=True):
        """
        Discover a circuit and verify it using the exact KL check.
        Prints a readable report.
        """
        if verbose:
            print(f"\nValidating encoding for [[{n},{k},{d}]]"
                  f"  c_Z={c_Z}  p_I={p_I}")
            print("-" * 55)

        result = self.encode(n, k, d, c_Z, p_I)

        if verbose:
            status = "✓ SUCCESS" if result["success"] else "✗ PARTIAL"
            print(f"  {status}")
            print(f"  Gates     : {len(result['circuit'])}")
            circuit_str = "  →  ".join(
                f"CNOT({g[1]}→{g[2]})" if g[0] == "CNOT"
                else f"{g[0]}({g[1]})"
                for g in result["circuit"]
            )
            print(f"  Circuit   : {circuit_str}")
            print(f"  Generators: {result['generators']}")
            print(f"  Detected  : {result['n_errors_detected']}"
                  f"/{result['n_errors_total']} errors")

        return result

    # ── save / load ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save the trained policy and training history."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "agent_state":    {
                "actor":  self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "train_step": self.agent.train_step,
                "env_steps":  self.agent.env_steps,
            },
            "config": {
                "n_max":         self.n_max,
                "obs_dim":       self.env.obs_dim,
                "num_actions":   self.env.num_actions,
                "p_I":           self.p_I,
                "max_steps":     self.max_steps,
                "success_bonus": self.success_bonus,
                "fail_penalty":  self.fail_penalty,
                "seed":          self.seed,
            },
            "best_circuits": self.best_circuits,
            "history_len":   len(self.history),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved policy. kwargs override saved config values."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        cfg = {**payload["config"], **kwargs}
        agent = cls(
            n_max=cfg["n_max"],
            p_I=cfg["p_I"],
            max_steps=cfg["max_steps"],
            success_bonus=cfg["success_bonus"],
            fail_penalty=cfg["fail_penalty"],
            seed=cfg["seed"],
        )
        agent.agent.actor.load_state_dict(payload["agent_state"]["actor"])
        agent.agent.critic.load_state_dict(payload["agent_state"]["critic"])
        agent.agent.train_step = payload["agent_state"]["train_step"]
        agent.agent.env_steps  = payload["agent_state"]["env_steps"]
        agent.best_circuits    = payload.get("best_circuits", {})
        return agent

    # ── quick info ───────────────────────────────────────────────────

    def info(self):
        """Print a summary of what this agent knows."""
        print("\nEncoding agent summary")
        print(f"  n_max        : {self.n_max} qubits")
        print(f"  obs_dim      : {self.env.obs_dim}")
        print(f"  num_actions  : {self.env.num_actions}")
        print(f"  train_steps  : {self.agent.train_step}")
        print(f"  env_steps    : {self.agent.env_steps}")
        print(f"  best circuits discovered so far:")
        if not self.best_circuits:
            print("    none yet — run agent.train() first")
        for (n, k, d), info in sorted(self.best_circuits.items()):
            print(f"    [[{n},{k},{d}]]  {info['length']} gates"
                  f"  c_Z={info['c_Z']:.2f}")


# =====================================================================
# Run directly — quick demo
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QEC Encoding Agent")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="number of training episodes (default 3000 for a quick demo)")
    parser.add_argument("--save",     type=str, default="encoding_policy.pkl",
                        help="path to save the trained policy")
    parser.add_argument("--load",     type=str, default=None,
                        help="load a saved policy instead of training")
    parser.add_argument("--n_max",    type=int, default=9,
                        help="maximum number of physical qubits")
    args = parser.parse_args()

    if args.load:
        print(f"Loading policy from {args.load}")
        agent = EncodingAgent.load(args.load)
        agent.info()
    else:
        agent = EncodingAgent(n_max=args.n_max, seed=0)
        agent.train(
            num_episodes=args.episodes,
            log_every=200,
            save_path=args.save,
            save_every=1000,
            verbose=True,
        )

    print("\n" + "=" * 65)
    print("ENCODING EXAMPLES")
    print("=" * 65)

    configs = [
        (3, 1, 2, 1.0, 0.9,  "3-qubit bit-flip (warmup)"),
        (5, 1, 3, 1.0, 0.9,  "[[5,1,3]] perfect code"),
        (5, 1, 3, 0.5, 0.9,  "[[5,1,3]] Z-biased noise"),
        (7, 1, 3, 1.0, 0.9,  "[[7,1,3]] Steane-like"),
        (9, 1, 3, 2.0, 0.9,  "[[9,1,3]] X-biased noise"),
    ]

    for n, k, d, c_Z, p_I, label in configs:
        print(f"\n{label}")
        result = agent.validate(n, k, d, c_Z, p_I, verbose=True)