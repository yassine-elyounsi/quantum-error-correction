"""
Demo: train an A2C + PPR agent to discover QEC encoding circuits from scratch,
using a Knill-Laflamme reward, in the spirit of Olle et al. (2024).

Two experiments:

  (1) [[3, 1]] bit-flip repetition code  (X errors only).
      Expected: the agent rediscovers the canonical 2-gate sequence
                CNOT(0 -> 1), CNOT(0 -> 2)        (paper Appendix A / Fig. 8)

  (2) [[5, 1, 3]] perfect code  (all weight-1 Pauli errors).
      Expected: the agent finds a short encoding on 5 qubits that detects
                all 15 weight-1 errors.

Run:    python train.py
"""

import time
import numpy as np
from Encoders.encode_env import QECEnv
from Encoders.A2C_agent import A2CAgent, collect_episode
from Encoders.Clifford_sim import (
    x_type_errors_up_to_weight, all_pauli_strings_up_to_weight,
    validate_circuit,
)


def train_one(env, agent, num_episodes, log_every=100, verbose=True):
    """Train one A2C agent on one env and track the shortest successful circuit."""
    best_circuit = None
    best_len = float('inf')
    recent_success = []
    for ep in range(num_episodes):
        batch, info = collect_episode(env, agent)
        agent.update(batch)
        recent_success.append(1.0 if info['success'] else 0.0)
        if info['success'] and info['length'] < best_len:
            best_len = info['length']
            best_circuit = list(info['history'])
        if verbose and (ep + 1) % log_every == 0:
            sr = float(np.mean(recent_success[-log_every:]))
            print(f"  ep {ep+1:5d}  success-rate {sr:.2f}  "
                  f"psi {agent._psi():.2f}  "
                  f"best-len {best_len if best_len < float('inf') else '-'}")
    return best_circuit, best_len


def format_circuit(history):
    parts = []
    for g in history:
        if g[0] in ('CNOT', 'CX'):
            parts.append(f"CNOT({g[1]} -> {g[2]})")
        elif g[0] == 'CZ':
            parts.append(f"CZ({g[1]}, {g[2]})")
        else:
            parts.append(f"{g[0]}({g[1]})")
    return '   '.join(parts)


# =====================================================================
# Experiment 1: [[3, 1]] bit-flip code
# =====================================================================
print("=" * 68)
print("Experiment 1:  [[3, 1]] bit-flip repetition code")
print("Target errors: X-type, weights 1-2  =>  XII, IXI, IIX, XXI, XIX, IXX")
print("=" * 68)

env1 = QECEnv(
    n=3, k=1,
    gate_set=('CNOT',), connectivity='directed_all',
    max_steps=6,
    error_strings=x_type_errors_up_to_weight(3, 2),
    success_bonus=10.0, fail_penalty=-5.0,
)
print(f"  available actions: {env1.actions}")
print(f"  target error set : {env1.error_strings}\n")

agent1 = A2CAgent(
    obs_dim=env1.obs_dim,
    n_actions=env1.num_actions,
    hidden=32, lr=5e-3, gamma=0.9,
    ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
    seed=0,
)

t0 = time.time()
best_hist, best_len = train_one(env1, agent1, num_episodes=400, log_every=100)
t1 = time.time()
print(f"\n  -> trained in {t1-t0:.1f}s")
print(f"  -> shortest encoding circuit found ({best_len} gates):")
print(f"     {format_circuit(best_hist)}")
print(f"     expected (canonical):  CNOT(0 -> 1)   CNOT(0 -> 2)")

result1 = validate_circuit(n=3, k=1, gate_list=best_hist,
                           error_strings=env1.error_strings,
                           exact=True, verbose=False)
print(f"  -> verified: {result1['num_detected']}/{result1['num_total']} errors detected")
print(f"  -> final generators: {result1['generators']}\n")


# =====================================================================
# Experiment 2: [[5, 1, 3]] perfect code  (with PPR transfer from Exp. 1)
# =====================================================================
print("=" * 68)
print("Experiment 2:  [[5, 1, 3]] perfect code  (PPR-assisted)")
print("Target errors: all weight-1 Paulis on 5 qubits  (15 errors)")
print("=" * 68)

env2 = QECEnv(
    n=5, k=1,
    gate_set=('H', 'CNOT'), connectivity='directed_all',
    max_steps=12,
    error_strings=all_pauli_strings_up_to_weight(5, 1),
    success_bonus=20.0, fail_penalty=-5.0,
)
print(f"  action space size: {env2.num_actions}")
print(f"  target error set : {len(env2.error_strings)} errors\n")

# A simple library policy: half the time pick a CNOT-from-qubit-0,
# half the time pick a Hadamard. This is the kind of cheap hint PPR
# exploits early in training, then phases out.
_rng = np.random.default_rng(1)
def library_policy(obs):
    if _rng.random() < 0.5:
        candidates = [i for i, a in enumerate(env2.actions)
                      if a[0] in ('CNOT', 'CX') and a[1] == 0]
        return int(_rng.choice(candidates))
    candidates = [i for i, a in enumerate(env2.actions) if a[0] == 'H']
    return int(_rng.choice(candidates))

agent2 = A2CAgent(
    obs_dim=env2.obs_dim,
    n_actions=env2.num_actions,
    hidden=64, lr=2e-3, gamma=0.95,
    ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5,
    library_policy=library_policy,
    psi_start=0.5, psi_end=0.05, psi_decay_steps=8000,
    seed=42,
)

t0 = time.time()
best_hist2, best_len2 = train_one(env2, agent2, num_episodes=3000, log_every=300)
t1 = time.time()
print(f"\n  -> trained in {t1-t0:.1f}s")
if best_hist2 is not None:
    print(f"  -> shortest encoding found ({best_len2} gates):")
    print(f"     {format_circuit(best_hist2)}")
    result2 = validate_circuit(n=5, k=1, gate_list=best_hist2,
                               error_strings=env2.error_strings,
                               exact=True, verbose=False)
    print(f"  -> verified: {result2['num_detected']}/{result2['num_total']} "
          f"weight-1 errors detected")
    print(f"  -> final generators: {result2['generators']}")
else:
    print("  -> no successful encoding within training budget. "
          "Try more episodes or higher entropy bonus.")

print("\nDone.")