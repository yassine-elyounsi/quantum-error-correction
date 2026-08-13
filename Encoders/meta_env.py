# """
# MetaQECEnv  —  one environment, many codes, many noise models.

# At the start of every episode the environment:
#   1. Samples a (n, k, d, c_Z) configuration from the current curriculum phase.
#   2. Builds the error set for that configuration.
#   3. Computes reward weights from the noise model.
#   4. Resets the Clifford tableau.

# The observation appended to the check matrix is:
#     [n_norm, k_norm, d_norm, c_Z_norm]
# where each value is linearly normalized to [0, 1] over the training range.

# This lets one agent learn patterns that transfer across code sizes and noise
# models.  The action space uses RELATIVE qubit indices so it stays the same
# size regardless of n.  We support up to n_max physical qubits.

# Relative action vocabulary (fixed size regardless of current n):
#     H(i)          for i in 0 .. n_max-1
#     S(i)          for i in 0 .. n_max-1
#     CNOT(i, i+1)  for i in 0 .. n_max-2   (nearest-neighbour directed)
#     CNOT(i, i+2)  for i in 0 .. n_max-3   (next-nearest-neighbour)
#     CNOT(0, i)    for i in 1 .. n_max-1   (star: qubit 0 talks to everyone)

# Actions that reference qubits >= current n are masked (the agent's
# action_mask() method returns False for them so they are never sampled).
# """

# import numpy as np
# from Encoders.Clifford_sim import (
#     CliffordTableau,
#     pauli_strings_to_binary,
#     kl_undetected_mask,
#     weighted_kl_sum,
#     all_pauli_strings_up_to_weight,
# )
# from Encoders.encode_env import paper_lambda_weights


# # ── Curriculum: ordered list of (n, k, d) configurations ────────────
# # We progress from simplest to hardest.  Each phase adds new configs
# # on top of the previous ones (the agent keeps seeing the easy cases).

# CURRICULUM_PHASES = [
#     # phase 0  — warmup: tiny codes, symmetric noise only
#     {"configs": [(3, 1, 2)],
#      "c_Z_range": (1.0, 1.0),
#      "episodes": 10000},

#     # phase 1  — add the [[5,1,3]] perfect code, slight noise variation
#     {"configs": [(3, 1, 2), (5, 1, 3)],
#      "c_Z_range": (0.8, 1.2),
#      "episodes": 20000},

#     # phase 2  — add [[7,1,3]] Steane, wider noise range
#     {"configs": [(3, 1, 2), (5, 1, 3), (7, 1, 3)],
#      "c_Z_range": (0.5, 1.5),
#      "episodes": 30000},

#     # phase 3  — add [[9,1,3]] Shor, full asymmetric noise
#     {"configs": [(3, 1, 2), (5, 1, 3), (7, 1, 3), (9, 1, 3)],
#      "c_Z_range": (0.5, 2.0),
#      "episodes": 40000},
# ]

# TOTAL_EPISODES = sum(p["episodes"] for p in CURRICULUM_PHASES)

# # ── Meta-environment ─────────────────────────────────────────────────

# class MetaQECEnv:
#     """
#     One environment that covers all codes in the curriculum.

#     Parameters
#     ----------
#     n_max        : maximum physical qubits across all codes.
#     p_I          : single-qubit identity probability for the noise model.
#     max_steps    : maximum gates per episode (set large enough for the
#                    hardest code in the curriculum).
#     success_bonus: reward added when all KL conditions are satisfied.
#     fail_penalty : reward added when the step budget is exhausted.
#     seed         : random seed.
#     """

#     def __init__(
#         self,
#         n_max=9,
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
#         self.rng = np.random.default_rng(seed)

#         # build the unified (relative) action list once
#         self.actions, self.action_kinds = self._build_actions(n_max)
#         self.num_actions = len(self.actions)

#         # obs = flattened G (n_max-k) × 2*n_max  +  5 context scalars
#         # We always allocate for n_max qubits and pad G with zeros.
#         self.G_dim   = (n_max - 1) * (2 * n_max)   # k=1 always
#         self.obs_dim = self.G_dim + 5               # +n,k,d,cZ,pI

#         # curriculum tracking
#         self.episode_count = 0
#         self._phase_idx    = 0
#         self._update_phase()

#         # will be set by reset()
#         self.n = self.k = self.d = None
#         self.c_Z = 1.0
#         self.tab = None
#         self.errors = None
#         self.lambda_weight = None
#         self.error_strings = None
#         self.step_count = 0
#         self.history = []

#     # ── action vocabulary ──────────────────────────────────────────

#     def _build_actions(self, n_max):
#         """
#         Build a fixed-size list of relative gate actions.
#         Actions that reference qubit indices >= current n are masked at
#         runtime but always exist in the list.
#         """
#         actions, kinds = [], []
#         # H and S on every qubit
#         for i in range(n_max):
#             actions.append(("H", i)); kinds.append("H")
#         for i in range(n_max):
#             actions.append(("S", i)); kinds.append("S")
#         # nearest-neighbour CNOTs i -> i+1
#         for i in range(n_max - 1):
#             actions.append(("CNOT", i, i + 1)); kinds.append("CNOT")
#         # next-nearest-neighbour CNOTs i -> i+2
#         for i in range(n_max - 2):
#             actions.append(("CNOT", i, i + 2)); kinds.append("CNOT")
#         # star CNOTs: qubit 0 -> every other qubit
#         for i in range(1, n_max):
#             actions.append(("CNOT", 0, i)); kinds.append("CNOT")
#         return actions, kinds

#     def action_mask(self):
#         """
#         Boolean mask of length num_actions.
#         An action is legal iff all its qubit indices are < current n.
#         """
#         mask = np.ones(self.num_actions, dtype=bool)
#         for idx, a in enumerate(self.actions):
#             if a[0] in ("H", "S"):
#                 if a[1] >= self.n:
#                     mask[idx] = False
#             elif a[0] == "CNOT":
#                 if a[1] >= self.n or a[2] >= self.n:
#                     mask[idx] = False
#         return mask

#     # ── curriculum ────────────────────────────────────────────────

#     def _update_phase(self):
#         cumulative = 0
#         for i, phase in enumerate(CURRICULUM_PHASES):
#             cumulative += phase["episodes"]
#             if self.episode_count < cumulative:
#                 self._phase_idx = i
#                 break
#         else:
#             self._phase_idx = len(CURRICULUM_PHASES) - 1
#         self._phase = CURRICULUM_PHASES[self._phase_idx]

#     def _sample_config(self):
#         """Sample (n, k, d, c_Z) from the current curriculum phase."""
#         configs = self._phase["configs"]
#         idx = self.rng.integers(len(configs))
#         n, k, d = configs[idx]
#         lo, hi = self._phase["c_Z_range"]
#         c_Z = float(self.rng.uniform(lo, hi))
#         return n, k, d, c_Z

#     # ── gym interface ─────────────────────────────────────────────

#     def reset(self):
#         self.episode_count += 1
#         self._update_phase()

#         self.n, self.k, self.d, self.c_Z = self._sample_config()
#         self.tab = CliffordTableau(self.n, self.k)
#         self.step_count = 0
#         self.history = []

#         # build error set for (n, d)
#         self.error_strings = all_pauli_strings_up_to_weight(
#             self.n, self.d - 1)
#         self.errors = pauli_strings_to_binary(self.error_strings, self.n)

#         # reward weights from the noise model
#         self.lambda_weight = paper_lambda_weights(
#             self.error_strings, self.n, self.p_I, self.c_Z
#         ).astype(np.float32)

#         return self._obs()

#     def _obs(self):
#         # pad the check matrix to n_max qubits with zeros
#         n_rows = self.n - self.k
#         n_max  = self.n_max
#         G_padded = np.zeros((n_max - 1, 2 * n_max), dtype=np.float32)
#         G_padded[:n_rows, :2 * self.n] = self.tab.G
#         flat = G_padded.flatten()

#         # normalised context scalars  (all in [0,1])
#         n_norm  = (self.n   - 3) / (self.n_max - 3) if self.n_max > 3 else 0.0
#         k_norm  = float(self.k - 1)                  # always 0 for k=1
#         d_norm  = (self.d   - 2) / 3.0               # d in {2,3,5} → 0,.33,1
#         cZ_norm = (self.c_Z - 0.5) / 1.5             # cZ in [0.5,2] → [0,1]
#         pI_norm = (self.p_I - 0.75) / 0.25           # pI in [0.75,1] → [0,1]

#         ctx = np.array([n_norm, k_norm, d_norm, cZ_norm, pI_norm],
#                        dtype=np.float32)
#         return np.concatenate([flat, ctx])

#     def reset_to(self, n, k, d, c_Z, p_I=None):
#         """
#         Reset the environment to a specific (n, k, d, c_Z, p_I) config.
#         Used by the encoding agent at inference time.
#         """
#         self.n   = n
#         self.k   = k
#         self.d   = d
#         self.c_Z = c_Z
#         if p_I is not None:
#             self.p_I = p_I
#         self.tab        = CliffordTableau(n, k)
#         self.step_count = 0
#         self.history    = []

#         self.error_strings = all_pauli_strings_up_to_weight(n, d - 1)
#         self.errors        = pauli_strings_to_binary(self.error_strings, n)
#         self.lambda_weight = paper_lambda_weights(
#             self.error_strings, n, self.p_I, c_Z
#         ).astype(np.float32)
#         return self._obs()

#     def _apply_action(self, gate):
#         op = gate[0]
#         if   op == "H":    self.tab.H(gate[1])
#         elif op == "S":    self.tab.S(gate[1])
#         elif op == "CNOT": self.tab.CX(gate[1], gate[2])

#     def step(self, action_idx):
#         mask = self.action_mask()
#         if not mask[action_idx]:
#             self.step_count += 1
#             done = self.step_count >= self.max_steps
#             return self._obs(), -1.0 + (self.fail_penalty if done else 0.0), done, {
#                 "kl_sum": float("nan"), "success": False, "illegal": True,
#                 "n": self.n, "d": self.d, "c_Z": self.c_Z,
#             }

#         gate = self.actions[action_idx]
#         self._apply_action(gate)
#         self.history.append(gate)
#         self.step_count += 1

#         kl = weighted_kl_sum(
#             self.errors, self.lambda_weight, self.tab.G, self.n, exact=True)
#         all_detected = not kl_undetected_mask(
#             self.errors, self.tab.G, self.n, exact=True).any()

#         reward = -float(kl)
#         done = False
#         if all_detected:
#             reward += self.success_bonus
#             done = True
#         elif self.step_count >= self.max_steps:
#             reward += self.fail_penalty
#             done = True

#         return self._obs(), reward, done, {
#             "kl_sum": float(kl), "success": bool(all_detected),
#             "illegal": False, "n": self.n, "d": self.d, "c_Z": self.c_Z,
#         }

#     def render_history(self):
#         out = []
#         for g in self.history:
#             if g[0] == "CNOT":
#                 out.append(f"CNOT({g[1]}→{g[2]})")
#             else:
#                 out.append(f"{g[0]}({g[1]})")
#         return out

#     def seed(self, s):
#         self.rng = np.random.default_rng(s)

#     @property
#     def phase_name(self):
#         codes = self._phase["configs"]
#         return f"phase {self._phase_idx}: {[f'[[{n},{k},{d}]]' for n,k,d in codes]}"
"""
MetaQECEnv  —  one environment, many codes, many noise models.

At the start of every episode the environment:
  1. Samples a (n, k, d, c_Z) configuration from the current curriculum phase.
  2. Builds the error set for that configuration.
  3. Computes reward weights from the noise model.
  4. Resets the Clifford tableau.

The observation appended to the check matrix is:
    [n_norm, k_norm, d_norm, c_Z_norm]
where each value is linearly normalized to [0, 1] over the training range.

This lets one agent learn patterns that transfer across code sizes and noise
models.  The action space uses RELATIVE qubit indices so it stays the same
size regardless of n.  We support up to n_max physical qubits.

Relative action vocabulary (fixed size regardless of current n):
    H(i)          for i in 0 .. n_max-1
    S(i)          for i in 0 .. n_max-1
    CNOT(i, i+1)  for i in 0 .. n_max-2   (nearest-neighbour directed)
    CNOT(i, i+2)  for i in 0 .. n_max-3   (next-nearest-neighbour)
    CNOT(0, i)    for i in 1 .. n_max-1   (star: qubit 0 talks to everyone)

Actions that reference qubits >= current n are masked (the agent's
action_mask() method returns False for them so they are never sampled).
"""

import numpy as np
from Encoders.Clifford_sim import (
    CliffordTableau,
    pauli_strings_to_binary,
    kl_undetected_mask,
    weighted_kl_sum,
    all_pauli_strings_up_to_weight,
    x_type_errors_up_to_weight,
)
from Encoders.encode_env import paper_lambda_weights


def build_error_set(n, k, d):
    """
    Return the list of target error strings for a given code.

    Special case: the [[3,1,2]] code is the 3-qubit *repetition* code,
    which physically protects against bit-flips (X errors) only — it is
    mathematically impossible for any [[3,1]] code to detect all X, Y and
    Z single-qubit errors (only 2 stabilizers => at most 4 syndromes for
    9 errors).  So for [[3,1,2]] we target X-type errors, matching the
    paper's Appendix A.  Every other code uses the full Pauli error set.
    """
    if (n, k, d) == (3, 1, 2):
        return x_type_errors_up_to_weight(n, d - 1)
    return all_pauli_strings_up_to_weight(n, d - 1)


# ── Curriculum: ordered list of (n, k, d) configurations ────────────
# We progress from simplest to hardest.  Each phase adds new configs
# on top of the previous ones (the agent keeps seeing the easy cases).

CURRICULUM_PHASES = [
    # phase 0  — warmup: tiny codes, symmetric noise only
    {"configs": [(3, 1, 2)],
     "c_Z_range": (1.0, 1.0),
     "episodes": 10000},

    # phase 1  — add the [[5,1,3]] perfect code, slight noise variation
    {"configs": [(3, 1, 2), (5, 1, 3)],
     "c_Z_range": (0.8, 1.2),
     "episodes": 20000},

    # phase 2  — add [[7,1,3]] Steane, wider noise range
    {"configs": [(3, 1, 2), (5, 1, 3), (7, 1, 3)],
     "c_Z_range": (0.5, 1.5),
     "episodes": 30000},

    # phase 3  — add [[9,1,3]] Shor, full asymmetric noise
    {"configs": [(3, 1, 2), (5, 1, 3), (7, 1, 3), (9, 1, 3)],
     "c_Z_range": (0.5, 2.0),
     "episodes": 40000},
]

TOTAL_EPISODES = sum(p["episodes"] for p in CURRICULUM_PHASES)

# ── Meta-environment ─────────────────────────────────────────────────

class MetaQECEnv:
    """
    One environment that covers all codes in the curriculum.

    Parameters
    ----------
    n_max        : maximum physical qubits across all codes.
    p_I          : single-qubit identity probability for the noise model.
    max_steps    : maximum gates per episode (set large enough for the
                   hardest code in the curriculum).
    success_bonus: reward added when all KL conditions are satisfied.
    fail_penalty : reward added when the step budget is exhausted.
    seed         : random seed.
    """

    def __init__(
        self,
        n_max=9,
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
        self.rng = np.random.default_rng(seed)

        # build the unified (relative) action list once
        self.actions, self.action_kinds = self._build_actions(n_max)
        self.num_actions = len(self.actions)

        # obs = flattened G (n_max-k) × 2*n_max  +  5 context scalars
        # We always allocate for n_max qubits and pad G with zeros.
        self.G_dim   = (n_max - 1) * (2 * n_max)   # k=1 always
        self.obs_dim = self.G_dim + 5               # +n,k,d,cZ,pI

        # curriculum tracking
        self.episode_count = 0
        self._phase_idx    = 0
        self._update_phase()

        # will be set by reset()
        self.n = self.k = self.d = None
        self.c_Z = 1.0
        self.tab = None
        self.errors = None
        self.lambda_weight = None
        self.error_strings = None
        self.step_count = 0
        self.history = []

    # ── action vocabulary ──────────────────────────────────────────

    def _build_actions(self, n_max):
        """
        Build a fixed-size list of relative gate actions.
        Actions that reference qubit indices >= current n are masked at
        runtime but always exist in the list.
        """
        actions, kinds = [], []
        # H and S on every qubit
        for i in range(n_max):
            actions.append(("H", i)); kinds.append("H")
        for i in range(n_max):
            actions.append(("S", i)); kinds.append("S")
        # nearest-neighbour CNOTs i -> i+1
        for i in range(n_max - 1):
            actions.append(("CNOT", i, i + 1)); kinds.append("CNOT")
        # next-nearest-neighbour CNOTs i -> i+2
        for i in range(n_max - 2):
            actions.append(("CNOT", i, i + 2)); kinds.append("CNOT")
        # star CNOTs: qubit 0 -> every other qubit
        for i in range(1, n_max):
            actions.append(("CNOT", 0, i)); kinds.append("CNOT")
        return actions, kinds

    def action_mask(self):
        """
        Boolean mask of length num_actions.
        An action is legal iff all its qubit indices are < current n.
        """
        mask = np.ones(self.num_actions, dtype=bool)
        for idx, a in enumerate(self.actions):
            if a[0] in ("H", "S"):
                if a[1] >= self.n:
                    mask[idx] = False
            elif a[0] == "CNOT":
                if a[1] >= self.n or a[2] >= self.n:
                    mask[idx] = False
        return mask

    # ── curriculum ────────────────────────────────────────────────

    def _update_phase(self):
        cumulative = 0
        for i, phase in enumerate(CURRICULUM_PHASES):
            cumulative += phase["episodes"]
            if self.episode_count < cumulative:
                self._phase_idx = i
                break
        else:
            self._phase_idx = len(CURRICULUM_PHASES) - 1
        self._phase = CURRICULUM_PHASES[self._phase_idx]

    def _sample_config(self):
        """Sample (n, k, d, c_Z) from the current curriculum phase."""
        configs = self._phase["configs"]
        idx = self.rng.integers(len(configs))
        n, k, d = configs[idx]
        lo, hi = self._phase["c_Z_range"]
        c_Z = float(self.rng.uniform(lo, hi))
        return n, k, d, c_Z

    # ── gym interface ─────────────────────────────────────────────

    def reset(self):
        self.episode_count += 1
        self._update_phase()

        self.n, self.k, self.d, self.c_Z = self._sample_config()
        self.tab = CliffordTableau(self.n, self.k)
        self.step_count = 0
        self.history = []

        # build error set for this code (X-only for [[3,1,2]] repetition)
        self.error_strings = build_error_set(self.n, self.k, self.d)
        self.errors = pauli_strings_to_binary(self.error_strings, self.n)

        # reward weights from the noise model
        self.lambda_weight = paper_lambda_weights(
            self.error_strings, self.n, self.p_I, self.c_Z
        ).astype(np.float32)

        return self._obs()

    def _obs(self):
        # pad the check matrix to n_max qubits with zeros
        n_rows = self.n - self.k
        n_max  = self.n_max
        G_padded = np.zeros((n_max - 1, 2 * n_max), dtype=np.float32)
        G_padded[:n_rows, :2 * self.n] = self.tab.G
        flat = G_padded.flatten()

        # normalised context scalars  (all in [0,1])
        n_norm  = (self.n   - 3) / (self.n_max - 3) if self.n_max > 3 else 0.0
        k_norm  = float(self.k - 1)                  # always 0 for k=1
        d_norm  = (self.d   - 2) / 3.0               # d in {2,3,5} → 0,.33,1
        cZ_norm = (self.c_Z - 0.5) / 1.5             # cZ in [0.5,2] → [0,1]
        pI_norm = (self.p_I - 0.75) / 0.25           # pI in [0.75,1] → [0,1]

        ctx = np.array([n_norm, k_norm, d_norm, cZ_norm, pI_norm],
                       dtype=np.float32)
        return np.concatenate([flat, ctx])

    def reset_to(self, n, k, d, c_Z, p_I=None):
        """
        Reset the environment to a specific (n, k, d, c_Z, p_I) config.
        Used by the encoding agent at inference time.
        """
        self.n   = n
        self.k   = k
        self.d   = d
        self.c_Z = c_Z
        if p_I is not None:
            self.p_I = p_I
        self.tab        = CliffordTableau(n, k)
        self.step_count = 0
        self.history    = []

        self.error_strings = build_error_set(n, k, d)
        self.errors        = pauli_strings_to_binary(self.error_strings, n)
        self.lambda_weight = paper_lambda_weights(
            self.error_strings, n, self.p_I, c_Z
        ).astype(np.float32)
        return self._obs()

    def _apply_action(self, gate):
        op = gate[0]
        if   op == "H":    self.tab.H(gate[1])
        elif op == "S":    self.tab.S(gate[1])
        elif op == "CNOT": self.tab.CX(gate[1], gate[2])

    def step(self, action_idx):
        mask = self.action_mask()
        if not mask[action_idx]:
            self.step_count += 1
            done = self.step_count >= self.max_steps
            return self._obs(), -1.0 + (self.fail_penalty if done else 0.0), done, {
                "kl_sum": float("nan"), "success": False, "illegal": True,
                "n": self.n, "d": self.d, "c_Z": self.c_Z,
            }

        gate = self.actions[action_idx]
        self._apply_action(gate)
        self.history.append(gate)
        self.step_count += 1

        kl = weighted_kl_sum(
            self.errors, self.lambda_weight, self.tab.G, self.n, exact=True)
        all_detected = not kl_undetected_mask(
            self.errors, self.tab.G, self.n, exact=True).any()

        reward = -float(kl)
        done = False
        if all_detected:
            reward += self.success_bonus
            done = True
        elif self.step_count >= self.max_steps:
            reward += self.fail_penalty
            done = True

        return self._obs(), reward, done, {
            "kl_sum": float(kl), "success": bool(all_detected),
            "illegal": False, "n": self.n, "d": self.d, "c_Z": self.c_Z,
        }

    def render_history(self):
        out = []
        for g in self.history:
            if g[0] == "CNOT":
                out.append(f"CNOT({g[1]}→{g[2]})")
            else:
                out.append(f"{g[0]}({g[1]})")
        return out

    def seed(self, s):
        self.rng = np.random.default_rng(s)

    @property
    def phase_name(self):
        codes = self._phase["configs"]
        return f"phase {self._phase_idx}: {[f'[[{n},{k},{d}]]' for n,k,d in codes]}"