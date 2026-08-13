# """
# Surface Code RL Environment — Distance 5  (REALISTIC, phenomenological noise)
# =============================================================================
# Continuous episodic decoding with phenomenological noise.

# Physics
# -------
# For a stabilizer code under Pauli noise, the syndrome is EXACTLY determined
# by the parity-check matrices:

#     syndrome_z = (H_z @ residual_X_errors) mod 2     (Z-stabs detect X errors)
#     syndrome_x = (H_x @ residual_Z_errors) mod 2     (X-stabs detect Z errors)

# This is exact for phenomenological noise (Pauli data errors + measurement
# flips, perfect gates) — equivalent to a code-space-initialized stabilizer
# simulator, but far faster and with no initialization subtleties.

# Noise model
# -----------
#   • Data qubits: each round, prob p_data → uniform X / Y / Z (depolarizing)
#   • Measurement: each ancilla read flipped with prob p_meas
#   • Gates: perfect

# Episode
# -------
#   reset():  prime k rounds (noise → measure) to fill observation window
#   step():   apply correction → inject noise → measure → detector events
#             → per-qubit reward → Pauli-frame logical-failure test
#   T = 500 rounds. Early termination only on logical failure.

# State  (15, 11, 11) = (3k, 2d+1, 2d+1)
#   channels  0-4 : detector events       (+1 X-stab change, -1 Z-stab change)
#   channels  5-9 : X-correction history  (+1 where X or Y applied)
#   channels 10-14: Z-correction history  (+1 where Z or Y applied)

# Action  MultiDiscrete([4]*25)
#   per qubit: 0=Identity, 1=X, 2=Z, 3=Y

# Reward (per-qubit vector, shape 25) — normalized (Option B)
#   delta_x[s] = prev_x_syn[s] - new_x_syn[s]
#   delta_z[s] = prev_z_syn[s] - new_z_syn[s]
#   r_q = H_x.T @ (delta_x / supp_x) + H_z.T @ (delta_z / supp_z)
#   → sum(r_q) = delta_weight exactly
#   terminal success: r_q += +R_success / n_qubits   (= +1.0 each for R=25)
#   terminal failure: r_q += -R_failure / n_qubits  and  -new_weight / n_qubits

# Logical failure (Pauli frame)
#   R_x = E_x XOR C_x ,  R_z = E_z XOR C_z
#   L_Z = first column of data grid {0,5,10,15,20}
#   L_X = first row    of data grid {0,1,2,3,4}
#   Failure_X = (R_x . L_Z) mod 2     Failure_Z = (R_z . L_X) mod 2
#   either == 1  → logical qubit dead → episode ends.
# """

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from collections import deque


# # ══════════════════════════════════════════════════════════════════════════════
# #  LAYOUT
# # ══════════════════════════════════════════════════════════════════════════════

# def data_qubit_positions(distance: int):
#     """Data qubits at odd (row, col). Index = row-major order (row*d + col)."""
#     return [(r, c) for r in range(1, 2*distance, 2)
#                    for c in range(1, 2*distance, 2)]


# def build_stabilizers(distance: int):
#     data_pos   = data_qubit_positions(distance)
#     data_set   = set(data_pos)
#     pos_to_idx = {p: i for i, p in enumerate(data_pos)}
#     size       = 2 * distance + 1
#     x_stab_pos, z_stab_pos       = [], []
#     x_stab_qubits, z_stab_qubits = [], []

#     def neighbours(r, c):
#         return [(r+dr, c+dc) for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]
#                 if (r+dr, c+dc) in data_set]

#     for r in range(size):
#         for c in range(size):
#             if r % 2 == 1 and c % 2 == 1:
#                 continue
#             nb = neighbours(r, c)
#             if not nb:
#                 continue
#             if (r + c) % 4 == 0:
#                 x_stab_pos.append((r, c))
#                 x_stab_qubits.append([pos_to_idx[p] for p in nb])
#             else:
#                 z_stab_pos.append((r, c))
#                 z_stab_qubits.append([pos_to_idx[p] for p in nb])

#     return x_stab_qubits, z_stab_qubits, x_stab_pos, z_stab_pos, data_pos


# def embed_detector_grid(x_events, z_events, x_stab_pos, z_stab_pos, grid_size):
#     grid = np.zeros((grid_size, grid_size), dtype=np.float32)
#     for bit, (r, c) in zip(x_events, x_stab_pos):
#         if bit: grid[r, c] = +1.0
#     for bit, (r, c) in zip(z_events, z_stab_pos):
#         if bit: grid[r, c] = -1.0
#     return grid


# def embed_correction_grid(mask, data_pos, grid_size):
#     grid = np.zeros((grid_size, grid_size), dtype=np.float32)
#     for qi, applied in enumerate(mask):
#         if applied:
#             r, c = data_pos[qi]
#             grid[r, c] = 1.0
#     return grid


# # ══════════════════════════════════════════════════════════════════════════════
# #  ENVIRONMENT
# # ══════════════════════════════════════════════════════════════════════════════

# class SurfaceCodeRealisticEnv(gym.Env):
#     """
#     Realistic d=5 surface code RL environment (phenomenological noise).

#     Parameters
#     ----------
#     distance, k, T, p_data, p_meas, R_success, R_failure
#     """

#     metadata = {"render_modes": ["human"]}

#     def __init__(
#         self,
#         distance:  int   = 5,
#         k:         int   = 5,
#         T:         int   = 500,
#         p_data:    float = 0.001,
#         p_meas:    float = 0.001,
#         R_success: float = 25.0,
#         R_failure: float = 25.0,
#     ):
#         super().__init__()
#         assert distance % 2 == 1

#         self.d, self.k, self.T = distance, k, T
#         self.p_data, self.p_meas = p_data, p_meas
#         self.R_success, self.R_failure = R_success, R_failure

#         (self.x_stab_qubits, self.z_stab_qubits,
#          self.x_stab_pos, self.z_stab_pos, self.data_pos) = build_stabilizers(distance)

#         self.n_qubits  = len(self.data_pos)
#         self.n_x_stabs = len(self.x_stab_pos)
#         self.n_z_stabs = len(self.z_stab_pos)
#         self.grid_size = 2 * distance + 1

#         # Parity-check matrices
#         #   H_z[s,q]=1 if qubit q in Z-stab s   (Z-stabs detect X errors)
#         #   H_x[s,q]=1 if qubit q in X-stab s   (X-stabs detect Z errors)
#         self.H_z = np.zeros((self.n_z_stabs, self.n_qubits), dtype=np.int8)
#         self.H_x = np.zeros((self.n_x_stabs, self.n_qubits), dtype=np.int8)
#         for s, qs in enumerate(self.z_stab_qubits):
#             for q in qs: self.H_z[s, q] = 1
#         for s, qs in enumerate(self.x_stab_qubits):
#             for q in qs: self.H_x[s, q] = 1

#         # Stabilizer support (for normalized reward)
#         self.supp_x = self.H_x.sum(axis=1).astype(np.float32)
#         self.supp_z = self.H_z.sum(axis=1).astype(np.float32)
#         self.supp_x[self.supp_x == 0] = 1.0
#         self.supp_z[self.supp_z == 0] = 1.0

#         # Logical operators
#         d = distance
#         self.L_Z_mask = np.zeros(self.n_qubits, dtype=np.int8)
#         self.L_X_mask = np.zeros(self.n_qubits, dtype=np.int8)
#         for r in range(d): self.L_Z_mask[r * d + 0] = 1     # first column
#         for c in range(d): self.L_X_mask[0 * d + c] = 1     # first row

#         # Spaces
#         self.action_space = spaces.MultiDiscrete([4] * self.n_qubits)
#         self.observation_space = spaces.Box(
#             low=-1.0, high=+1.0,
#             shape=(3 * self.k, self.grid_size, self.grid_size),
#             dtype=np.float32,
#         )

#         # Buffers
#         self._detector_history = deque(maxlen=self.k)
#         self._x_corr_history   = deque(maxlen=self.k)
#         self._z_corr_history   = deque(maxlen=self.k)

#         self.E_x = np.zeros(self.n_qubits, dtype=np.int8)
#         self.E_z = np.zeros(self.n_qubits, dtype=np.int8)
#         self.C_x = np.zeros(self.n_qubits, dtype=np.int8)
#         self.C_z = np.zeros(self.n_qubits, dtype=np.int8)

#         self._prev_x_syn  = np.zeros(self.n_x_stabs, dtype=np.int8)
#         self._prev_z_syn  = np.zeros(self.n_z_stabs, dtype=np.int8)
#         self._prev_weight = 0
#         self._step_count  = 0

#     # ──────────────────────────────────────────────────────────────────────────
#     #  SYNDROME (parity-check, exact for Pauli noise)
#     # ──────────────────────────────────────────────────────────────────────────

#     def _measure_syndrome(self):
#         """
#         Compute the TRUE syndrome from the current residual error (E XOR C),
#         then add per-measurement bit-flip noise.

#         Returns noisy (x_syn, z_syn).
#         """
#         R_x = self.E_x ^ self.C_x        # net X errors
#         R_z = self.E_z ^ self.C_z        # net Z errors

#         x_syn = (self.H_x @ R_z) % 2     # X-stabs detect Z errors
#         z_syn = (self.H_z @ R_x) % 2     # Z-stabs detect X errors
#         x_syn = x_syn.astype(np.int8)
#         z_syn = z_syn.astype(np.int8)

#         if self.p_meas > 0:
#             rng = self.np_random
#             xf  = (rng.random(self.n_x_stabs) < self.p_meas).astype(np.int8)
#             zf  = (rng.random(self.n_z_stabs) < self.p_meas).astype(np.int8)
#             x_syn = x_syn ^ xf
#             z_syn = z_syn ^ zf

#         return x_syn, z_syn

#     # ──────────────────────────────────────────────────────────────────────────
#     #  NOISE
#     # ──────────────────────────────────────────────────────────────────────────

#     def _inject_data_noise(self):
#         """Depolarizing on data qubits: prob p_data → uniform X / Y / Z."""
#         rng     = self.np_random
#         errors  = rng.random(self.n_qubits) < self.p_data
#         choices = rng.integers(0, 3, size=self.n_qubits)
#         for q in range(self.n_qubits):
#             if not errors[q]: continue
#             ch = choices[q]
#             if ch == 0:        # X
#                 self.E_x[q] ^= 1
#             elif ch == 1:      # Y
#                 self.E_x[q] ^= 1; self.E_z[q] ^= 1
#             else:              # Z
#                 self.E_z[q] ^= 1

#     # ──────────────────────────────────────────────────────────────────────────
#     #  LOGICAL FAILURE (Pauli frame)
#     # ──────────────────────────────────────────────────────────────────────────

#     def _logical_failure(self) -> bool:
#         R_x = self.E_x ^ self.C_x
#         R_z = self.E_z ^ self.C_z
#         fail_x = int((R_x * self.L_Z_mask).sum() % 2)
#         fail_z = int((R_z * self.L_X_mask).sum() % 2)
#         return (fail_x == 1) or (fail_z == 1)

#     # ──────────────────────────────────────────────────────────────────────────
#     #  RESET
#     # ──────────────────────────────────────────────────────────────────────────

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         self.E_x.fill(0); self.E_z.fill(0)
#         self.C_x.fill(0); self.C_z.fill(0)
#         self._detector_history.clear()
#         self._x_corr_history.clear()
#         self._z_corr_history.clear()

#         zero   = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
#         prev_x = np.zeros(self.n_x_stabs, dtype=np.int8)
#         prev_z = np.zeros(self.n_z_stabs, dtype=np.int8)

#         # Prime k rounds to fill the observation window
#         for _ in range(self.k):
#             self._inject_data_noise()
#             x_syn, z_syn = self._measure_syndrome()
#             x_ev = x_syn ^ prev_x
#             z_ev = z_syn ^ prev_z
#             prev_x, prev_z = x_syn, z_syn
#             self._detector_history.append(
#                 embed_detector_grid(x_ev, z_ev, self.x_stab_pos,
#                                     self.z_stab_pos, self.grid_size))
#             self._x_corr_history.append(zero.copy())
#             self._z_corr_history.append(zero.copy())

#         self._prev_x_syn  = prev_x
#         self._prev_z_syn  = prev_z
#         self._prev_weight = int(prev_x.sum() + prev_z.sum())
#         self._step_count  = 0
#         return self._get_obs(), self._info()

#     # ──────────────────────────────────────────────────────────────────────────
#     #  STEP
#     # ──────────────────────────────────────────────────────────────────────────

#     def step(self, action):
#         action = np.asarray(action, dtype=np.int64).flatten()
#         assert action.shape == (self.n_qubits,)
#         self._step_count += 1
#         # A correct decoder never acts on a trivial (all-zero) syndrome: there
#         current_syndrome_weight = int(self._prev_x_syn.sum() + self._prev_z_syn.sum())
#         if current_syndrome_weight == 0:
#             action = np.zeros(self.n_qubits, dtype=np.int64)

#         # 1. Apply corrections to Pauli frame
#         x_mask = np.zeros(self.n_qubits, dtype=np.int8)
#         z_mask = np.zeros(self.n_qubits, dtype=np.int8)
#         for q in range(self.n_qubits):
#             a = int(action[q])
#             if a == 1:                 # X
#                 self.C_x[q] ^= 1; x_mask[q] = 1
#             elif a == 2:               # Z
#                 self.C_z[q] ^= 1; z_mask[q] = 1
#             elif a == 3:               # Y = X·Z
#                 self.C_x[q] ^= 1; self.C_z[q] ^= 1
#                 x_mask[q] = 1; z_mask[q] = 1

#         # 2. Inject new data noise (continuous operation)
#         self._inject_data_noise()

#         # 3. Measure syndrome (parity-check + measurement noise)
#         x_syn, z_syn = self._measure_syndrome()

#         # 4. Detector events (XOR with previous round)
#         x_ev = x_syn ^ self._prev_x_syn
#         z_ev = z_syn ^ self._prev_z_syn

#         # 5. Per-qubit normalized reward
#         delta_x = self._prev_x_syn.astype(np.float32) - x_syn.astype(np.float32)
#         delta_z = self._prev_z_syn.astype(np.float32) - z_syn.astype(np.float32)
#         r_q  = self.H_x.T @ (delta_x / self.supp_x)
#         r_q += self.H_z.T @ (delta_z / self.supp_z)

#         # Action cost — LOCAL, only on the qubits that actually acted.
#         # A Y correction sets BOTH masks → pays the penalty twice (correct:
#         # a wrong Y introduces both an X and a Z error simultaneously).
#         penalty = 0.05
#         r_q[x_mask == 1] -= penalty
#         r_q[z_mask == 1] -= penalty

#         self._prev_x_syn = x_syn
#         self._prev_z_syn = z_syn
#         new_weight = int(x_syn.sum() + z_syn.sum())
#         self._prev_weight = new_weight

#         # 6. Sliding-window history
#         self._detector_history.append(
#             embed_detector_grid(x_ev, z_ev, self.x_stab_pos,
#                                 self.z_stab_pos, self.grid_size))
#         self._x_corr_history.append(
#             embed_correction_grid(x_mask, self.data_pos, self.grid_size))
#         self._z_corr_history.append(
#             embed_correction_grid(z_mask, self.data_pos, self.grid_size))

#         # 7. Logical failure test
#         logical_dead = self._logical_failure()

#         # 8. Terminal conditions
#         if logical_dead:
#             terminated, truncated = True, False
#             r_q += (-self.R_failure / self.n_qubits)
#             corrected, logical_err = False, True
#         elif self._step_count >= self.T:
#             terminated, truncated = False, True
#             r_q += (+self.R_success / self.n_qubits)
#             corrected, logical_err = True, False
#         else:
#             terminated, truncated = False, False
#             corrected, logical_err = False, False

#         return (
#             self._get_obs(),
#             r_q.astype(np.float32),
#             terminated,
#             truncated,
#             self._info(corrected=corrected, logical_err=logical_err,
#                        new_weight=new_weight),
#         )

#     # ──────────────────────────────────────────────────────────────────────────
#     #  OBSERVATION / INFO
#     # ──────────────────────────────────────────────────────────────────────────

#     def _get_obs(self) -> np.ndarray:
#         zero = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
#         det = list(self._detector_history)
#         xc  = list(self._x_corr_history)
#         zc  = list(self._z_corr_history)
#         while len(det) < self.k: det.insert(0, zero.copy())
#         while len(xc)  < self.k: xc.insert(0,  zero.copy())
#         while len(zc)  < self.k: zc.insert(0,  zero.copy())
#         return np.stack(det + xc + zc, axis=0).astype(np.float32)

#     def _info(self, corrected=False, logical_err=False, new_weight=None) -> dict:
#         return {
#             "syndrome_weight": new_weight if new_weight is not None else self._prev_weight,
#             "logical_error":   logical_err,
#             "corrected":       corrected,
#             "step":            self._step_count,
#             "E_x_weight":      int(self.E_x.sum()),
#             "E_z_weight":      int(self.E_z.sum()),
#             "C_x_weight":      int(self.C_x.sum()),
#             "C_z_weight":      int(self.C_z.sum()),
#         }

#     def render(self, mode="human"):
#         print(f"── Step {self._step_count}/{self.T} | weight={self._prev_weight} "
#               f"| E_x={int(self.E_x.sum())} E_z={int(self.E_z.sum())} "
#               f"| C_x={int(self.C_x.sum())} C_z={int(self.C_z.sum())}")

#     def __repr__(self):
#         return (f"SurfaceCodeRealisticEnv(d={self.d}, k={self.k}, T={self.T}, "
#                 f"p_data={self.p_data}, p_meas={self.p_meas}, "
#                 f"state=({3*self.k},{self.grid_size},{self.grid_size}), "
#                 f"actions=MultiDiscrete([4]*{self.n_qubits}))")
"""
Surface Code RL Environment — Distance 5  (REALISTIC, phenomenological noise)
=============================================================================
Continuous episodic decoding with phenomenological noise.

Physics
-------
For a stabilizer code under Pauli noise, the syndrome is EXACTLY determined
by the parity-check matrices:

    syndrome_z = (H_z @ residual_X_errors) mod 2     (Z-stabs detect X errors)
    syndrome_x = (H_x @ residual_Z_errors) mod 2     (X-stabs detect Z errors)

This is exact for phenomenological noise (Pauli data errors + measurement
flips, perfect gates) — equivalent to a code-space-initialized stabilizer
simulator, but far faster and with no initialization subtleties.

Noise model
-----------
  • Data qubits: each round, prob p_data → uniform X / Y / Z (depolarizing)
  • Measurement: each ancilla read flipped with prob p_meas
  • Gates: perfect

Episode
-------
  reset():  prime k rounds (noise → measure) to fill observation window
  step():   apply correction → inject noise → measure → detector events
            → per-qubit reward → Pauli-frame logical-failure test
  T = 500 rounds. Early termination only on logical failure.

State  (15, 11, 11) = (3k, 2d+1, 2d+1)
  channels  0-4 : detector events       (+1 X-stab change, -1 Z-stab change)
  channels  5-9 : X-correction history  (+1 where X or Y applied)
  channels 10-14: Z-correction history  (+1 where Z or Y applied)

Action  MultiDiscrete([4]*25)
  per qubit: 0=Identity, 1=X, 2=Z, 3=Y

Reward (per-qubit vector, shape 25) — normalized (Option B)
  delta_x[s] = prev_x_syn[s] - new_x_syn[s]
  delta_z[s] = prev_z_syn[s] - new_z_syn[s]
  r_q = H_x.T @ (delta_x / supp_x) + H_z.T @ (delta_z / supp_z)
  → sum(r_q) = delta_weight exactly
  terminal success: r_q += +R_success / n_qubits   (= +1.0 each for R=25)
  terminal failure: r_q += -R_failure / n_qubits  and  -new_weight / n_qubits

Logical failure (Pauli frame)
  R_x = E_x XOR C_x ,  R_z = E_z XOR C_z
  L_Z = first column of data grid {0,5,10,15,20}
  L_X = first row    of data grid {0,1,2,3,4}
  Failure_X = (R_x . L_Z) mod 2     Failure_Z = (R_z . L_X) mod 2
  either == 1  → logical qubit dead → episode ends.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def data_qubit_positions(distance: int):
    """Data qubits at odd (row, col). Index = row-major order (row*d + col)."""
    return [(r, c) for r in range(1, 2*distance, 2)
                   for c in range(1, 2*distance, 2)]


def build_stabilizers(distance: int):
    data_pos   = data_qubit_positions(distance)
    data_set   = set(data_pos)
    pos_to_idx = {p: i for i, p in enumerate(data_pos)}
    size       = 2 * distance + 1
    x_stab_pos, z_stab_pos       = [], []
    x_stab_qubits, z_stab_qubits = [], []

    def neighbours(r, c):
        return [(r+dr, c+dc) for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]
                if (r+dr, c+dc) in data_set]

    for r in range(size):
        for c in range(size):
            if r % 2 == 1 and c % 2 == 1:
                continue
            nb = neighbours(r, c)
            if not nb:
                continue
            if (r + c) % 4 == 0:
                x_stab_pos.append((r, c))
                x_stab_qubits.append([pos_to_idx[p] for p in nb])
            else:
                z_stab_pos.append((r, c))
                z_stab_qubits.append([pos_to_idx[p] for p in nb])

    return x_stab_qubits, z_stab_qubits, x_stab_pos, z_stab_pos, data_pos


def embed_detector_grid(x_events, z_events, x_stab_pos, z_stab_pos, grid_size):
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for bit, (r, c) in zip(x_events, x_stab_pos):
        if bit: grid[r, c] = +1.0
    for bit, (r, c) in zip(z_events, z_stab_pos):
        if bit: grid[r, c] = -1.0
    return grid


def embed_correction_grid(mask, data_pos, grid_size):
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for qi, applied in enumerate(mask):
        if applied:
            r, c = data_pos[qi]
            grid[r, c] = 1.0
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class SurfaceCodeRealisticEnv(gym.Env):
    """
    Realistic d=5 surface code RL environment (phenomenological noise).

    Parameters
    ----------
    distance, k, T, p_data, p_meas, R_success, R_failure
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        distance:  int   = 5,
        k:         int   = 5,
        T:         int   = 500,
        p_data:    float = 0.001,
        p_meas:    float = 0.001,
        R_success: float = 25.0,
        R_failure: float = 25.0,
    ):
        super().__init__()
        assert distance % 2 == 1

        self.d, self.k, self.T = distance, k, T
        self.p_data, self.p_meas = p_data, p_meas
        self.R_success, self.R_failure = R_success, R_failure

        (self.x_stab_qubits, self.z_stab_qubits,
         self.x_stab_pos, self.z_stab_pos, self.data_pos) = build_stabilizers(distance)

        self.n_qubits  = len(self.data_pos)
        self.n_x_stabs = len(self.x_stab_pos)
        self.n_z_stabs = len(self.z_stab_pos)
        self.grid_size = 2 * distance + 1

        # Parity-check matrices
        #   H_z[s,q]=1 if qubit q in Z-stab s   (Z-stabs detect X errors)
        #   H_x[s,q]=1 if qubit q in X-stab s   (X-stabs detect Z errors)
        self.H_z = np.zeros((self.n_z_stabs, self.n_qubits), dtype=np.int8)
        self.H_x = np.zeros((self.n_x_stabs, self.n_qubits), dtype=np.int8)
        for s, qs in enumerate(self.z_stab_qubits):
            for q in qs: self.H_z[s, q] = 1
        for s, qs in enumerate(self.x_stab_qubits):
            for q in qs: self.H_x[s, q] = 1

        # Stabilizer support (for normalized reward)
        self.supp_x = self.H_x.sum(axis=1).astype(np.float32)
        self.supp_z = self.H_z.sum(axis=1).astype(np.float32)
        self.supp_x[self.supp_x == 0] = 1.0
        self.supp_z[self.supp_z == 0] = 1.0

        # Logical operators
        d = distance
        self.L_Z_mask = np.zeros(self.n_qubits, dtype=np.int8)
        self.L_X_mask = np.zeros(self.n_qubits, dtype=np.int8)
        for r in range(d): self.L_Z_mask[r * d + 0] = 1     # first column
        for c in range(d): self.L_X_mask[0 * d + c] = 1     # first row

        # Spaces
        self.action_space = spaces.MultiDiscrete([4] * self.n_qubits)
        self.observation_space = spaces.Box(
            low=-1.0, high=+1.0,
            shape=(3 * self.k, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        # Buffers
        self._detector_history = deque(maxlen=self.k)
        self._x_corr_history   = deque(maxlen=self.k)
        self._z_corr_history   = deque(maxlen=self.k)

        self.E_x = np.zeros(self.n_qubits, dtype=np.int8)
        self.E_z = np.zeros(self.n_qubits, dtype=np.int8)
        self.C_x = np.zeros(self.n_qubits, dtype=np.int8)
        self.C_z = np.zeros(self.n_qubits, dtype=np.int8)

        self._prev_x_syn  = np.zeros(self.n_x_stabs, dtype=np.int8)
        self._prev_z_syn  = np.zeros(self.n_z_stabs, dtype=np.int8)
        self._prev_weight = 0
        self._step_count  = 0

    # ──────────────────────────────────────────────────────────────────────────
    #  SYNDROME (parity-check, exact for Pauli noise)
    # ──────────────────────────────────────────────────────────────────────────

    def _measure_syndrome(self):
        """
        Compute the TRUE syndrome from the current residual error (E XOR C),
        then add per-measurement bit-flip noise.

        Returns noisy (x_syn, z_syn).
        """
        R_x = self.E_x ^ self.C_x        # net X errors
        R_z = self.E_z ^ self.C_z        # net Z errors

        x_syn = (self.H_x @ R_z) % 2     # X-stabs detect Z errors
        z_syn = (self.H_z @ R_x) % 2     # Z-stabs detect X errors
        x_syn = x_syn.astype(np.int8)
        z_syn = z_syn.astype(np.int8)

        if self.p_meas > 0:
            rng = self.np_random
            xf  = (rng.random(self.n_x_stabs) < self.p_meas).astype(np.int8)
            zf  = (rng.random(self.n_z_stabs) < self.p_meas).astype(np.int8)
            x_syn = x_syn ^ xf
            z_syn = z_syn ^ zf

        return x_syn, z_syn

    # ──────────────────────────────────────────────────────────────────────────
    #  NOISE
    # ──────────────────────────────────────────────────────────────────────────

    def _inject_data_noise(self):
        """Depolarizing on data qubits: prob p_data → uniform X / Y / Z."""
        rng     = self.np_random
        errors  = rng.random(self.n_qubits) < self.p_data
        choices = rng.integers(0, 3, size=self.n_qubits)
        for q in range(self.n_qubits):
            if not errors[q]: continue
            ch = choices[q]
            if ch == 0:        # X
                self.E_x[q] ^= 1
            elif ch == 1:      # Y
                self.E_x[q] ^= 1; self.E_z[q] ^= 1
            else:              # Z
                self.E_z[q] ^= 1

    # ──────────────────────────────────────────────────────────────────────────
    #  LOGICAL FAILURE (Pauli frame)
    # ──────────────────────────────────────────────────────────────────────────

    def _logical_failure(self) -> bool:
        R_x = self.E_x ^ self.C_x
        R_z = self.E_z ^ self.C_z
        fail_x = int((R_x * self.L_Z_mask).sum() % 2)
        fail_z = int((R_z * self.L_X_mask).sum() % 2)
        return (fail_x == 1) or (fail_z == 1)

    # ──────────────────────────────────────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.E_x.fill(0); self.E_z.fill(0)
        self.C_x.fill(0); self.C_z.fill(0)
        self._detector_history.clear()
        self._x_corr_history.clear()
        self._z_corr_history.clear()

        zero   = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        prev_x = np.zeros(self.n_x_stabs, dtype=np.int8)
        prev_z = np.zeros(self.n_z_stabs, dtype=np.int8)

        # Prime k rounds to fill the observation window
        for _ in range(self.k):
            self._inject_data_noise()
            x_syn, z_syn = self._measure_syndrome()
            x_ev = x_syn ^ prev_x
            z_ev = z_syn ^ prev_z
            prev_x, prev_z = x_syn, z_syn
            self._detector_history.append(
                embed_detector_grid(x_ev, z_ev, self.x_stab_pos,
                                    self.z_stab_pos, self.grid_size))
            self._x_corr_history.append(zero.copy())
            self._z_corr_history.append(zero.copy())

        self._prev_x_syn  = prev_x
        self._prev_z_syn  = prev_z
        self._prev_weight = int(prev_x.sum() + prev_z.sum())
        self._step_count  = 0
        return self._get_obs(), self._info()

    # ──────────────────────────────────────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────────────────────────────────────

    def _actionable_components(self):
        """
        Per-qubit boolean masks of which Pauli corrections are sensible,
        given the currently-fired stabilizers.

        Returns
        -------
        allow_x : bool array (n_qubits,)  True if an X (or the X-part of Y)
                  is allowed — i.e. the qubit touches a fired Z-stabilizer
                  (Z-stabs detect X errors).
        allow_z : bool array (n_qubits,)  True if a Z (or the Z-part of Y)
                  is allowed — i.e. the qubit touches a fired X-stabilizer
                  (X-stabs detect Z errors).
        """
        allow_x = np.zeros(self.n_qubits, dtype=bool)
        allow_z = np.zeros(self.n_qubits, dtype=bool)
        # Fired Z-stabilizers → their data qubits may take X / Y
        for s in range(self.n_z_stabs):
            if self._prev_z_syn[s]:
                for q in self.z_stab_qubits[s]:
                    allow_x[q] = True
        # Fired X-stabilizers → their data qubits may take Z / Y
        for s in range(self.n_x_stabs):
            if self._prev_x_syn[s]:
                for q in self.x_stab_qubits[s]:
                    allow_z[q] = True
        return allow_x, allow_z

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).flatten()
        assert action.shape == (self.n_qubits,)
        self._step_count += 1

        # ── Clean-syndrome override ───────────────────────────────────────────
        # A correct decoder never acts on a trivial (all-zero) syndrome: there
        # is no error signal to respond to, and any correction on a clean code
        # only introduces errors. When the current measured syndrome is empty,
        # force the identity action. This restricts the agent's learning to the
        # cases that actually matter (a non-trivial syndrome present) and stops
        # it from corrupting a clean code state. Standard decoder behaviour —
        # MWPM likewise returns no correction for a trivial syndrome.
        current_syndrome_weight = int(self._prev_x_syn.sum() + self._prev_z_syn.sum())
        if current_syndrome_weight == 0:
            action = np.zeros(self.n_qubits, dtype=np.int64)
        else:
            # ── Neighbour-restriction mask ────────────────────────────────────
            # A correction only makes sense on a data qubit adjacent to a
            # currently-fired stabilizer — that is where the error signal is.
            # Correcting a qubit that touches no violated stabilizer can only
            # inject errors, never remove existing ones (it is the dominant
            # death cause in the continuing task). We therefore restrict each
            # qubit's allowed Pauli by what kind of stabilizer fired near it:
            #   • adjacent to a fired Z-stabilizer → X or Y allowed (fix X error)
            #   • adjacent to a fired X-stabilizer → Z or Y allowed (fix Z error)
            # Disallowed components are stripped from the action; a qubit with
            # no fired neighbour is forced to Identity. This mirrors MWPM, which
            # only ever matches through qubits between fired detectors.
            allow_x, allow_z = self._actionable_components()
            for q in range(self.n_qubits):
                a = int(action[q])
                if a == 1 and not allow_x[q]:           # X not allowed here
                    action[q] = 0
                elif a == 2 and not allow_z[q]:         # Z not allowed here
                    action[q] = 0
                elif a == 3:                            # Y = X·Z: keep only allowed parts
                    if allow_x[q] and allow_z[q]:
                        action[q] = 3                   # both → Y
                    elif allow_x[q]:
                        action[q] = 1                   # only X part survives
                    elif allow_z[q]:
                        action[q] = 2                   # only Z part survives
                    else:
                        action[q] = 0                   # neither → Identity

        # 1. Apply corrections to Pauli frame
        x_mask = np.zeros(self.n_qubits, dtype=np.int8)
        z_mask = np.zeros(self.n_qubits, dtype=np.int8)
        for q in range(self.n_qubits):
            a = int(action[q])
            if a == 1:                 # X
                self.C_x[q] ^= 1; x_mask[q] = 1
            elif a == 2:               # Z
                self.C_z[q] ^= 1; z_mask[q] = 1
            elif a == 3:               # Y = X·Z
                self.C_x[q] ^= 1; self.C_z[q] ^= 1
                x_mask[q] = 1; z_mask[q] = 1

        # 2. Inject new data noise (continuous operation)
        self._inject_data_noise()

        # 3. Measure syndrome (parity-check + measurement noise)
        x_syn, z_syn = self._measure_syndrome()

        # 4. Detector events (XOR with previous round)
        x_ev = x_syn ^ self._prev_x_syn
        z_ev = z_syn ^ self._prev_z_syn

        # 5. Per-qubit normalized reward
        delta_x = self._prev_x_syn.astype(np.float32) - x_syn.astype(np.float32)
        delta_z = self._prev_z_syn.astype(np.float32) - z_syn.astype(np.float32)
        r_q  = self.H_x.T @ (delta_x / self.supp_x)
        r_q += self.H_z.T @ (delta_z / self.supp_z)

        # Action cost — LOCAL, only on the qubits that actually acted.
        # A Y correction sets BOTH masks → pays the penalty twice (correct:
        # a wrong Y introduces both an X and a Z error simultaneously).
        penalty = 0.05
        r_q[x_mask == 1] -= penalty
        r_q[z_mask == 1] -= penalty

        self._prev_x_syn = x_syn
        self._prev_z_syn = z_syn
        new_weight = int(x_syn.sum() + z_syn.sum())
        self._prev_weight = new_weight

        # 6. Sliding-window history
        self._detector_history.append(
            embed_detector_grid(x_ev, z_ev, self.x_stab_pos,
                                self.z_stab_pos, self.grid_size))
        self._x_corr_history.append(
            embed_correction_grid(x_mask, self.data_pos, self.grid_size))
        self._z_corr_history.append(
            embed_correction_grid(z_mask, self.data_pos, self.grid_size))

        # 7. Logical failure test
        logical_dead = self._logical_failure()

        # 8. Terminal conditions
        if logical_dead:
            terminated, truncated = True, False
            r_q += (-self.R_failure / self.n_qubits)
            corrected, logical_err = False, True
        elif self._step_count >= self.T:
            terminated, truncated = False, True
            r_q += (+self.R_success / self.n_qubits)
            corrected, logical_err = True, False
        else:
            terminated, truncated = False, False
            corrected, logical_err = False, False

        return (
            self._get_obs(),
            r_q.astype(np.float32),
            terminated,
            truncated,
            self._info(corrected=corrected, logical_err=logical_err,
                       new_weight=new_weight),
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  OBSERVATION / INFO
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        zero = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        det = list(self._detector_history)
        xc  = list(self._x_corr_history)
        zc  = list(self._z_corr_history)
        while len(det) < self.k: det.insert(0, zero.copy())
        while len(xc)  < self.k: xc.insert(0,  zero.copy())
        while len(zc)  < self.k: zc.insert(0,  zero.copy())
        return np.stack(det + xc + zc, axis=0).astype(np.float32)

    def _info(self, corrected=False, logical_err=False, new_weight=None) -> dict:
        return {
            "syndrome_weight": new_weight if new_weight is not None else self._prev_weight,
            "logical_error":   logical_err,
            "corrected":       corrected,
            "step":            self._step_count,
            "E_x_weight":      int(self.E_x.sum()),
            "E_z_weight":      int(self.E_z.sum()),
            "C_x_weight":      int(self.C_x.sum()),
            "C_z_weight":      int(self.C_z.sum()),
        }

    def render(self, mode="human"):
        print(f"── Step {self._step_count}/{self.T} | weight={self._prev_weight} "
              f"| E_x={int(self.E_x.sum())} E_z={int(self.E_z.sum())} "
              f"| C_x={int(self.C_x.sum())} C_z={int(self.C_z.sum())}")

    def __repr__(self):
        return (f"SurfaceCodeRealisticEnv(d={self.d}, k={self.k}, T={self.T}, "
                f"p_data={self.p_data}, p_meas={self.p_meas}, "
                f"state=({3*self.k},{self.grid_size},{self.grid_size}), "
                f"actions=MultiDiscrete([4]*{self.n_qubits}))")