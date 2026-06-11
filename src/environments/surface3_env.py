# """
# Surface Code RL Environment — Distance 3
# ==========================================
# Exact implementation of the spec:

#   state.shape  = (2k, 2*d+1, 2*d+1)  first k channels = syndrome history
#                                       last  k channels = action  history
#   action space = 2*d*d + 1            (X_i, Z_i, Identity)
#   reward       = (prev_weight - new_weight) - alpha
#                + R_success on success
#                - R_fail    on logical error

# Hidden state : X/Z error arrays on d² data qubits (agent NEVER sees this)
# Observed     : syndrome history + action history (two separate channel blocks)
# """

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from collections import deque


# # ═══════════════════════════════════════════════════════════════════════════════
# #  SECTION 1 — Rotated Surface Code Layout (d=3)
# # ═══════════════════════════════════════════════════════════════════════════════
# #
# #  Rotated surface code on a (2d+1)×(2d+1) grid:
# #
# #      col  0  1  2  3  4  5  6
# #  row 0  [ .  .  D  .  D  .  . ]
# #      1  [ .  Z  .  X  .  Z  . ]
# #      2  [ D  .  D  .  D  .  D ]   D = data qubit   (even row, even col)
# #      3  [ .  X  .  Z  .  X  . ]   X = X-stabilizer (odd row, odd col, X-type)
# #      4  [ D  .  D  .  D  .  D ]   Z = Z-stabilizer (odd row, odd col, Z-type)
# #      5  [ .  Z  .  X  .  Z  . ]
# #      6  [ .  .  D  .  D  .  . ]
# #
# #  For d=3:  grid = 7×7,  d²=9 data qubits,  4 X-stabs,  4 Z-stabs
# #
# #  Data qubit positions  (row, col) — read left-to-right, top-to-bottom:
# #    (0,2),(0,4), (2,0),(2,2),(2,4),(2,6), (4,0),(4,2),(4,4),(4,6), (6,2),(6,4)
# #    → but we only take the first d²=9
# #
# #  Stabilizer positions and their neighbour data qubits are hand-coded below
# #  using the standard rotated-surface-code connectivity.

# def _build_layout(d: int):
#     """
#     Build the rotated surface code layout for distance d.

#     Coordinate convention (matching Stim):
#       - Data qubits at (col, row) where col and row are both ODD, in [1..2d-1]
#       - Stabilizers at (col, row) where at least one coordinate is EVEN
#         placed on the (2d+1)×(2d+1) integer grid

#     Grid mapping: grid position = (row, col)  (row = y, col = x)

#     For d=3 on a 7×7 grid:
#       Data qubits (row, col): (1,1),(1,3),(1,5),(3,1),(3,3),(3,5),(5,1),(5,3),(5,5)
#       X-stabilizers (measure X on data, detect Z errors):
#           boundary edges + interior — placed at even-row positions
#       Z-stabilizers (measure Z on data, detect X errors):
#           placed at even-col positions

#     Connectivity: each stabilizer checks its ≤4 data-qubit neighbours
#     (the four odd-odd positions diagonally adjacent to it on the grid).

#     Returns
#     -------
#     data_pos   : list of (row, col) for d² data qubits
#     x_stab_pos : list of (row, col) for X-stabilizers
#     z_stab_pos : list of (row, col) for Z-stabilizers
#     x_checks   : x_checks[s] = list of qubit indices measured by X-stab s
#     z_checks   : z_checks[s] = list of qubit indices measured by Z-stab s
#     """
#     size = 2 * d + 1  # grid side length

#     # ── Data qubits: both row and col odd, in [1 .. 2d-1] ────────────────────
#     data_pos = [(r, c)
#                 for r in range(1, 2 * d, 2)
#                 for c in range(1, 2 * d, 2)]
#     # exactly d² qubits
#     assert len(data_pos) == d * d
#     pos_to_idx = {p: i for i, p in enumerate(data_pos)}
#     data_set = set(data_pos)

#     # ── Stabilizer positions and type ─────────────────────────────────────────
#     # In the rotated surface code the stabilizers sit on a checkerboard of the
#     # "even" sublattice of the (2d+1)×(2d+1) grid.
#     # Each stabilizer is at a position with at least one even coordinate.
#     # We classify them by the standard checkerboard rule:
#     #   (row + col) % 4 == 0  → X-stabilizer
#     #   (row + col) % 4 == 2  → Z-stabilizer
#     # We include only positions that are adjacent to ≥1 data qubit.

#     def data_neighbours(r, c):
#         """Data qubits diagonally adjacent to stabilizer at (r,c)."""
#         nb = []
#         for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
#             p = (r + dr, c + dc)
#             if p in data_set:
#                 nb.append(p)
#         return nb

#     x_stab_pos, x_checks_pos = [], []
#     z_stab_pos, z_checks_pos = [], []

#     for r in range(size):
#         for c in range(size):
#             # Skip data qubit positions (both odd)
#             if r % 2 == 1 and c % 2 == 1:
#                 continue
#             nb = data_neighbours(r, c)
#             if not nb:
#                 continue
#             if (r + c) % 4 == 0:
#                 x_stab_pos.append((r, c))
#                 x_checks_pos.append(nb)
#             else:  # (r+c) % 4 == 2  (or other even combos on the boundary)
#                 z_stab_pos.append((r, c))
#                 z_checks_pos.append(nb)

#     # Convert neighbour positions to qubit indices
#     x_checks = [[pos_to_idx[p] for p in nb] for nb in x_checks_pos]
#     z_checks = [[pos_to_idx[p] for p in nb] for nb in z_checks_pos]

#     return data_pos, x_stab_pos, z_stab_pos, x_checks, z_checks


# # ═══════════════════════════════════════════════════════════════════════════════
# #  SECTION 2 — Hidden Error State
# # ═══════════════════════════════════════════════════════════════════════════════

# class ErrorState:
#     """
#     Maintains the HIDDEN X/Z Pauli error configuration on d² data qubits.
#     The agent never observes this directly — only syndromes are exposed.

#       x_err[i] = 1  means qubit i has an X error (flips Z-stabilizers)
#       z_err[i] = 1  means qubit i has a Z error  (flips X-stabilizers)
#     """

#     def __init__(self, n_qubits: int):
#         self.n = n_qubits
#         self.x_err = np.zeros(n_qubits, dtype=np.int8)
#         self.z_err = np.zeros(n_qubits, dtype=np.int8)

#     def reset(self):
#         self.x_err[:] = 0
#         self.z_err[:] = 0

#     def inject_depolarizing(self, p: float):
#         """
#         Independent single-qubit depolarizing noise:
#           p/3 chance of X,  p/3 chance of Z,  p/3 chance of Y (= X+Z)
#         """
#         for i in range(self.n):
#             r = np.random.random()
#             if r < p / 3:
#                 self.x_err[i] ^= 1
#             elif r < 2 * p / 3:
#                 self.z_err[i] ^= 1
#             elif r < p:
#                 self.x_err[i] ^= 1
#                 self.z_err[i] ^= 1

#     def apply_x(self, qubit: int):
#         """Apply X correction — XOR into x_err (X corrects X errors)."""
#         self.x_err[qubit] ^= 1

#     def apply_z(self, qubit: int):
#         """Apply Z correction — XOR into z_err."""
#         self.z_err[qubit] ^= 1

#     def measure_x_syndrome(self, x_checks) -> np.ndarray:
#         """
#         X-stabilizer syndrome: each X-stab checks its data qubits for Z errors.
#         syndrome[s] = XOR of z_err on qubits in x_checks[s]
#         """
#         syn = np.array(
#             [int(sum(self.z_err[idx] for idx in check) % 2) if check else 0
#              for check in x_checks],
#             dtype=np.int8,
#         )
#         return syn

#     def measure_z_syndrome(self, z_checks) -> np.ndarray:
#         """
#         Z-stabilizer syndrome: each Z-stab checks its data qubits for X errors.
#         syndrome[s] = XOR of x_err on qubits in z_checks[s]
#         """
#         syn = np.array(
#             [int(sum(self.x_err[idx] for idx in check) % 2) if check else 0
#              for check in z_checks],
#             dtype=np.int8,
#         )
#         return syn

#     def measure_full_syndrome(self, x_checks, z_checks) -> np.ndarray:
#         """Concatenated [X-stab | Z-stab] syndrome vector."""
#         return np.concatenate([
#             self.measure_x_syndrome(x_checks),
#             self.measure_z_syndrome(z_checks),
#         ])

#     def add_measurement_noise(self, syndrome: np.ndarray, p_meas: float) -> np.ndarray:
#         """Bit-flip each syndrome bit independently with probability p_meas."""
#         flips = (np.random.random(len(syndrome)) < p_meas).astype(np.int8)
#         return (syndrome ^ flips).astype(np.int8)

#     def logical_x_error(self, d: int) -> bool:
#         """
#         Logical X error: X-error chain crosses the code from top to bottom.
#         Detected by the parity of x_err along any vertical line of qubits.
#         """
#         err = self.x_err.reshape(d, d)
#         return bool(np.any(np.sum(err, axis=0) % 2 == 1))

#     def logical_z_error(self, d: int) -> bool:
#         """
#         Logical Z error: Z-error chain crosses code left to right.
#         """
#         err = self.z_err.reshape(d, d)
#         return bool(np.any(np.sum(err, axis=1) % 2 == 1))

#     def has_logical_error(self, d: int) -> bool:
#         return self.logical_x_error(d) or self.logical_z_error(d)

#     def syndrome_weight(self, x_checks, z_checks) -> int:
#         """Number of triggered stabilizers."""
#         syn = self.measure_full_syndrome(x_checks, z_checks)
#         return int(np.sum(syn))


# # ═══════════════════════════════════════════════════════════════════════════════
# #  SECTION 3 — Syndrome Grid Embedding
# # ═══════════════════════════════════════════════════════════════════════════════

# def embed_syndrome_to_grid(
#     x_syndrome: np.ndarray,
#     z_syndrome: np.ndarray,
#     x_stab_pos: list,
#     z_stab_pos: list,
#     grid_size: int,
# ) -> np.ndarray:
#     """
#     Embed syndrome bits into a (grid_size, grid_size) float32 grid.

#       X-stabilizer triggered → grid[r, c] = +1.0
#       Z-stabilizer triggered → grid[r, c] = -1.0   (distinguishable)
#       Data-qubit positions   → 0.0
#       Empty positions        → 0.0

#     Using ±1 encoding lets the CNN distinguish X- from Z-type syndrome bits.
#     """
#     grid = np.zeros((grid_size, grid_size), dtype=np.float32)
#     for bit, (r, c) in zip(x_syndrome, x_stab_pos):
#         if bit:
#             grid[r, c] = +1.0
#     for bit, (r, c) in zip(z_syndrome, z_stab_pos):
#         if bit:
#             grid[r, c] = -1.0
#     return grid


# # ═══════════════════════════════════════════════════════════════════════════════
# #  SECTION 4 — Gymnasium Environment
# # ═══════════════════════════════════════════════════════════════════════════════

# class SurfaceCodeEnv(gym.Env):
#     """
#     Gymnasium environment for RL-based surface code decoding.

#     ┌────────────────────────────────────────────────────────┐
#     │  state.shape  = (k, 2d+1, 2d+1)   k = d rounds        │
#     │  n_actions    = 2*d*d + 1                              │
#     │  actions      = X_0..X_{d²-1} | Z_0..Z_{d²-1} | I    │
#     │  reward       = Δweight - alpha [± terminal bonuses]   │
#     └────────────────────────────────────────────────────────┘

#     Parameters
#     ----------
#     distance        : code distance (default 3, must be odd)
#     noise           : physical depolarizing error rate
#     meas_noise      : measurement bit-flip probability (default = noise/10)
#     syndrome_rounds : k — number of syndrome rounds to stack (default = d)
#     max_steps       : episode step limit
#     alpha           : per-step penalty (default 0.01)
#     R_success       : terminal reward on success (default 10)
#     R_fail          : terminal penalty on logical failure (default 10)
#     """

#     metadata = {"render_modes": ["human", "ansi"]}

#     def __init__(
#         self,
#         distance:        int   = 3,
#         noise:           float = 0.01,
#         meas_noise:      float = None,
#         syndrome_rounds: int   = None,
#         max_steps:       int   = 50,
#         alpha:           float = 0.01,
#         R_success:       float = 10.0,
#         R_fail:          float = 10.0,
#     ):
#         super().__init__()
#         assert distance % 2 == 1, "distance must be odd"

#         self.d        = distance
#         self.noise    = noise
#         self.p_meas   = meas_noise if meas_noise is not None else noise / 10.0
#         self.k        = syndrome_rounds if syndrome_rounds is not None else distance
#         self.max_steps = max_steps
#         self.alpha    = alpha
#         self.R_success = R_success
#         self.R_fail   = R_fail

#         # Layout
#         (self.data_pos,
#          self.x_stab_pos,
#          self.z_stab_pos,
#          self.x_checks,
#          self.z_checks) = _build_layout(distance)

#         self.n_qubits  = len(self.data_pos)     # = d²
#         self.n_x_stabs = len(self.x_stab_pos)
#         self.n_z_stabs = len(self.z_stab_pos)
#         self.grid_size = 2 * distance + 1        # = 2d+1

#         # Action space: X_i (0..d²-1) | Z_i (d²..2d²-1) | I (2d²)
#         self.n_actions = 2 * self.n_qubits + 1
#         self.action_space = spaces.Discrete(self.n_actions)
#         self.ACTION_IDENTITY = self.n_actions - 1

#         # Observation space: (2k, 2d+1, 2d+1)
#         #   channels  0..k-1  → syndrome history (k rounds)
#         #   channels  k..2k-1 → action  history (k rounds)
#         self.observation_space = spaces.Box(
#             low  = -1.0,
#             high = +1.0,
#             shape = (2 * self.k, self.grid_size, self.grid_size),
#             dtype = np.float32,
#         )

#         # Internal objects
#         self.error_state = ErrorState(self.n_qubits)
#         self._syndrome_history = deque(maxlen=self.k)
#         self._action_history  = deque(maxlen=self.k)   # ← NEW
#         self._step_count   = 0
#         self._prev_weight  = 0

#     # ── Reset ─────────────────────────────────────────────────────────────────

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         # 1. Clean code state
#         self.error_state.reset()

#         # 2. Inject random physical errors
#         self.error_state.inject_depolarizing(self.noise)

#         # 3. Initialise syndrome history with k rounds of measurements
#         self._syndrome_history.clear()
#         self._action_history.clear()                          # ← NEW
#         zero_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
#         for _ in range(self.k):
#             grid = self._measure_and_embed()
#             self._syndrome_history.append(grid)
#             self._action_history.append(zero_grid.copy())    # ← NEW: no actions yet

#         self._step_count  = 0
#         self._prev_weight = self.error_state.syndrome_weight(self.x_checks, self.z_checks)

#         return self._get_obs(), {}

#     # ── Step ──────────────────────────────────────────────────────────────────

#     def step(self, action: int):
#         assert self.action_space.contains(action), f"Invalid action {action}"
#         self._step_count += 1

#         # 1. Apply action to hidden error state + build action grid
#         action_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
#         if action < self.n_qubits:                      # X correction
#             self.error_state.apply_x(action)
#             r, c = self.data_pos[action]
#             action_grid[r, c] = +1.0                    # +1 = X applied
#         elif action < 2 * self.n_qubits:                # Z correction
#             qubit = action - self.n_qubits
#             self.error_state.apply_z(qubit)
#             r, c = self.data_pos[qubit]
#             action_grid[r, c] = -1.0                    # -1 = Z applied
#         # else: Identity — action_grid stays all zeros
#         self._action_history.append(action_grid)        # ← NEW

#         # 2. Measure new syndrome & embed into grid
#         grid = self._measure_and_embed()
#         self._syndrome_history.append(grid)

#         # 3. Compute syndrome weight
#         new_weight = self.error_state.syndrome_weight(self.x_checks, self.z_checks)

#         # 4. Check termination conditions
#         logical_err = self.error_state.has_logical_error(self.d)
#         corrected   = (new_weight == 0) and (not logical_err)
#         timeout     = (self._step_count >= self.max_steps)

#         terminated = corrected or logical_err
#         truncated  = (not terminated) and timeout

#         # 5. Reward   r = (prev_weight - new_weight) - alpha  ± terminal
#         reward = float(self._prev_weight - new_weight) - self.alpha
#         if corrected:
#             reward += self.R_success
#         elif logical_err:
#             reward -= self.R_fail

#         self._prev_weight = new_weight

#         info = {
#             "syndrome_weight": new_weight,
#             "logical_error":   logical_err,
#             "corrected":       corrected,
#             "step":            self._step_count,
#         }
#         return self._get_obs(), reward, terminated, truncated, info

#     # ── Observation ───────────────────────────────────────────────────────────

#     def _measure_and_embed(self) -> np.ndarray:
#         """Measure stabilizers (with noise) and return embedded grid."""
#         x_syn = self.error_state.measure_x_syndrome(self.x_checks)
#         z_syn = self.error_state.measure_z_syndrome(self.z_checks)

#         if self.p_meas > 0:
#             full = np.concatenate([x_syn, z_syn])
#             full = self.error_state.add_measurement_noise(full, self.p_meas)
#             x_syn = full[:self.n_x_stabs]
#             z_syn = full[self.n_x_stabs:]

#         return embed_syndrome_to_grid(
#             x_syn, z_syn,
#             self.x_stab_pos, self.z_stab_pos,
#             self.grid_size,
#         )

#     def _get_obs(self) -> np.ndarray:
#         """
#         Stack into observation tensor of shape (2k, 2d+1, 2d+1):
#           channels  0 .. k-1  →  syndrome history  (S_{t-k+1} .. S_t)
#           channels  k .. 2k-1 →  action  history   (A_{t-k+1} .. A_t)

#         Values:
#           syndrome channels: +1 X-stab fired, -1 Z-stab fired, 0 nothing
#           action  channels:  +1 X applied,    -1 Z applied,    0 nothing / Identity
#         """
#         zero = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

#         syn_frames = list(self._syndrome_history)
#         while len(syn_frames) < self.k:
#             syn_frames.insert(0, zero.copy())

#         act_frames = list(self._action_history)
#         while len(act_frames) < self.k:
#             act_frames.insert(0, zero.copy())

#         # shape: (2k, H, W)
#         return np.stack(syn_frames + act_frames, axis=0).astype(np.float32)

#     # ── Render ────────────────────────────────────────────────────────────────

#     def render(self, mode="human"):
#         grid = self._syndrome_history[-1] if self._syndrome_history else \
#                np.zeros((self.grid_size, self.grid_size))
#         lines = [f"\n── Step {self._step_count} | weight={self._prev_weight} ──"]
#         symbols = {0.0: "·", 1.0: "X", -1.0: "Z"}
#         for r in range(self.grid_size):
#             row = ""
#             for c in range(self.grid_size):
#                 v = grid[r, c]
#                 row += symbols.get(v, "?") + " "
#             lines.append(row)
#         print("\n".join(lines))

#     # ── Info helpers ──────────────────────────────────────────────────────────

#     @property
#     def true_syndrome_weight(self) -> int:
#         return self.error_state.syndrome_weight(self.x_checks, self.z_checks)

#     @property
#     def obs_shape(self):
#         return self.observation_space.shape

#     def __repr__(self):
#         return (
#             f"SurfaceCodeEnv(d={self.d}, noise={self.noise}, "
#             f"k={self.k}, actions={self.n_actions}, "
#             f"obs=(2k={2*self.k}, {self.grid_size}, {self.grid_size}))"
#         )
"""
Surface Code RL Environment (Stim-backed) — Distance 3
=========================================================

Uses Stim to simulate the rotated surface code with circuit-level noise.
This makes the RL agent and MWPM directly comparable on the same noise samples.

Design
------
At each `reset()`:
    1. Build the noisy d=3 surface code memory circuit  (k rounds + final readout)
    2. Sample ONE shot → (detector_events, true_observable)
       * detector_events : binary array of length n_detectors
         each detector = parity check between two stabilizer rounds
       * true_observable : the logical bit AFTER perfect decoding (Stim already
         flipped it depending on the actual errors)
    3. Reshape detectors into a (k, n_x_stabs + n_z_stabs) syndrome trajectory
    4. Embed each round into a 7×7 grid
    5. Initialise the agent's "Pauli frame"  → all zeros
       (this tracks the corrections the agent has applied so far)

At each `step(action)`:
    1. Apply the correction (X or Z on a data qubit) to the agent's Pauli frame
    2. Re-measure stabilizers based on (true_errors XOR agent_corrections)
       NOTE: we don't have direct access to true_errors, but we know that
             at each round the detector tells us the syndrome AT THAT ROUND.
       To "re-measure after correction" we simulate what the new syndrome
       would look like, given our applied corrections.
    3. Build new observation
    4. Compute reward & termination

Episode end:
    - Compare residual error pattern vs the logical observable
    - residual logical = true_observable  XOR  (agent_correction_logical)
    - if residual logical == 0  → success
    - if residual logical == 1  → logical error

State shape: (2k, 2d+1, 2d+1)  = (6, 7, 7)
Action space: 2*d² + 1 = 19    (X_i, Z_i, Identity)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import stim


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — Stim circuit and detector layout
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
    """
    Build a rotated surface code memory experiment with circuit-level noise.

    Stim's `surface_code:rotated_memory_z` template gives us:
      - data qubits at odd (x,y) with x,y ∈ {1,3,...,2d-1}
      - X- and Z-ancillas placed between them
      - `rounds` repetitions of the syndrome cycle
      - final destructive measurement of data qubits in Z basis
      - logical Z observable defined automatically
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )


def get_detector_layout(circuit: stim.Circuit):
    """
    Parse Stim's detector coordinates to figure out where each detector lives.

    Stim places each detector at (x, y, t):
      - (x, y) → grid position of the ancilla being checked
      - t      → round index (0, 1, ..., rounds-1)

    Returns
    -------
    detector_positions : list of (row, col, round)  one per detector
    n_per_round        : number of detectors per round
    rounds             : total number of rounds
    """
    coords = circuit.get_detector_coordinates()
    positions = []
    rounds_set = set()
    for i in range(circuit.num_detectors):
        x, y, t = coords[i]
        # In our grid convention: row = y, col = x
        positions.append((int(y), int(x), int(t)))
        rounds_set.add(int(t))

    n_rounds   = len(rounds_set)
    n_per_round = circuit.num_detectors // n_rounds
    return positions, n_per_round, n_rounds


def data_qubit_positions(distance: int):
    """
    Standard rotated surface code data qubit positions in (row, col).
    Matches Stim's coordinate convention: data qubits at odd (x, y).

    For d=3: (1,1), (1,3), (1,5), (3,1), (3,3), (3,5), (5,1), (5,3), (5,5)
    """
    return [(r, c) for r in range(1, 2*distance, 2)
                   for c in range(1, 2*distance, 2)]


# ══════════════════════════════════════════════════════════════════════════════
#  PARITY-CHECK MATRICES  (for syndrome update after corrections)
# ══════════════════════════════════════════════════════════════════════════════
#
# To know how the syndrome CHANGES when we apply a correction, we need to know
# which stabilizers each data qubit touches.
# We pre-compute this from the Stim circuit's stabilizer structure.

def build_check_matrices(circuit: stim.Circuit, distance: int):
    """
    Extract X- and Z-stabilizer check matrices from the Stim circuit.

    Returns
    -------
    x_stab_qubits : list of lists of data qubit indices (one per X-stab)
    z_stab_qubits : list of lists of data qubit indices (one per Z-stab)
    x_stab_pos    : grid positions (row, col) of X-stabilizers
    z_stab_pos    : grid positions (row, col) of Z-stabilizers
    data_pos      : grid positions of data qubits (index → (row, col))
    """
    data_pos = data_qubit_positions(distance)
    data_set = set(data_pos)
    pos_to_idx = {p: i for i, p in enumerate(data_pos)}

    size = 2 * distance + 1
    x_stab_pos, z_stab_pos = [], []
    x_stab_qubits, z_stab_qubits = [], []

    def neighbours(r, c):
        nb = []
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            p = (r + dr, c + dc)
            if p in data_set:
                nb.append(p)
        return nb

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


# ══════════════════════════════════════════════════════════════════════════════
#  SYNDROME EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def embed_syndrome_to_grid(
    x_syn: np.ndarray, z_syn: np.ndarray,
    x_stab_pos: list, z_stab_pos: list,
    grid_size: int,
) -> np.ndarray:
    """
    Embed binary syndrome bits into a (grid_size, grid_size) float32 grid.
    Encoding:
      X-stabilizer fired → +1.0
      Z-stabilizer fired → -1.0
      otherwise          →  0.0
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for bit, (r, c) in zip(x_syn, x_stab_pos):
        if bit:
            grid[r, c] = +1.0
    for bit, (r, c) in zip(z_syn, z_stab_pos):
        if bit:
            grid[r, c] = -1.0
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  THE ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class SurfaceCodeEnv(gym.Env):
    """
    Stim-backed surface code RL environment.

    Each episode:
      1. Stim samples ONE noisy execution → detector trace + true observable
      2. The k rounds of detectors are converted into syndrome grids → history
      3. The agent applies corrections (Pauli frame updates)
      4. After every action, the *displayed* syndrome is updated by XOR-ing
         the effect of the correction onto the latest round
      5. Episode ends when:
         - syndrome weight == 0   → check logical error → success or fail
         - max_steps reached       → check logical error → fail / timeout
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        distance:        int   = 3,
        noise:           float = 0.01,
        syndrome_rounds: int   = None,
        max_steps:       int   = 50,
        alpha:           float = 0.01,
        R_success:       float = 10.0,
        R_fail:          float = 10.0,
    ):
        super().__init__()
        assert distance % 2 == 1

        self.d         = distance
        self.noise     = noise
        self.k         = syndrome_rounds if syndrome_rounds is not None else distance
        self.max_steps = max_steps
        self.alpha     = alpha
        self.R_success = R_success
        self.R_fail    = R_fail

        # ── Build Stim circuit (k rounds) ─────────────────────────────────────
        self.circuit  = build_circuit(distance, self.k, noise)
        self.sampler  = self.circuit.compile_detector_sampler()
        # Detector error model for the logical observable parity check
        self.dem      = self.circuit.detector_error_model(decompose_errors=True)

        # ── Parse detector layout ─────────────────────────────────────────────
        self.detector_positions, self.n_per_round, n_rounds = \
            get_detector_layout(self.circuit)
        # Stim produces k+1 detector "rounds":
        #   round 0     : initial syndrome (Z-stabs only — agent's reset state)
        #   round 1..k-1: full syndrome rounds
        #   round k     : final data-measurement-derived syndrome (Z-stabs only)
        # For RL we want the k full rounds → indices [0 .. k-1] in detector trace.
        # We will simply use *all* rounds that Stim produces and treat the final
        # "fold-in" as part of the same trajectory.
        self._stim_rounds = n_rounds

        # ── Check matrices for syndrome updates after corrections ─────────────
        (self.x_stab_qubits,
         self.z_stab_qubits,
         self.x_stab_pos,
         self.z_stab_pos,
         self.data_pos) = build_check_matrices(self.circuit, distance)

        self.n_qubits  = len(self.data_pos)        # = d²
        self.n_x_stabs = len(self.x_stab_pos)
        self.n_z_stabs = len(self.z_stab_pos)
        self.grid_size = 2 * distance + 1

        # ── Map: detector index → (stabilizer index in current layout, type) ──
        # We need to know which detector corresponds to which (X or Z) stabilizer
        # in our pre-built grid layout.
        self._build_detector_to_stab_map()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.n_actions = 2 * self.n_qubits + 1
        self.action_space = spaces.Discrete(self.n_actions)
        self.ACTION_IDENTITY = self.n_actions - 1

        self.observation_space = spaces.Box(
            low  = -1.0, high = +1.0,
            shape = (2 * self.k, self.grid_size, self.grid_size),
            dtype = np.float32,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._syndrome_history = deque(maxlen=self.k)
        self._action_history   = deque(maxlen=self.k)

        # Current X/Z syndrome bits (most recent round, after agent's corrections)
        self._cur_x_syn = np.zeros(self.n_x_stabs, dtype=np.int8)
        self._cur_z_syn = np.zeros(self.n_z_stabs, dtype=np.int8)

        # Agent's accumulated Pauli frame (XOR-tracked)
        self._x_corr = np.zeros(self.n_qubits, dtype=np.int8)  # X corrections
        self._z_corr = np.zeros(self.n_qubits, dtype=np.int8)  # Z corrections

        # True logical observable from Stim
        self._true_observable = 0

        self._step_count  = 0
        self._prev_weight = 0

    # ── Build detector → stab map ─────────────────────────────────────────────

    def _build_detector_to_stab_map(self):
        """
        Map each detector to (round_idx, is_x_type, stab_idx_in_grid).
        Allows us to convert flat detector arrays into per-round X/Z syndrome.
        """
        # Position → stabilizer index in our layout
        x_pos_to_idx = {p: i for i, p in enumerate(self.x_stab_pos)}
        z_pos_to_idx = {p: i for i, p in enumerate(self.z_stab_pos)}

        self._det_map = []   # entry per detector: (round, type, idx)
        for (row, col, t) in self.detector_positions:
            pos = (row, col)
            if pos in x_pos_to_idx:
                self._det_map.append((t, "X", x_pos_to_idx[pos]))
            elif pos in z_pos_to_idx:
                self._det_map.append((t, "Z", z_pos_to_idx[pos]))
            else:
                # Shouldn't happen for a clean rotated surface code
                self._det_map.append((t, None, None))

    def _detectors_to_syndrome_rounds(self, detector_events: np.ndarray):
        """
        Convert a flat detector event array into per-round (X-syn, Z-syn).

        Stim's detectors are *parity checks between successive rounds*, so the
        cumulative syndrome at round t is the XOR of all detectors with that
        position seen up to and including round t.

        Returns
        -------
        x_rounds : list of length k, each entry shape (n_x_stabs,)
        z_rounds : list of length k, each entry shape (n_z_stabs,)
        Each entry holds the absolute (cumulative) syndrome at that round.
        """
        # We have self._stim_rounds total rounds in detector trace
        # Initialise per-round syndromes
        n_total = self._stim_rounds
        x_rounds = [np.zeros(self.n_x_stabs, dtype=np.int8) for _ in range(n_total)]
        z_rounds = [np.zeros(self.n_z_stabs, dtype=np.int8) for _ in range(n_total)]

        # Cumulative XOR per stabilizer
        cum_x = np.zeros(self.n_x_stabs, dtype=np.int8)
        cum_z = np.zeros(self.n_z_stabs, dtype=np.int8)

        # We need to walk through detectors in temporal order. Stim's detectors
        # are already ordered by round, but we group by round to update one at a time.
        detectors_by_round = [[] for _ in range(n_total)]
        for det_idx, (t, stab_type, stab_idx) in enumerate(self._det_map):
            detectors_by_round[t].append((det_idx, stab_type, stab_idx))

        for t in range(n_total):
            for det_idx, stab_type, stab_idx in detectors_by_round[t]:
                if stab_type is None:
                    continue
                bit = int(detector_events[det_idx])
                if stab_type == "X":
                    cum_x[stab_idx] ^= bit
                else:
                    cum_z[stab_idx] ^= bit
            x_rounds[t] = cum_x.copy()
            z_rounds[t] = cum_z.copy()

        return x_rounds, z_rounds

    # ──────────────────────────────────────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Seed Stim sampler for reproducibility (uses NumPy generator internally)
        # Stim doesn't expose a seed param on sample(), but our env uses
        # gym's np_random for downstream randomness.

        # 1. Sample one noisy execution from Stim
        detector_events, observables = self.sampler.sample(
            shots=1, separate_observables=True
        )
        # detector_events shape: (1, n_detectors)
        # observables shape:     (1, 1)

        det = detector_events[0].astype(np.int8)
        self._true_observable = int(observables[0, 0])

        # 2. Convert to per-round X / Z syndromes (cumulative)
        x_rounds, z_rounds = self._detectors_to_syndrome_rounds(det)
        # Stim gives us self._stim_rounds rounds; we use the LAST k for history.
        x_rounds = x_rounds[-self.k:]
        z_rounds = z_rounds[-self.k:]

        # 3. Most recent round becomes the "current" syndrome
        self._cur_x_syn = x_rounds[-1].copy()
        self._cur_z_syn = z_rounds[-1].copy()

        # Save the Stim-sampled full detector trace for benchmarking
        self._stim_detectors = det
        self._stim_x_rounds  = x_rounds
        self._stim_z_rounds  = z_rounds

        # 4. Initialise observation history with Stim's k syndrome rounds
        self._syndrome_history.clear()
        self._action_history.clear()
        zero_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for t in range(self.k):
            grid = embed_syndrome_to_grid(
                x_rounds[t], z_rounds[t],
                self.x_stab_pos, self.z_stab_pos,
                self.grid_size,
            )
            self._syndrome_history.append(grid)
            self._action_history.append(zero_grid.copy())

        # 5. Reset Pauli frame
        self._x_corr[:] = 0
        self._z_corr[:] = 0

        self._step_count  = 0
        self._prev_weight = int(self._cur_x_syn.sum() + self._cur_z_syn.sum())

        return self._get_obs(), self._info()

    # ──────────────────────────────────────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1

        # ── 1. Apply correction to Pauli frame + build action grid ────────────
        action_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if action < self.n_qubits:                         # X correction
            qubit = action
            self._x_corr[qubit] ^= 1
            # Toggle the Z-stabilizers that this qubit touches
            # (X errors flip Z-stabilizers)
            for s_idx, qubits in enumerate(self.z_stab_qubits):
                if qubit in qubits:
                    self._cur_z_syn[s_idx] ^= 1
            r, c = self.data_pos[qubit]
            action_grid[r, c] = +1.0

        elif action < 2 * self.n_qubits:                   # Z correction
            qubit = action - self.n_qubits
            self._z_corr[qubit] ^= 1
            for s_idx, qubits in enumerate(self.x_stab_qubits):
                if qubit in qubits:
                    self._cur_x_syn[s_idx] ^= 1
            r, c = self.data_pos[qubit]
            action_grid[r, c] = -1.0
        # else Identity → no change

        # ── 2. Append new syndrome grid + action grid to history ──────────────
        syn_grid = embed_syndrome_to_grid(
            self._cur_x_syn, self._cur_z_syn,
            self.x_stab_pos, self.z_stab_pos,
            self.grid_size,
        )
        self._syndrome_history.append(syn_grid)
        self._action_history.append(action_grid)

        # ── 3. Check termination ──────────────────────────────────────────────
        new_weight = int(self._cur_x_syn.sum() + self._cur_z_syn.sum())

        # The agent has "decoded" once syndrome is trivial
        if new_weight == 0:
            # Check logical: did the agent's corrections flip the logical observable?
            logical_residual = self._compute_residual_logical()
            corrected   = (logical_residual == 0)
            logical_err = (logical_residual == 1)
            terminated  = True
            truncated   = False
        else:
            corrected   = False
            logical_err = False
            terminated  = False
            truncated   = self._step_count >= self.max_steps

        # ── 4. Reward ─────────────────────────────────────────────────────────
        reward = float(self._prev_weight - new_weight) - self.alpha
        if corrected:
            reward += self.R_success
        elif logical_err:
            reward -= self.R_fail
        elif truncated:
            # No bonus / penalty on timeout — just step penalty already applied
            pass

        self._prev_weight = new_weight

        return self._get_obs(), reward, terminated, truncated, self._info(
            corrected=corrected, logical_err=logical_err
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  LOGICAL RESIDUAL
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_residual_logical(self) -> int:
        """
        Compute whether the agent's corrections (combined with the true error)
        result in a logical error.

        Stim's `_true_observable` is what the logical bit reads WITHOUT any
        correction — it already reflects the true error pattern.
        Our agent's X corrections form a Pauli string; whether that string
        flips the logical Z observable depends on whether the total X correction
        crosses the logical Z operator.

        For a memory_z experiment:
          - Logical Z observable is along a column of data qubits.
          - X corrections that hit an ODD number of qubits along that column
            will flip the logical observable.
        """
        d = self.d
        # X corrections in d×d grid
        x_frame = self._x_corr.reshape(d, d)
        # Logical Z observable = product of Z on first column (Stim convention)
        # Number of X errors along that column changes whether observable flips
        x_correction_logical = int(x_frame[:, 0].sum() % 2)

        # Residual = stim observable XOR agent's correction effect
        residual = self._true_observable ^ x_correction_logical
        return residual

    # ──────────────────────────────────────────────────────────────────────────
    #  OBSERVATION
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        zero = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        syn_frames = list(self._syndrome_history)
        while len(syn_frames) < self.k:
            syn_frames.insert(0, zero.copy())
        act_frames = list(self._action_history)
        while len(act_frames) < self.k:
            act_frames.insert(0, zero.copy())
        return np.stack(syn_frames + act_frames, axis=0).astype(np.float32)

    def _info(self, corrected=False, logical_err=False) -> dict:
        return {
            "syndrome_weight": int(self._cur_x_syn.sum() + self._cur_z_syn.sum()),
            "logical_error":   logical_err,
            "corrected":       corrected,
            "step":            self._step_count,
            "true_observable": self._true_observable,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  BENCHMARKING HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def get_stim_detectors(self) -> np.ndarray:
        """
        Return the raw flat detector array of the current episode
        (suitable for MWPM input).
        """
        return self._stim_detectors.copy()

    def get_true_observable(self) -> int:
        """Return Stim's true observable for this episode (0 or 1)."""
        return self._true_observable

    # ──────────────────────────────────────────────────────────────────────────
    #  RENDER + REPR
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, mode="human"):
        print(f"\n── Step {self._step_count}  weight={self._prev_weight} ──")

    def __repr__(self):
        return (
            f"SurfaceCodeEnv(d={self.d}, noise={self.noise}, "
            f"k={self.k}, actions={self.n_actions}, "
            f"obs=(2k={2*self.k},{self.grid_size},{self.grid_size}))"
        )