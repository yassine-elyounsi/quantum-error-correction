"""
Surface Code RL Environment — Distance 5 (Stim-backed)
========================================================
Identical design to the d=3 environment. Only the parameters change:

  d=3  →  state (6,  7,  7 )   actions 19   detectors 24
  d=5  →  state (10, 11, 11)   actions 51   detectors 120

The environment is fully general and works for any odd distance d.
Set distance=5 to get the d=5 configuration.

State shape : (2k, 2d+1, 2d+1) = (10, 11, 11)   for d=5, k=5
Action space: 2*d² + 1          = 51              for d=5
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import stim


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS  (identical to d=3 env — fully general)
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit(distance: int, rounds: int, noise: float) -> stim.Circuit:
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
    coords     = circuit.get_detector_coordinates()
    positions  = []
    rounds_set = set()
    for i in range(circuit.num_detectors):
        x, y, t = coords[i]
        positions.append((int(y), int(x), int(t)))
        rounds_set.add(int(t))
    n_rounds    = len(rounds_set)
    n_per_round = circuit.num_detectors // n_rounds
    return positions, n_per_round, n_rounds


def data_qubit_positions(distance: int):
    """Data qubits at odd (row, col) — matches Stim's rotated code layout."""
    return [(r, c) for r in range(1, 2*distance, 2)
                   for c in range(1, 2*distance, 2)]


def build_check_matrices(circuit: stim.Circuit, distance: int):
    """
    Build X/Z stabilizer check matrices from the grid layout.
    Uses the same boundary-aware rule as the d=3 environment:
      X-stabs: top/bottom boundary + interior  (r+c) % 4 == 2
      Z-stabs: left/right boundary + interior  (r+c) % 4 == 0
    """
    data_pos  = data_qubit_positions(distance)
    data_set  = set(data_pos)
    pos_to_idx = {p: i for i, p in enumerate(data_pos)}
    size       = 2 * distance + 1

    x_stab_pos, z_stab_pos     = [], []
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


def embed_syndrome_to_grid(
    x_syn, z_syn, x_stab_pos, z_stab_pos, grid_size
) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for bit, (r, c) in zip(x_syn, x_stab_pos):
        if bit: grid[r, c] = +1.0
    for bit, (r, c) in zip(z_syn, z_stab_pos):
        if bit: grid[r, c] = -1.0
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class SurfaceCodeEnv(gym.Env):
    """
    Stim-backed surface code RL environment.
    General implementation — works for any odd distance d.
    For d=5: state (10,11,11), 51 actions.

    Parameters
    ----------
    distance        : code distance (default 5)
    noise           : physical error rate (circuit-level depolarizing)
    syndrome_rounds : k syndrome rounds to stack  (default = d = 5)
    max_steps       : max correction steps per episode
    alpha           : per-step penalty
    R_success       : terminal bonus on success
    R_fail          : terminal penalty on logical error
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        distance:        int   = 5,
        noise:           float = 0.01,
        syndrome_rounds: int   = None,
        max_steps:       int   = 100,
        alpha:           float = 0.01,
        R_success:       float = 10.0,
        R_fail:          float = 10.0,
    ):
        super().__init__()
        assert distance % 2 == 1, "distance must be odd"

        self.d         = distance
        self.noise     = noise
        self.k         = syndrome_rounds if syndrome_rounds is not None else distance
        self.max_steps = max_steps
        self.alpha     = alpha
        self.R_success = R_success
        self.R_fail    = R_fail

        # ── Stim circuit ──────────────────────────────────────────────────────
        self.circuit = build_circuit(distance, self.k, noise)
        self.sampler = self.circuit.compile_detector_sampler()
        self.dem     = self.circuit.detector_error_model(decompose_errors=True)

        # ── Detector layout ───────────────────────────────────────────────────
        self.detector_positions, self.n_per_round, n_rounds = \
            get_detector_layout(self.circuit)
        self._stim_rounds = n_rounds

        # ── Check matrices ────────────────────────────────────────────────────
        (self.x_stab_qubits,
         self.z_stab_qubits,
         self.x_stab_pos,
         self.z_stab_pos,
         self.data_pos) = build_check_matrices(self.circuit, distance)

        self.n_qubits  = len(self.data_pos)       # d² = 25
        self.n_x_stabs = len(self.x_stab_pos)
        self.n_z_stabs = len(self.z_stab_pos)
        self.grid_size = 2 * distance + 1          # 11

        # ── Detector → stabilizer map ─────────────────────────────────────────
        self._build_detector_to_stab_map()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.n_actions = 2 * self.n_qubits + 1    # 51
        self.action_space = spaces.Discrete(self.n_actions)
        self.ACTION_IDENTITY = self.n_actions - 1

        self.observation_space = spaces.Box(
            low=-1.0, high=+1.0,
            shape=(2 * self.k, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._syndrome_history = deque(maxlen=self.k)
        self._action_history   = deque(maxlen=self.k)
        self._cur_x_syn        = np.zeros(self.n_x_stabs, dtype=np.int8)
        self._cur_z_syn        = np.zeros(self.n_z_stabs, dtype=np.int8)
        self._x_corr           = np.zeros(self.n_qubits,  dtype=np.int8)
        self._z_corr           = np.zeros(self.n_qubits,  dtype=np.int8)
        self._true_observable  = 0
        self._step_count       = 0
        self._prev_weight      = 0

    # ── Detector → stab map ───────────────────────────────────────────────────

    def _build_detector_to_stab_map(self):
        x_pos_to_idx = {p: i for i, p in enumerate(self.x_stab_pos)}
        z_pos_to_idx = {p: i for i, p in enumerate(self.z_stab_pos)}
        self._det_map = []
        for (row, col, t) in self.detector_positions:
            pos = (row, col)
            if pos in x_pos_to_idx:
                self._det_map.append((t, "X", x_pos_to_idx[pos]))
            elif pos in z_pos_to_idx:
                self._det_map.append((t, "Z", z_pos_to_idx[pos]))
            else:
                self._det_map.append((t, None, None))

    def _detectors_to_syndrome_rounds(self, detector_events: np.ndarray):
        n_total  = self._stim_rounds
        x_rounds = [np.zeros(self.n_x_stabs, dtype=np.int8) for _ in range(n_total)]
        z_rounds = [np.zeros(self.n_z_stabs, dtype=np.int8) for _ in range(n_total)]
        cum_x    = np.zeros(self.n_x_stabs, dtype=np.int8)
        cum_z    = np.zeros(self.n_z_stabs, dtype=np.int8)

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

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample one noisy Stim execution
        detector_events, observables = self.sampler.sample(
            shots=1, separate_observables=True
        )
        det = detector_events[0].astype(np.int8)
        self._true_observable = int(observables[0, 0])

        # Convert detectors → per-round syndromes, take last k rounds
        x_rounds, z_rounds = self._detectors_to_syndrome_rounds(det)
        x_rounds = x_rounds[-self.k:]
        z_rounds = z_rounds[-self.k:]

        self._cur_x_syn = x_rounds[-1].copy()
        self._cur_z_syn = z_rounds[-1].copy()

        # Save for benchmarking
        self._stim_detectors = det
        self._stim_x_rounds  = x_rounds
        self._stim_z_rounds  = z_rounds

        # Build syndrome + action history
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

        # Reset Pauli frame
        self._x_corr[:] = 0
        self._z_corr[:] = 0
        self._step_count  = 0
        self._prev_weight = int(self._cur_x_syn.sum() + self._cur_z_syn.sum())

        return self._get_obs(), self._info()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1

        # Build action grid + apply correction to Pauli frame
        action_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if action < self.n_qubits:                      # X correction
            qubit = action
            self._x_corr[qubit] ^= 1
            for s_idx, qubits in enumerate(self.z_stab_qubits):
                if qubit in qubits:
                    self._cur_z_syn[s_idx] ^= 1
            r, c = self.data_pos[qubit]
            action_grid[r, c] = +1.0

        elif action < 2 * self.n_qubits:                # Z correction
            qubit = action - self.n_qubits
            self._z_corr[qubit] ^= 1
            for s_idx, qubits in enumerate(self.x_stab_qubits):
                if qubit in qubits:
                    self._cur_x_syn[s_idx] ^= 1
            r, c = self.data_pos[qubit]
            action_grid[r, c] = -1.0
        # else: Identity — no change

        # Update history
        syn_grid = embed_syndrome_to_grid(
            self._cur_x_syn, self._cur_z_syn,
            self.x_stab_pos, self.z_stab_pos,
            self.grid_size,
        )
        self._syndrome_history.append(syn_grid)
        self._action_history.append(action_grid)

        # Termination
        new_weight  = int(self._cur_x_syn.sum() + self._cur_z_syn.sum())
        corrected   = False
        logical_err = False

        if new_weight == 0:
            logical_residual = self._compute_residual_logical()
            corrected   = (logical_residual == 0)
            logical_err = (logical_residual == 1)
            terminated  = True
            truncated   = False
        else:
            terminated = False
            truncated  = self._step_count >= self.max_steps

        # Reward
        reward = float(self._prev_weight - new_weight) - self.alpha
        if corrected:
            reward += self.R_success
        elif logical_err:
            reward -= self.R_fail

        self._prev_weight = new_weight

        return self._get_obs(), reward, terminated, truncated, self._info(
            corrected=corrected, logical_err=logical_err
        )

    # ── Logical residual ──────────────────────────────────────────────────────

    def _compute_residual_logical(self) -> int:
        """
        Check whether agent's X corrections create a logical error.
        Logical Z operator = parity of first column of data qubit grid.
        """
        d = self.d
        x_frame = self._x_corr.reshape(d, d)
        x_correction_logical = int(x_frame[:, 0].sum() % 2)
        return self._true_observable ^ x_correction_logical

    # ── Observation ───────────────────────────────────────────────────────────

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

    # ── Benchmarking helpers ──────────────────────────────────────────────────

    def get_stim_detectors(self) -> np.ndarray:
        return self._stim_detectors.copy()

    def get_true_observable(self) -> int:
        return self._true_observable

    # ── Render / repr ─────────────────────────────────────────────────────────

    def render(self, mode="human"):
        print(f"\n── Step {self._step_count}  weight={self._prev_weight} ──")

    def __repr__(self):
        return (
            f"SurfaceCodeEnv(d={self.d}, noise={self.noise}, "
            f"k={self.k}, actions={self.n_actions}, "
            f"obs=(2k={2*self.k},{self.grid_size},{self.grid_size}))"
        )