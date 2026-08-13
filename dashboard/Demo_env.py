"""
DemoSurfaceCodeEnv  —  code-capacity d=3 rotated surface code for the live dashboard
====================================================================================

What the user asked for
-----------------------
  * noise = depolarizing on the data qubits only
  * perfect encoding (and perfect stabilizer measurement)
  * watch errors fire stabilizers, then watch the trained agent apply corrections

Design (two layers, kept deliberately separate)
-----------------------------------------------
1. AGENT-FACING layer  (must match training so the network stays in-distribution):
   the observation tensor is built with the SAME 7x7, 16-plaquette layout and the
   SAME embedding / action toggling as src/environments/surface3_env.py.
       - syndrome embedding : X-stab fired -> +1, Z-stab fired -> -1
       - action  a < 9      : X on qubit a
                 9<=a<18    : Z on qubit a-9
                 a == 18    : identity
       - perfect measurement -> the k=3 syndrome "rounds" are identical copies.

2. PHYSICS layer  (the real, correct rotated surface code):
   only the 8 genuine stabilizers (4 X + 4 Z) carry physical meaning. We use them
   to compute which stabilizers actually fire for a given error, to render the
   lattice, and to decide logical failure exactly via the real logical operators
   (logZ = left column {0,3,6}, logX = top row {0,1,2}, both weight 3).

The genuine stabilizers are recovered from Stim (the same source the training env
uses) by intersecting the geometric plaquettes with Stim's real ancilla positions.
A hard-coded d=3 fallback is used if Stim is unavailable.

Termination is decided on the REAL syndrome (codespace reached), which is the
physically correct stopping point and avoids the geometric layout's spurious
weight-1 plaquettes ever making a corrected error look unsolved.
"""

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
#  GEOMETRY  (copied from surface3_env so the agent layout matches training)
# ════════════════════════════════════════════════════════════════════════════

def data_qubit_positions(distance):
    return [(r, c) for r in range(1, 2 * distance, 2)
                   for c in range(1, 2 * distance, 2)]


def build_check_matrices(distance):
    """Geometric 16-plaquette layout used for the agent's observation/toggling."""
    data_pos = data_qubit_positions(distance)
    data_set = set(data_pos)
    pos_to_idx = {p: i for i, p in enumerate(data_pos)}
    size = 2 * distance + 1
    x_stab_pos, z_stab_pos, x_stab_qubits, z_stab_qubits = [], [], [], []

    def neighbours(r, c):
        return [(r + dr, c + dc) for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                if (r + dr, c + dc) in data_set]

    for r in range(size):
        for c in range(size):
            if r % 2 == 1 and c % 2 == 1:
                continue
            nb = neighbours(r, c)
            if not nb:
                continue
            if (r + c) % 4 == 0:
                x_stab_pos.append((r, c)); x_stab_qubits.append([pos_to_idx[p] for p in nb])
            else:
                z_stab_pos.append((r, c)); z_stab_qubits.append([pos_to_idx[p] for p in nb])
    return x_stab_qubits, z_stab_qubits, x_stab_pos, z_stab_pos, data_pos


def embed_syndrome_to_grid(x_syn, z_syn, x_stab_pos, z_stab_pos, grid_size):
    """X-stab fired -> +1.0 ; Z-stab fired -> -1.0 ; else 0.0 (training convention)."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for bit, (r, c) in zip(x_syn, x_stab_pos):
        if bit:
            grid[r, c] = +1.0
    for bit, (r, c) in zip(z_syn, z_stab_pos):
        if bit:
            grid[r, c] = -1.0
    return grid


# ════════════════════════════════════════════════════════════════════════════
#  REAL STABILIZERS  (from Stim, with a validated d=3 fallback)
# ════════════════════════════════════════════════════════════════════════════

_FALLBACK_D3 = {
    "x": {(2, 2): [0, 1, 3, 4], (2, 6): [2, 5], (4, 0): [3, 6], (4, 4): [4, 5, 7, 8]},
    "z": {(0, 2): [0, 1], (2, 4): [1, 2, 4, 5], (4, 2): [3, 4, 6, 7], (6, 4): [7, 8]},
}


def real_stabilizers(distance, x_stab_pos, x_stab_qubits, z_stab_pos, z_stab_qubits):
    """
    Return {(row,col): [qubit indices]} for the genuine X- and Z-stabilizers,
    recovered from Stim's real ancilla positions (fallback to hard-coded d=3).
    """
    ancilla_pos = None
    try:
        import stim
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=distance,
            before_round_data_depolarization=0.01,
        )
        coords = circ.get_detector_coordinates()
        ancilla_pos = {(int(coords[i][1]), int(coords[i][0]))   # (row, col)
                       for i in range(circ.num_detectors)}
    except Exception:
        if distance == 3:
            return ({p: list(q) for p, q in _FALLBACK_D3["x"].items()},
                    {p: list(q) for p, q in _FALLBACK_D3["z"].items()})
        raise

    real_x = {p: list(q) for p, q in zip(x_stab_pos, x_stab_qubits) if p in ancilla_pos}
    real_z = {p: list(q) for p, q in zip(z_stab_pos, z_stab_qubits) if p in ancilla_pos}
    if not real_x or not real_z:                     # convention mismatch -> fallback
        if distance == 3:
            return ({p: list(q) for p, q in _FALLBACK_D3["x"].items()},
                    {p: list(q) for p, q in _FALLBACK_D3["z"].items()})
        raise RuntimeError("Could not recover real stabilizers from Stim.")
    return real_x, real_z


# ════════════════════════════════════════════════════════════════════════════
#  GF(2) LINEAR ALGEBRA  →  exact logical operators
# ════════════════════════════════════════════════════════════════════════════

def _gf2_rref(M):
    M = (np.array(M, dtype=np.int8) % 2).copy()
    rows, cols = M.shape
    pivots, r = [], 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c); r += 1
        if r == rows:
            break
    return M, pivots


def _gf2_nullspace(M):
    M = np.array(M, dtype=np.int8) % 2
    cols = M.shape[1]
    R, pivots = _gf2_rref(M)
    free = [c for c in range(cols) if c not in set(pivots)]
    basis = []
    for f in free:
        vec = np.zeros(cols, dtype=np.int8); vec[f] = 1
        for ridx, pc in enumerate(pivots):
            if R[ridx, f]:
                vec[pc] = 1
        basis.append(vec)
    return basis


def _in_rowspace(v, M):
    R, pivots = _gf2_rref(M)
    v = (np.array(v, dtype=np.int8) % 2).copy()
    for ridx, pc in enumerate(pivots):
        if v[pc]:
            v ^= R[ridx]
    return not v.any()


def _logical_operators(Mx, Mz):
    logZ = next(v for v in _gf2_nullspace(Mx) if not _in_rowspace(v, Mz))
    logX = next(v for v in _gf2_nullspace(Mz) if not _in_rowspace(v, Mx))
    return logX.astype(np.int8), logZ.astype(np.int8)


# ════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════════

class DemoSurfaceCodeEnv:

    def __init__(self, distance=3, k_rounds=3, max_steps=50):
        assert distance % 2 == 1
        self.d, self.k, self.max_steps = distance, k_rounds, max_steps
        self.grid_size = 2 * distance + 1

        # ── agent-facing geometric layout (16 plaquettes) ─────────────────────
        (self.x_stab_qubits, self.z_stab_qubits,
         self.x_stab_pos, self.z_stab_pos, self.data_pos) = build_check_matrices(distance)
        self.n_qubits  = len(self.data_pos)          # 9
        self.n_actions = 2 * self.n_qubits + 1       # 19
        self.ACTION_IDENTITY = self.n_actions - 1

        # ── physics: the genuine stabilizers ──────────────────────────────────
        real_x, real_z = real_stabilizers(distance, self.x_stab_pos, self.x_stab_qubits,
                                           self.z_stab_pos, self.z_stab_qubits)
        self.real_x_pos = list(real_x.keys());  self.real_x_sup = list(real_x.values())
        self.real_z_pos = list(real_z.keys());  self.real_z_sup = list(real_z.values())

        # index of each real stab inside the 16-plaquette arrays
        xpi = {p: i for i, p in enumerate(self.x_stab_pos)}
        zpi = {p: i for i, p in enumerate(self.z_stab_pos)}
        self.real_x_idx16 = [xpi[p] for p in self.real_x_pos]
        self.real_z_idx16 = [zpi[p] for p in self.real_z_pos]

        # real check matrices + logicals (+ integrity asserts)
        self.Mx = np.zeros((len(self.real_x_sup), self.n_qubits), np.int8)
        for i, qs in enumerate(self.real_x_sup): self.Mx[i, qs] = 1
        self.Mz = np.zeros((len(self.real_z_sup), self.n_qubits), np.int8)
        for i, qs in enumerate(self.real_z_sup): self.Mz[i, qs] = 1
        self.logX, self.logZ = _logical_operators(self.Mx, self.Mz)
        assert not (self.Mx @ self.Mz.T % 2).any(), "stabilizers must commute (CSS)"
        assert int(self.logX @ self.logZ % 2) == 1, "logX/logZ must anticommute"

        self._reset_state()

    # ── internal state ────────────────────────────────────────────────────────
    def _reset_state(self):
        z = lambda n: np.zeros(n, dtype=np.int8)
        self._x_error = z(self.n_qubits);  self._z_error = z(self.n_qubits)
        self._x_corr  = z(self.n_qubits);  self._z_corr  = z(self.n_qubits)
        self._cur_x_syn16 = z(len(self.x_stab_pos))   # agent-facing syndrome
        self._cur_z_syn16 = z(len(self.z_stab_pos))
        self._syn_history, self._act_history = [], []
        self._step_count, self._last_action = 0, None
        self._prev_real_weight = 0

    # ── sampling / syndrome ───────────────────────────────────────────────────
    def _sample_depolarizing(self, p, rng):
        x_err = np.zeros(self.n_qubits, np.int8)
        z_err = np.zeros(self.n_qubits, np.int8)
        for q in range(self.n_qubits):
            if rng.random() < p:
                pauli = rng.integers(3)          # 0=X 1=Y 2=Z
                if pauli in (0, 1): x_err[q] = 1
                if pauli in (1, 2): z_err[q] = 1
        return x_err, z_err

    def _seed_syndrome(self, x_err, z_err):
        """Set the agent-facing syndrome from the real syndrome of the error."""
        self._cur_x_syn16[:] = 0;  self._cur_z_syn16[:] = 0
        for i, sup in enumerate(self.real_x_sup):           # X-stabs detect Z errors
            if int(z_err[sup].sum()) % 2:
                self._cur_x_syn16[self.real_x_idx16[i]] = 1
        for i, sup in enumerate(self.real_z_sup):           # Z-stabs detect X errors
            if int(x_err[sup].sum()) % 2:
                self._cur_z_syn16[self.real_z_idx16[i]] = 1

    def _real_weight(self):
        return (int(self._cur_x_syn16[self.real_x_idx16].sum())
                + int(self._cur_z_syn16[self.real_z_idx16].sum()))

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, p=0.05, seed=None, error=None):
        self._reset_state()
        if error is not None:
            x_err = np.array(error[0], np.int8) % 2
            z_err = np.array(error[1], np.int8) % 2
        else:
            x_err, z_err = self._sample_depolarizing(p, np.random.default_rng(seed))
        self._x_error, self._z_error = x_err, z_err
        self._seed_syndrome(x_err, z_err)

        grid = embed_syndrome_to_grid(self._cur_x_syn16, self._cur_z_syn16,
                                      self.x_stab_pos, self.z_stab_pos, self.grid_size)
        zero = np.zeros((self.grid_size, self.grid_size), np.float32)
        self._syn_history = [grid.copy() for _ in range(self.k)]   # perfect measurement
        self._act_history = [zero.copy() for _ in range(self.k)]
        self._prev_real_weight = self._real_weight()
        return self._get_obs(), self._info()

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action):
        assert 0 <= action < self.n_actions
        self._step_count += 1
        self._last_action = int(action)
        action_grid = np.zeros((self.grid_size, self.grid_size), np.float32)

        if action < self.n_qubits:                          # X correction
            q = action; self._x_corr[q] ^= 1
            for s, qs in enumerate(self.z_stab_qubits):     # X flips Z-stabs
                if q in qs: self._cur_z_syn16[s] ^= 1
            r, c = self.data_pos[q]; action_grid[r, c] = +1.0
        elif action < 2 * self.n_qubits:                    # Z correction
            q = action - self.n_qubits; self._z_corr[q] ^= 1
            for s, qs in enumerate(self.x_stab_qubits):     # Z flips X-stabs
                if q in qs: self._cur_x_syn16[s] ^= 1
            r, c = self.data_pos[q]; action_grid[r, c] = -1.0
        # else identity

        syn_grid = embed_syndrome_to_grid(self._cur_x_syn16, self._cur_z_syn16,
                                          self.x_stab_pos, self.z_stab_pos, self.grid_size)
        self._syn_history = (self._syn_history + [syn_grid])[-self.k:]
        self._act_history = (self._act_history + [action_grid])[-self.k:]

        real_w = self._real_weight()
        if real_w == 0:                                     # codespace reached
            logical_err = self.is_logical_error()
            corrected, terminated, truncated = (not logical_err), True, False
        else:
            logical_err, corrected, terminated = False, False, False
            truncated = self._step_count >= self.max_steps

        reward = float(self._prev_real_weight - real_w) - 0.01
        reward += 10.0 if corrected else (-10.0 if logical_err else 0.0)
        self._prev_real_weight = real_w
        return (self._get_obs(), reward, terminated, truncated,
                self._info(corrected=corrected, logical_err=logical_err))

    # ── logical verdict (exact) ───────────────────────────────────────────────
    def is_logical_error(self):
        res_x = (self._x_error ^ self._x_corr) % 2
        res_z = (self._z_error ^ self._z_corr) % 2
        return bool(int(res_x @ self.logZ % 2) or int(res_z @ self.logX % 2))

    def logical_breakdown(self):
        res_x = (self._x_error ^ self._x_corr) % 2
        res_z = (self._z_error ^ self._z_corr) % 2
        return {"bit_flip(X)": int(res_x @ self.logZ % 2),     # crosses logical Z
                "phase(Z)":    int(res_z @ self.logX % 2)}     # crosses logical X

    # ── observation / info / render ───────────────────────────────────────────
    def _get_obs(self):
        return np.stack(self._syn_history + self._act_history, axis=0).astype(np.float32)

    def _info(self, corrected=False, logical_err=False):
        return {"syndrome_weight": self._real_weight(),
                "corrected": corrected, "logical_error": logical_err,
                "step": self._step_count}

    def get_render_state(self):
        x_fired = [bool(self._cur_x_syn16[self.real_x_idx16[i]]) for i in range(len(self.real_x_pos))]
        z_fired = [bool(self._cur_z_syn16[self.real_z_idx16[i]]) for i in range(len(self.real_z_pos))]
        return {
            "grid_size": self.grid_size, "n_qubits": self.n_qubits,
            "data_pos": list(self.data_pos),
            "real_x_pos": list(self.real_x_pos), "real_x_sup": [list(s) for s in self.real_x_sup],
            "real_z_pos": list(self.real_z_pos), "real_z_sup": [list(s) for s in self.real_z_sup],
            "x_fired": x_fired, "z_fired": z_fired,
            "x_error": self._x_error.copy(), "z_error": self._z_error.copy(),
            "x_corr": self._x_corr.copy(), "z_corr": self._z_corr.copy(),
            "last_action": self._last_action, "weight": self._real_weight(),
            "logX": self.logX.copy(), "logZ": self.logZ.copy(),
        }


def action_label(a, n_qubits=9):
    if a is None:
        return "-"
    if a < n_qubits:
        return f"X·q{a}"
    if a < 2 * n_qubits:
        return f"Z·q{a - n_qubits}"
    return "Idle"


# ════════════════════════════════════════════════════════════════════════════
#  MWPM baseline (PyMatching) on the real check matrices  —  optional
# ════════════════════════════════════════════════════════════════════════════

class MWPMDecoder:
    """Independent X/Z minimum-weight perfect matching on the real code."""

    def __init__(self, env: DemoSurfaceCodeEnv):
        self.env = env
        self.ok = False
        try:
            from pymatching import Matching
            self.mx = Matching(env.Mz)      # Z-stabs / X errors sector
            self.mz = Matching(env.Mx)      # X-stabs / Z errors sector
            self.ok = True
        except Exception as e:
            self.err = str(e)

    def decode(self, x_error, z_error):
        """Return (corrected: bool, x_corr, z_corr) for the same error."""
        if not self.ok:
            return None
        z_syn = (self.env.Mz @ x_error) % 2
        x_syn = (self.env.Mx @ z_error) % 2
        x_corr = self.mx.decode(z_syn).astype(np.int8)   # predicted X correction
        z_corr = self.mz.decode(x_syn).astype(np.int8)   # predicted Z correction
        res_x = (x_error ^ x_corr) % 2
        res_z = (z_error ^ z_corr) % 2
        fail = bool(int(res_x @ self.env.logZ % 2) or int(res_z @ self.env.logX % 2))
        return (not fail), x_corr, z_corr