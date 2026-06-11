"""
RL environment for QEC code + encoding discovery.

This wraps the Clifford simulator (clifford_sim.py) into an OpenAI-Gym-style
environment. Each episode is one attempt to build a valid encoding circuit
from scratch.

The environment supports five distinct modes from the paper:

1.  Symmetric noise           -- uniform reward weights, X/Y/Z all equally likely.
2.  Asymmetric noise          -- weights set by per-error probabilities p_mu
                                 (Eq. 14 of the paper).
3.  Noise-aware meta-training -- noise bias c_Z appears as an extra observation,
                                 so a single agent can learn many noise regimes.
4.  Stabilizer-code search    -- agent picks any gate from the gate set.
5.  CSS-code search           -- circuit constrained to a Hadamard prefix
                                 followed by CNOT-only suffix (Section V),
                                 enforced by action masking.

Observation (default mode):
    flattened binary check matrix, shape ((n - k) * 2 * n,).

Observation (noise-aware mode):
    [flattened G,  c_Z]      shape ((n - k) * 2 * n + 1,).

Action:
    index into a discrete list of allowed Clifford gates.
    Illegal-in-current-mode actions are reported via `action_mask()`.

Reward (per step):
    Knill-Laflamme reward, Eq. (10) of the paper:
        r_t = - sum_mu  lambda_mu * K_mu
    where K_mu = 0 if error E_mu is detected, else 1.
    Plus  +success_bonus  on full success, -fail_penalty on timeout.
"""

import numpy as np
from Encoders.Clifford_sim import (
    CliffordTableau,
    pauli_strings_to_binary,
    weighted_kl_sum,
    kl_undetected_mask,
    all_pauli_strings_up_to_weight,
)


# ====================================================================
# Helpers for the noise-aware reward
# ====================================================================

def depolarizing_error_probabilities(error_strings, n, p_I=0.9, c_Z=1.0):
    """
    Per-error probabilities under a global depolarizing channel with
    asymmetry parameter c_Z (Eq. 6 of the paper).

    Args:
        error_strings : list of Pauli strings.
        n             : number of physical qubits.
        p_I           : single-qubit identity (no-error) probability.
        c_Z           : bias parameter.  p_Z = p_X ** c_Z.
                        c_Z = 1  -> symmetric channel  (p_X = p_Y = p_Z).
                        c_Z < 1  -> Z errors dominate.
                        c_Z > 1  -> X / Y errors dominate.

    Returns:
        (probs, p_X)  where probs[i] is the probability of error_strings[i]
        under the chosen channel. Probabilities are not normalised; this
        matches the paper, which normalises by max(p_mu) -- see
        `paper_lambda_weights` below.
    """
    # Single-qubit error rate per Pauli channel:
    #   p_I + p_X + p_Y + p_Z = 1   (symmetric)
    # For the asymmetric case the paper keeps p_I fixed and solves for p_X
    # from the constraint  p_I + 2*p_X + p_X**c_Z = 1   (since p_X = p_Y).
    def _solve_p_X(p_I, c_Z, tol=1e-10):
        # bisection on (0, (1 - p_I) / 2]
        lo, hi = 0.0, (1.0 - p_I) / 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lhs = p_I + 2 * mid + (mid ** c_Z if mid > 0 else 0.0)
            if lhs > 1.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    p_X = _solve_p_X(p_I, c_Z)
    p_Y = p_X
    p_Z = p_X ** c_Z if p_X > 0 else 0.0

    probs = np.empty(len(error_strings), dtype=np.float64)
    for i, s in enumerate(error_strings):
        # multiplicative probability across qubits
        p = 1.0
        for ch in s:
            if   ch == 'I': p *= p_I
            elif ch == 'X': p *= p_X
            elif ch == 'Y': p *= p_Y
            elif ch == 'Z': p *= p_Z
        probs[i] = p
    return probs, p_X


def paper_lambda_weights(error_strings, n, p_I=0.9, c_Z=1.0):
    """
    Reward weights as used in the paper (Eq. 14):
        lambda_mu = p_mu / max(p_mu)
    Normalising by the max keeps the reward magnitudes stable across
    different c_Z values.
    """
    probs, _ = depolarizing_error_probabilities(error_strings, n, p_I, c_Z)
    m = probs.max()
    if m == 0:
        return np.ones_like(probs)
    return probs / m


# ====================================================================
# Main env
# ====================================================================

class QECEnv:
    """
    Gym-style environment for code + encoder discovery.

    The constructor builds a discrete action list once based on
    `gate_set` and `connectivity`. After that the agent calls reset()
    and step(action_idx) in the usual RL loop.
    """

    def __init__(
        self,
        n,                          # physical qubits
        k,                          # logical qubits
        target_distance=3,          # detect all errors of weight < d
        gate_set=('H', 'CNOT'),
        connectivity='directed_all',
        max_steps=20,
        # ---- reward ----
        reward_mode='symmetric',    # 'symmetric' | 'asymmetric' | 'noise_aware'
        p_I=0.9,                    # used by asymmetric / noise_aware
        c_Z=1.0,                    # fixed bias for 'asymmetric',
                                    # sampled per-episode for 'noise_aware'
        c_Z_range=(0.5, 2.0),       # noise_aware: range of c_Z values
        success_bonus=10.0,
        fail_penalty=-10.0,
        # ---- structural restriction (CSS mode, Section V) ----
        css_mode=False,
        # ---- KL exactness (Rule 5) ----
        exact_kl=True,
        # ---- error set: overridable ----
        error_strings=None,
    ):
        self.n = n
        self.k = k
        self.d = target_distance
        self.max_steps = max_steps
        self.success_bonus = success_bonus
        self.fail_penalty = fail_penalty
        self.exact_kl = exact_kl
        self.css_mode = css_mode
        self.reward_mode = reward_mode
        self.p_I = p_I
        self.c_Z_fixed = c_Z
        self.c_Z_range = c_Z_range
        self.rng = np.random.default_rng(0)

        # --- action list -------------------------------------------------
        self.actions, self.gate_kinds = self._build_action_list(
            gate_set, connectivity)
        self.num_actions = len(self.actions)
        # for CSS mode: pre-compute which action indices are H vs CNOT
        self.h_action_idxs = np.array(
            [i for i, k_ in enumerate(self.gate_kinds) if k_ == 'H'],
            dtype=np.int64)
        self.cnot_action_idxs = np.array(
            [i for i, k_ in enumerate(self.gate_kinds) if k_ == 'CNOT'],
            dtype=np.int64)

        # --- error set ---------------------------------------------------
        if error_strings is None:
            error_strings = all_pauli_strings_up_to_weight(n, target_distance - 1)
        self.error_strings = error_strings
        self.errors = pauli_strings_to_binary(error_strings, n)
        self._refresh_lambda_weights()    # also sets self.c_Z_current

        # --- observation dim --------------------------------------------
        self._G_dim = (n - k) * 2 * n
        if reward_mode == 'noise_aware':
            self.obs_dim = self._G_dim + 1   # append c_Z
        else:
            self.obs_dim = self._G_dim

        self.reset()

    # ----- one-time action-list construction --------------------------

    def _build_action_list(self, gate_set, connectivity):
        actions, kinds = [], []
        if 'H' in gate_set:
            for i in range(self.n):
                actions.append(('H', i)); kinds.append('H')
        if 'S' in gate_set:
            for i in range(self.n):
                actions.append(('S', i)); kinds.append('S')
        if 'CNOT' in gate_set or 'CX' in gate_set:
            pairs = self._connectivity_pairs(connectivity)
            for c, t in pairs:
                actions.append(('CNOT', c, t)); kinds.append('CNOT')
        if 'CZ' in gate_set:
            pairs = self._connectivity_pairs(connectivity)
            for c, t in pairs:
                if c < t:                     # CZ is symmetric, avoid dup
                    actions.append(('CZ', c, t)); kinds.append('CZ')
        return actions, kinds

    def _connectivity_pairs(self, connectivity):
        if connectivity == 'all':
            return [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        if connectivity == 'directed_all':
            return [(i, j) for i in range(self.n) for j in range(self.n) if i < j]
        return list(connectivity)

    # ----- reward weights (depend on c_Z in noise_aware mode) ---------

    def _refresh_lambda_weights(self):
        if self.reward_mode == 'symmetric':
            self.lambda_weight = np.ones(len(self.errors), dtype=np.float32)
            self.c_Z_current = 1.0
        elif self.reward_mode == 'asymmetric':
            self.lambda_weight = paper_lambda_weights(
                self.error_strings, self.n, self.p_I, self.c_Z_fixed
            ).astype(np.float32)
            self.c_Z_current = self.c_Z_fixed
        elif self.reward_mode == 'noise_aware':
            # sample a new bias every episode
            lo, hi = self.c_Z_range
            self.c_Z_current = float(self.rng.uniform(lo, hi))
            self.lambda_weight = paper_lambda_weights(
                self.error_strings, self.n, self.p_I, self.c_Z_current
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    # ----- gym-like interface ----------------------------------------

    def reset(self):
        self.tab = CliffordTableau(self.n, self.k)
        self.step_count = 0
        self.history = []
        self._css_seen_cnot = False   # for CSS-mode action masking
        if self.reward_mode == 'noise_aware':
            # only re-sample c_Z at the start of an episode
            self._refresh_lambda_weights()
        return self._obs()

    def _obs(self):
        flat_G = self.tab.G.flatten().astype(np.float32)
        if self.reward_mode == 'noise_aware':
            return np.concatenate([flat_G, [np.float32(self.c_Z_current)]])
        return flat_G

    def action_mask(self):
        """
        Return a 1-D bool array of length `num_actions`.  True means the
        action is currently legal. In standard mode every action is
        legal. In CSS mode, once any CNOT has been applied, no further
        Hadamard is allowed (Section V / Appendix I of the paper).
        """
        mask = np.ones(self.num_actions, dtype=bool)
        if self.css_mode and self._css_seen_cnot:
            mask[self.h_action_idxs] = False
        return mask

    def _apply_action(self, gate):
        op = gate[0]
        if   op == 'H':              self.tab.H(gate[1])
        elif op == 'S':              self.tab.S(gate[1])
        elif op in ('CNOT', 'CX'):   self.tab.CX(gate[1], gate[2])
        elif op == 'CZ':             self.tab.CZ(gate[1], gate[2])
        else:
            raise ValueError(f"Unhandled gate: {gate}")
        # track CSS prefix/suffix transition
        if op in ('CNOT', 'CX'):
            self._css_seen_cnot = True

    def step(self, action_idx):
        # In CSS mode an illegal action incurs a small penalty and the
        # state is unchanged. (Most agents will be told the mask anyway.)
        mask = self.action_mask()
        if not mask[action_idx]:
            self.step_count += 1
            terminated = self.step_count >= self.max_steps
            penalty = -1.0 + (self.fail_penalty if terminated else 0.0)
            return self._obs(), penalty, terminated, {
                'kl_sum': float('nan'),
                'success': False,
                'illegal': True,
                'c_Z': self.c_Z_current,
            }

        gate = self.actions[action_idx]
        self._apply_action(gate)
        self.history.append(gate)
        self.step_count += 1

        # weighted KL sum via the vectorized routine
        kl = weighted_kl_sum(
            self.errors, self.lambda_weight, self.tab.G, self.n,
            exact=self.exact_kl,
        )
        # are ALL errors detected? Use the unweighted mask for the check.
        mask_undet = kl_undetected_mask(
            self.errors, self.tab.G, self.n, exact=self.exact_kl)
        all_detected = not mask_undet.any()

        reward = -float(kl)
        terminated = False
        if all_detected:
            reward += self.success_bonus
            terminated = True
        elif self.step_count >= self.max_steps:
            reward += self.fail_penalty
            terminated = True

        return self._obs(), reward, terminated, {
            'kl_sum': float(kl),
            'success': bool(all_detected),
            'illegal': False,
            'c_Z': self.c_Z_current,
        }

    # ----- convenience ------------------------------------------------

    def render_history(self):
        out = []
        for g in self.history:
            if g[0] in ('CNOT', 'CX'):
                out.append(f"CNOT({g[1]} -> {g[2]})")
            elif g[0] == 'CZ':
                out.append(f"CZ({g[1]}, {g[2]})")
            else:
                out.append(f"{g[0]}({g[1]})")
        return out

    def seed(self, s):
        self.rng = np.random.default_rng(s)