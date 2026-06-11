"""
Vectorized Clifford simulator using the binary symplectic formalism.

Implements the formalism described in Section III.D and Appendix B of
Olle, Zen, Puviani, Marquardt (2024).

Pauli string encoding (ignoring global phases):
    A Pauli string on n qubits is a length-2n binary vector
    (x_1, ..., x_n, z_1, ..., z_n) where
        I -> (0,0)   X -> (1,0)   Y -> (1,1)   Z -> (0,1)

A stabilizer code is an (n-k) x 2n binary check matrix; each row is
one stabilizer generator g_i.

Two Pauli strings P1, P2 commute iff   P1 . Omega . P2^T = 0 (mod 2),
with Omega = [[0, I_n], [I_n, 0]].

This module provides:
  * single-circuit simulation     (CliffordTableau)
  * batched simulation            (BatchedCliffordTableau)
  * exact KL detection            (is_detected, kl_undetected_mask)
  * vectorized weighted reward    (weighted_kl_sum)
  * circuit replay / validation   (replay_circuit, validate_circuit)
"""

import numpy as np
from itertools import combinations, product


# ====================================================================
# 1. Pauli string utilities
# ====================================================================

def pauli_string_to_binary(s, n):
    """'IXZIY' (length n)  ->  length-2n binary vector (x|z)."""
    assert len(s) == n, f"Expected length-{n} string, got '{s}'"
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    for i, ch in enumerate(s):
        if   ch == 'I': pass
        elif ch == 'X': x[i] = 1
        elif ch == 'Y': x[i] = 1; z[i] = 1
        elif ch == 'Z': z[i] = 1
        else: raise ValueError(f"Unknown Pauli char: {ch}")
    return np.concatenate([x, z])


def binary_to_pauli_string(v, n):
    """Inverse of `pauli_string_to_binary`."""
    chars = []
    for xi, zi in zip(v[:n], v[n:]):
        if   xi == 0 and zi == 0: chars.append('I')
        elif xi == 1 and zi == 0: chars.append('X')
        elif xi == 1 and zi == 1: chars.append('Y')
        elif xi == 0 and zi == 1: chars.append('Z')
    return ''.join(chars)


def pauli_strings_to_binary(strings, n):
    """Batch version. Returns a (num_errors, 2n) uint8 matrix."""
    return np.stack([pauli_string_to_binary(s, n) for s in strings]).astype(np.uint8)


# ====================================================================
# 2. Symplectic form, commutation, weight
# ====================================================================

def symplectic_form(n):
    """Build Omega = [[0, I_n], [I_n, 0]]   (size 2n x 2n)."""
    Omega = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    Omega[:n, n:] = np.eye(n, dtype=np.uint8)
    Omega[n:, :n] = np.eye(n, dtype=np.uint8)
    return Omega


def commute(P1, P2, n):
    """1 if P1 anticommutes with P2, else 0."""
    return int((P1 @ symplectic_form(n) @ P2) % 2)


def weight(P, n):
    """Pauli weight: number of non-identity tensor factors."""
    return int(np.sum((P[:n] | P[n:]) > 0))


def weight_batch(P_batch, n):
    """Weights of a batch of Paulis. P_batch: (B, 2n) -> (B,)."""
    x = P_batch[:, :n]
    z = P_batch[:, n:]
    return ((x | z) > 0).sum(axis=1)


# ====================================================================
# 3. Clifford gates as binary matrices
# ====================================================================
#
# A Clifford gate U acts on the check matrix G as  G' = G @ M (mod 2)
# for a fixed 2n x 2n binary matrix M that depends on the gate.

def hadamard_matrix(i, n):
    """H_i  : swap columns i and i+n  (X_i <-> Z_i)."""
    M = np.eye(2 * n, dtype=np.uint8)
    M[i, i] = 0;          M[i, i + n] = 1
    M[i + n, i + n] = 0;  M[i + n, i] = 1
    return M


def phase_matrix(i, n):
    """S_i  : X_i -> Y_i,  Z_i -> Z_i   (add x-col into z-col)."""
    M = np.eye(2 * n, dtype=np.uint8)
    M[i, i + n] = 1
    return M


def cnot_matrix(c, t, n):
    """CNOT(c -> t)."""
    M = np.eye(2 * n, dtype=np.uint8)
    M[c, t] = 1            # x_c -> x_t
    M[t + n, c + n] = 1    # z_t -> z_c
    return M


def cz_matrix(a, b, n):
    """CZ(a, b) = H_b . CNOT(a,b) . H_b. Symmetric in a, b."""
    return (hadamard_matrix(b, n) @ cnot_matrix(a, b, n) @ hadamard_matrix(b, n)) % 2


def swap_matrix(a, b, n):
    """SWAP(a, b): exchange columns a<->b and (a+n)<->(b+n)."""
    M = np.eye(2 * n, dtype=np.uint8)
    M[a, a] = 0; M[b, b] = 0
    M[a, b] = 1; M[b, a] = 1
    M[a + n, a + n] = 0; M[b + n, b + n] = 0
    M[a + n, b + n] = 1; M[b + n, a + n] = 1
    return M


# X, Y, Z gates are Clifford but only change global signs at the
# stabilizer level (which we ignore). Kept for gate-set flexibility.
def x_matrix(i, n): return np.eye(2 * n, dtype=np.uint8)
def y_matrix(i, n): return np.eye(2 * n, dtype=np.uint8)
def z_matrix(i, n): return np.eye(2 * n, dtype=np.uint8)


# ====================================================================
# 4. Single-circuit tableau
# ====================================================================

class CliffordTableau:
    """
    The stabilizer state of a partially-encoded code.

    Initialization (Eq. 9 of the paper):
        first k qubits   = unencoded logical state |psi>
        qubits k..n-1    = |0>, stabilized by Z_{k+1}, ..., Z_n
    """

    def __init__(self, n, k):
        self.n = n
        self.k = k
        G = np.zeros((n - k, 2 * n), dtype=np.uint8)
        for row, q in enumerate(range(k, n)):
            G[row, q + n] = 1
        self.G = G

    # --- gate application -------------------------------------------
    def apply(self, M):
        self.G = (self.G @ M) % 2

    def H(self, i):       self.apply(hadamard_matrix(i, self.n))
    def S(self, i):       self.apply(phase_matrix(i, self.n))
    def CX(self, c, t):   self.apply(cnot_matrix(c, t, self.n))
    def CZ(self, a, b):   self.apply(cz_matrix(a, b, self.n))
    def SWAP(self, a, b): self.apply(swap_matrix(a, b, self.n))

    # --- utilities --------------------------------------------------
    def copy(self):
        new = CliffordTableau(self.n, self.k)
        new.G = self.G.copy()
        return new

    def generators_str(self):
        return [binary_to_pauli_string(row, self.n) for row in self.G]

    def __repr__(self):
        return f"CliffordTableau(n={self.n}, k={self.k}, gens={self.generators_str()})"


# ====================================================================
# 4b. Batched tableau -- many circuits in parallel
# ====================================================================

class BatchedCliffordTableau:
    """
    Holds B independent check matrices in a single (B, n-k, 2n) array.
    Lets the agent run a batch of episodes in parallel -- the paper's
    main speed trick (Section III.D and Appendix B).
    """

    def __init__(self, n, k, batch_size):
        self.n = n
        self.k = k
        self.B = batch_size
        G0 = np.zeros((n - k, 2 * n), dtype=np.uint8)
        for row, q in enumerate(range(k, n)):
            G0[row, q + n] = 1
        self.G = np.broadcast_to(G0, (batch_size, n - k, 2 * n)).copy()

    def apply(self, M, env_mask=None):
        """Apply the SAME gate matrix M to every (or masked) circuit."""
        if env_mask is None:
            self.G = (self.G @ M) % 2
        else:
            self.G[env_mask] = (self.G[env_mask] @ M) % 2

    def apply_per_env(self, gate_matrices):
        """Different gate per circuit. gate_matrices: (B, 2n, 2n)."""
        self.G = np.einsum('bij,bjk->bik', self.G, gate_matrices) % 2

    def reset_envs(self, env_mask):
        G0 = np.zeros((self.n - self.k, 2 * self.n), dtype=np.uint8)
        for row, q in enumerate(range(self.k, self.n)):
            G0[row, q + self.n] = 1
        self.G[env_mask] = G0


# ====================================================================
# 5. Knill-Laflamme detection -- exact and vectorized
# ====================================================================

def _row_reduce_gf2(M):
    """
    Reduced row echelon form of binary matrix M over GF(2).
    Returns (R, pivot_cols). M is not modified.
    """
    R = M.copy().astype(np.uint8)
    rows, cols = R.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.where(R[r:, c] == 1)[0]
        if len(nz) == 0:
            continue
        pr = r + int(nz[0])
        if pr != r:
            R[[r, pr]] = R[[pr, r]]
        for rr in range(rows):
            if rr != r and R[rr, c] == 1:
                R[rr] = (R[rr] + R[r]) % 2
        pivots.append(c)
        r += 1
    return R, pivots


def in_rowspace_gf2(G, v):
    """True iff v lies in the row space of G over GF(2)."""
    G = G.astype(np.uint8)
    v = v.astype(np.uint8)
    _, pivots_G = _row_reduce_gf2(G)
    aug = np.vstack([G, v[None, :]])
    _, pivots_aug = _row_reduce_gf2(aug)
    return len(pivots_aug) == len(pivots_G)


def is_detected(error_P, G, n, stabilizer_subgroup=None, exact=True):
    """
    Knill-Laflamme detection test, Eqs. (4)-(5) of the paper.

    Args:
        error_P : (2n,) binary error vector
        G       : (n-k, 2n) check matrix
        n       : number of qubits
        stabilizer_subgroup : optional, used only if exact=False
        exact   : if True, use Gaussian elimination for Rule 5
                  (correct for ALL codes including degenerate ones).
                  if False, use the legacy 'softness' subgroup membership
                  (faster but heuristic -- can miss degenerate detections).
    """
    # Rule 4 -- anticommute with any generator
    Omega = symplectic_form(n)
    if np.any((G @ Omega @ error_P) % 2):
        return True

    # Rule 5 -- error in S_C ?
    if exact:
        return in_rowspace_gf2(G, error_P)
    if stabilizer_subgroup is None:
        return False
    for s in stabilizer_subgroup:
        if np.array_equal(s, error_P):
            return True
    return False


def build_stabilizer_subgroup(G, softness=2):
    """
    Legacy heuristic kept for backward compatibility.
    Enumerates products of up to `softness` generators.
    Prefer `exact=True` in `is_detected` for correctness.
    """
    n_gen = G.shape[0]
    subgroup = [np.zeros(G.shape[1], dtype=np.uint8)]
    if softness >= 1:
        for i in range(n_gen):
            subgroup.append(G[i].copy())
    if softness >= 2:
        for i in range(n_gen):
            for j in range(i + 1, n_gen):
                subgroup.append((G[i] + G[j]) % 2)
    if softness >= 3:
        for i in range(n_gen):
            for j in range(i + 1, n_gen):
                for kk in range(j + 1, n_gen):
                    subgroup.append((G[i] + G[j] + G[kk]) % 2)
    return subgroup


def kl_undetected_mask(E_batch, G, n, exact=True):
    """
    Vectorized KL test for a batch of errors against one check matrix G.

    Returns:
        undetected_mask : (num_errors,) bool array
                          True  = error is NOT detected (bad)
                          False = error is detected     (good)
    """
    Omega = symplectic_form(n)
    syndromes = (E_batch @ Omega @ G.T) % 2          # (num_errors, n-k)
    detected_by_rule4 = syndromes.any(axis=1)

    if not exact:
        return ~detected_by_rule4

    undetected_so_far = ~detected_by_rule4
    if not undetected_so_far.any():
        return np.zeros(E_batch.shape[0], dtype=bool)

    _, pivots_G = _row_reduce_gf2(G)
    rank_G = len(pivots_G)
    final_undetected = np.zeros(E_batch.shape[0], dtype=bool)
    for i in np.where(undetected_so_far)[0]:
        aug = np.vstack([G, E_batch[i][None, :]])
        _, p = _row_reduce_gf2(aug)
        if len(p) != rank_G:
            final_undetected[i] = True  # not in row space -> undetected
    return final_undetected


def weighted_kl_sum(E_batch, lambdas, G, n, exact=True):
    """
    The paper's reward signal (Eq. 10):
        weighted KL sum  =  sum_mu  lambda_mu * K_mu
        K_mu = 1 if E_mu undetected, else 0.

    The RL agent maximises  -weighted_kl_sum.
    """
    mask = kl_undetected_mask(E_batch, G, n, exact=exact)
    return float((lambdas * mask).sum())


# ====================================================================
# 6. Error set generation
# ====================================================================

def all_pauli_strings_up_to_weight(n, max_weight):
    """All non-identity Pauli strings on n qubits with weight <= max_weight."""
    errors = []
    for w in range(1, max_weight + 1):
        for positions in combinations(range(n), w):
            for paulis in product(['X', 'Y', 'Z'], repeat=w):
                s = ['I'] * n
                for pos, p in zip(positions, paulis):
                    s[pos] = p
                errors.append(''.join(s))
    return errors


def x_type_errors_up_to_weight(n, max_weight):
    """X-only errors -- the bit-flip channel."""
    errors = []
    for w in range(1, max_weight + 1):
        for positions in combinations(range(n), w):
            s = ['I'] * n
            for pos in positions: s[pos] = 'X'
            errors.append(''.join(s))
    return errors


def z_type_errors_up_to_weight(n, max_weight):
    """Z-only errors -- the phase-flip channel."""
    errors = []
    for w in range(1, max_weight + 1):
        for positions in combinations(range(n), w):
            s = ['I'] * n
            for pos in positions: s[pos] = 'Z'
            errors.append(''.join(s))
    return errors


# ====================================================================
# 7. Circuit replay & validation
# ====================================================================

def replay_circuit(n, k, gate_list):
    """
    Apply a list of gates to a fresh tableau and return it.

    Each gate is a tuple:
        ('H', i)            ('S', i)             ('X'|'Y'|'Z', i)
        ('CNOT', c, t)      ('CX', c, t)         ('CZ', a, b)
        ('SWAP', a, b)
    """
    tab = CliffordTableau(n, k)
    for gate in gate_list:
        op = gate[0]
        if   op == 'H':              tab.H(gate[1])
        elif op == 'S':              tab.S(gate[1])
        elif op in ('CNOT', 'CX'):   tab.CX(gate[1], gate[2])
        elif op == 'CZ':             tab.CZ(gate[1], gate[2])
        elif op == 'SWAP':           tab.SWAP(gate[1], gate[2])
        elif op in ('X', 'Y', 'Z'):  pass  # trivial at stabilizer level
        else: raise ValueError(f"Unknown gate: {gate}")
    return tab


def validate_circuit(n, k, gate_list, error_strings, exact=True, verbose=False):
    """
    Replay a circuit and report which errors are detected by the resulting code.

    Returns:
        {
          'generators'        : list[str],
          'num_detected'      : int,
          'num_total'         : int,
          'undetected_errors' : list[str],
          'success'           : bool
        }
    """
    tab = replay_circuit(n, k, gate_list)
    E_batch = pauli_strings_to_binary(error_strings, n)
    mask = kl_undetected_mask(E_batch, tab.G, n, exact=exact)
    undetected = [error_strings[i] for i in np.where(mask)[0]]
    result = {
        'generators': tab.generators_str(),
        'num_detected': int((~mask).sum()),
        'num_total': len(error_strings),
        'undetected_errors': undetected,
        'success': bool(not mask.any()),
    }
    if verbose:
        print(f"  Generators : {result['generators']}")
        print(f"  Detected   : {result['num_detected']}/{result['num_total']}")
        if undetected:
            preview = undetected[:10]
            more = f' (+{len(undetected)-10} more)' if len(undetected) > 10 else ''
            print(f"  Undetected : {preview}{more}")
    return result