# =============================================================================
# mwpm_repetition_clean.py
# ONE-SYSTEM MWPM Benchmark — 3-Qubit Repetition Code
# =============================================================================

import numpy as np
import pymatching
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# =============================================================================
# PARITY CHECK MATRIX
# =============================================================================

H = np.array([
    [1, 1, 0],
    [0, 1, 1],
], dtype=np.uint8)


# =============================================================================
# NOISE MODEL
# =============================================================================

def build_noise_model(p):
    nm = NoiseModel()

    p2 = min(2*p, 0.99)

    nm.add_all_qubit_quantum_error(
        depolarizing_error(p, 1),
        ['x', 'h', 'id']
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p2, 2),
        ['cx']
    )
    return nm


# =============================================================================
# MWPM DECODER
# =============================================================================

class MWPMDecoder:
    def __init__(self):
        self.matching = pymatching.Matching(H)

    def decode(self, syndrome):
        s = np.array(syndrome, dtype=np.uint8)

        if np.sum(s) == 0:
            return -1

        correction = self.matching.decode(s)
        idx = np.where(correction == 1)[0]

        return int(idx[0]) if len(idx) > 0 else -1


# =============================================================================
# BUILD FULL ONE-SYSTEM CIRCUIT
# =============================================================================

def build_full_circuit(logical_state='0', p=0.01):
    """
    ONE circuit:
    encode → noise → syndrome → correction → decode → measure
    """

    data = QuantumRegister(3, 'data')
    anc  = QuantumRegister(2, 'anc')

    syn  = ClassicalRegister(2, 'syn')
    log  = ClassicalRegister(1, 'log')

    qc = QuantumCircuit(data, anc, syn, log)

    # ── Encode logical qubit ─────────────────────────────────────
    if logical_state == '1':
        qc.x(data[0])

    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])

    # ── Syndrome extraction ──────────────────────────────────────
    qc.cx(data[0], anc[0])
    qc.cx(data[1], anc[0])

    qc.cx(data[1], anc[1])
    qc.cx(data[2], anc[1])

    qc.measure(anc[0], syn[0])
    qc.measure(anc[1], syn[1])

    # ── Classical correction (dynamic circuit) ───────────────────
    # If syndrome == 10 → correct q0
    # ── Conditional correction (dynamic circuit) ─────────────────

    with qc.if_test((syn, 1)):   # syndrome = 01 → q0 error
      qc.x(data[0])

    with qc.if_test((syn, 3)):   # syndrome = 11 → q1 error
      qc.x(data[1])

    with qc.if_test((syn, 2)):   # syndrome = 10 → q2 error
      qc.x(data[2])

    # ── Decode ──────────────────────────────────────────────────
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[1])

    # ── Measure logical qubit ───────────────────────────────────
    qc.measure(data[0], log[0])

    return qc


# =============================================================================
# RUN ONE SHOT
# =============================================================================

def run_shots(p, shots=1000, logical_state='0'):
    qc = build_full_circuit(logical_state, p)

    sim = AerSimulator(noise_model=build_noise_model(p))

    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    failures = 0

    for outcome, count in counts.items():
        # format: log syn1 syn0
        bits = outcome.replace(' ', '')

        logical_bit = bits[-1]  # last bit = logical

        if logical_bit != logical_state:
            failures += count

    return failures / shots


# =============================================================================
# LER vs PER
# =============================================================================

def run_ler_benchmark(per_values, shots=2000):
    results = []

    print("\nPER    LER")
    print("----------------")

    for p in per_values:
        ler = run_shots(p, shots)

        print(f"{p:.3f}  {ler:.5f}")

        results.append(ler)

    return results


# =============================================================================
# MAIN
# =============================================================================



# ── PLOTTING FUNCTION ─────────────────────────────────────────────
def plot_ler_vs_per(per_values, ler_values):
    plt.figure(figsize=(7,5))
    plt.plot(per_values, ler_values, 'o-', label='3-qubit Repetition + MWPM')
    plt.plot(per_values, per_values, '--', color='gray', label='No QEC (LER=PER)')
    plt.xlabel("Physical Error Rate (PER)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("LER vs PER — 3-Qubit Repetition Code")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()


# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    per_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

    results = run_ler_benchmark(per_values, shots=3000)
    plot_ler_vs_per(per_values,results)