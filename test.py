# =============================================================================
# shor_mwpm_conditional.py
# Shor Code + MWPM with Conditional Gates in Circuit
# =============================================================================

import numpy as np
import pymatching
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

# =============================================================================
# Shor Code Parity Check Matrices
# =============================================================================

H_X = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
], dtype=np.uint8)

H_Z = np.array([
    [1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8)

# MWPM decoders
matching_x = pymatching.Matching(H_X)
matching_z = pymatching.Matching(H_Z)

# =============================================================================
# Noise model
# =============================================================================

def build_noise_model(p):
    nm = NoiseModel()
    p2 = min(2*p, 0.99)
    # Single-qubit depolarizing
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ['h','x','id'])
    # Two-qubit depolarizing for CNOTs
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    return nm

# =============================================================================
# Circuit construction for one shot
# =============================================================================

def build_shor_mwpm_circuit(logical_state='0', p=0.01):
    # Registers
    data = QuantumRegister(9, 'data')
    anc_x = QuantumRegister(6, 'anc_x')
    anc_z = QuantumRegister(2, 'anc_z')
    syn_x = ClassicalRegister(6, 'sx')
    syn_z = ClassicalRegister(2, 'sz')
    log_c = ClassicalRegister(1, 'log')
    
    qc = QuantumCircuit(data, anc_x, anc_z, syn_x, syn_z, log_c)

    # Encode logical |0_L> or |1_L>
    if logical_state == '1':
        qc.x(data[0])

    # Shor encoding (repetition + Hadamard blocks)
    qc.cx(data[0], data[3]); qc.cx(data[0], data[6])
    qc.h(data[0]); qc.h(data[3]); qc.h(data[6])
    qc.cx(data[0], data[1]); qc.cx(data[0], data[2])
    qc.cx(data[3], data[4]); qc.cx(data[3], data[5])
    qc.cx(data[6], data[7]); qc.cx(data[6], data[8])
    
    # Barrier for clarity
    qc.barrier()

    # --- Syndrome extraction ---
    # X-type stabilizers (detect Z/Y errors)
    qc.cx(data[0], anc_x[0]); qc.cx(data[1], anc_x[0])
    qc.cx(data[1], anc_x[1]); qc.cx(data[2], anc_x[1])
    qc.cx(data[3], anc_x[2]); qc.cx(data[4], anc_x[2])
    qc.cx(data[4], anc_x[3]); qc.cx(data[5], anc_x[3])
    qc.cx(data[6], anc_x[4]); qc.cx(data[7], anc_x[4])
    qc.cx(data[7], anc_x[5]); qc.cx(data[8], anc_x[5])
    
    # Z-type stabilizers (detect X/Y errors)
    for idx, qubits in enumerate([[0,1,2,3,4,5],[3,4,5,6,7,8]]):
        qc.h(anc_z[idx])
        for q in qubits:
            qc.cx(anc_z[idx], data[q])
        qc.h(anc_z[idx])

    # Measure ancilla
    qc.measure(anc_x, syn_x)
    qc.measure(anc_z, syn_z)

    # --- Conditional MWPM corrections (simulated) ---
    # We simulate MWPM corrections classically and apply conditional X/Z gates
    # in the circuit via `qc.x(data[i]).c_if(...)` or `qc.z(data[i]).c_if(...)`
    
    # X corrections (from Z syndrome)
    sz_names = ['sz0','sz1']
    for shot_syndrome in range(4):  # 00,01,10,11
        sz = [int(b) for b in f'{shot_syndrome:02b}']
        z_corr = matching_z.decode(sz)
        if np.any(z_corr):
            for i, val in enumerate(z_corr):
                if val==1:
                    # Apply X correction conditioned on measured classical bits
                    qc.x(data[i]).c_if(syn_z, shot_syndrome)
    
    # Z corrections (from X syndrome)
    for shot_syndrome in range(64):  # 6 bits
        sx = [int(b) for b in f'{shot_syndrome:06b}']
        x_corr = matching_x.decode(sx)
        if np.any(x_corr):
            for i, val in enumerate(x_corr):
                if val==1:
                    qc.z(data[i]).c_if(syn_x, shot_syndrome)

    # Measure logical qubit (first data qubit)
    qc.measure(data[0], log_c[0])

    return qc

# =============================================================================
# Run shots and compute LER
# =============================================================================

def run_ler(per_values, shots=1000, logical_state='0'):
    sim = AerSimulator()
    ler_list = []
    print("\nPER      LER")
    print("-------------------")
    for p in per_values:
        qc = build_shor_mwpm_circuit(logical_state, p)
        result = sim.run(qc, shots=shots, noise_model=build_noise_model(p)).result()
        counts = result.get_counts()
        logical_errors = 0
        for outcome, count in counts.items():
            log_bit = outcome.replace(' ','')[-1]
            if log_bit != logical_state:
                logical_errors += count
        ler = logical_errors / shots
        print(f"{p:.4f}  {ler:.4f}")
        ler_list.append(ler)
    return ler_list

# =============================================================================
# Plot results
# =============================================================================

def plot_ler(per_values, ler_values):
    plt.figure(figsize=(7,5))
    plt.plot(per_values, ler_values, 'o-', label='Shor + MWPM Conditional')
    plt.plot(per_values, per_values, '--', color='gray', label='No QEC (LER=PER)')
    plt.xlabel("Physical Error Rate (PER)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    per_values = [0.0001,0.0005,0.001,0.005,0.01,0.03,0.05,0.1]
    ler_values = run_ler(per_values, shots=2000)
    plot_ler(per_values, ler_values)
    