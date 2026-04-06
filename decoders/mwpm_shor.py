
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


# =============================================================================
# PARITY CHECK MATRICES — unchanged, these are correct
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

matching_x = pymatching.Matching(H_X)
matching_z = pymatching.Matching(H_Z)


# =============================================================================
# ONE CIRCUIT PER SHOT
# =============================================================================
def run_one_shot(p, sim, basis='Z'):

    data  = QuantumRegister(9, 'data')
    anc_x = QuantumRegister(6, 'anc_x')
    anc_z = QuantumRegister(2, 'anc_z')
    syn_x = ClassicalRegister(6, 'sx')
    syn_z = ClassicalRegister(2, 'sz')
    c_d   = ClassicalRegister(9, 'out')

    qc = QuantumCircuit(data, anc_x, anc_z, syn_x, syn_z, c_d)

    # ── Encode ────────────────────────────────────────────────────────────────
    if basis == 'X':
        qc.h(data[0])          # encode |+_L> instead of |0_L>

    qc.cx(data[0], data[3])
    qc.cx(data[0], data[6])
    qc.h(data[0]); qc.h(data[3]); qc.h(data[6])
    qc.cx(data[0], data[1]); qc.cx(data[0], data[2])
    qc.cx(data[3], data[4]); qc.cx(data[3], data[5])
    qc.cx(data[6], data[7]); qc.cx(data[6], data[8])

    qc.barrier()

    # ── Bernoulli noise ───────────────────────────────────────────────────────
    for i in range(9):
        r = np.random.rand()
        if   r < p / 3:     qc.x(data[i])
        elif r < 2 * p / 3: qc.y(data[i])
        elif r < p:          qc.z(data[i])

    qc.barrier()

    # ── X syndrome — detects X and Y errors ───────────────────────────────────
    qc.cx(data[0], anc_x[0]); qc.cx(data[1], anc_x[0])
    qc.cx(data[1], anc_x[1]); qc.cx(data[2], anc_x[1])
    qc.cx(data[3], anc_x[2]); qc.cx(data[4], anc_x[2])
    qc.cx(data[4], anc_x[3]); qc.cx(data[5], anc_x[3])
    qc.cx(data[6], anc_x[4]); qc.cx(data[7], anc_x[4])
    qc.cx(data[7], anc_x[5]); qc.cx(data[8], anc_x[5])

    # ── Z syndrome — detects Z and Y errors (CORRECT) ──

# First stabilizer: X0 X1 X2 X3 X4 X5
    qc.h(anc_z[0])
    for i in [0,1,2,3,4,5]:
      qc.cx(anc_z[0], data[i])
    qc.h(anc_z[0])

# Second stabilizer: X3 X4 X5 X6 X7 X8
    qc.h(anc_z[1])
    for i in [3,4,5,6,7,8]:
      qc.cx(anc_z[1], data[i])
    qc.h(anc_z[1])
    qc.measure(anc_x, syn_x)
    qc.measure(anc_z, syn_z)

    # ── Rotate to X basis before measuring if needed ──────────────────────────
    if basis == 'X':
        for i in range(9):
            qc.h(data[i])      # H rotates Z basis → X basis

    qc.measure(data, c_d)
    
    result = sim.run(qc, shots=1).result()
    counts = result.get_counts()
    bitstr = list(counts.keys())[0]

    # reverse whole string then slice
    bitstr = bitstr.replace(' ', '')[::-1]

    data_bits = np.array([int(b) for b in bitstr[:9]])
    sz        = np.array([int(b) for b in bitstr[9:11]])
    sx        = np.array([int(b) for b in bitstr[11:]])

    # bitstr = list(result.get_counts().keys())[0]

    # parts     = bitstr.split(' ')
    # data_bits = np.array([int(b) for b in reversed(parts[0])], dtype=np.uint8)
    # sz        = np.array([int(b) for b in reversed(parts[1])], dtype=np.uint8)
    # sx        = np.array([int(b) for b in reversed(parts[2])], dtype=np.uint8)

    return data_bits, sx, sz

def compute_ler(per_values, shots=2000):
    sim      = AerSimulator()
    ler_list = []
    ci_list  = []

    print(f'\n{"─"*55}')
    print(f'  Shor Code + MWPM — LER vs PER (fixed)')
    print(f'  shots per point: {shots}')
    print(f'{"─"*55}')
    print(f'  {"PER":>8}  {"LER":>10}  {"±95%CI":>10}  {"LER/PER":>8}')
    print(f'  {"─"*51}')

    for p in per_values:
        logical_errors = 0

        for _ in range(shots):

            # ============================================================
            # 1) X-error correction test (Z basis)
            # ============================================================
            data_z, sx_z, sz_z = run_one_shot(p, sim, basis='Z')

            # MWPM decode bit-flip
            x_corr      = matching_x.decode(sx_z)
            corrected_z = (data_z + x_corr) % 2

            # Logical |0_L⟩ → expect ALL zeros after correction
            # logical_x_error = int(np.any(corrected_z != 0))
            def majority(block):
                return int(np.sum(block) >= 2)

            b0 = majority(corrected_z[0:3])
            b1 = majority(corrected_z[3:6])
            b2 = majority(corrected_z[6:9])

            logical_bit = majority([b0, b1, b2])

# expected logical value = 0 (since you encode |0_L⟩)
            logical_x_error = int(logical_bit != 0)


            # ============================================================
            # 2) Z-error correction test (X basis)
            # ============================================================
            data_x, sx_x, sz_x = run_one_shot(p, sim, basis='X')

            # MWPM decode phase-flip
            z_corr      = matching_z.decode(sz_x)
            corrected_x = (data_x + z_corr) % 2

            # Logical |+_L⟩ → expect ALL zeros in X-basis measurement
            # logical_z_error = int(np.any(corrected_x != 0))
            b0 = majority(corrected_x[0:3])
            b1 = majority(corrected_x[3:6])
            b2 = majority(corrected_x[6:9])

            logical_bit = majority([b0, b1, b2])

# expected logical value = 0 (since |+⟩ → 0 in X basis)
            logical_z_error = int(logical_bit != 0)

            # ============================================================
            # Combine
            # ============================================================
            if logical_x_error or logical_z_error:
                logical_errors += 1


        ler = logical_errors / shots
        ci  = 1.96 * np.sqrt(ler * (1 - ler) / shots)

        ler_list.append(ler)
        ci_list.append(ci)

        ratio  = ler / p if p > 0 else float('nan')
        status = 'QEC helps' if ratio < 1.0 else 'above threshold'

        print(f'  {p:>8.4f}  {ler:>10.4f}  {ci:>10.4f}  '
              f'{ratio:>8.3f}  {status}')

    print(f'{"─"*55}')
    return np.array(ler_list), np.array(ci_list)


def plot_results(per_values, ler_values, ci_values, shots):
    per_arr = np.array(per_values)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Shor Code + MWPM — LER vs PER  |  {shots} shots per point',
        fontsize=12, fontweight='bold'
    )

    for ax, scale in zip(axes, ['log', 'linear']):
        ax.fill_between(per_arr,
                        np.maximum(ler_values - ci_values, 1e-8),
                        ler_values + ci_values,
                        alpha=0.15, color='#378ADD')
        ax.plot(per_arr, ler_values, 'o-',
                color='#378ADD', lw=2, ms=7, label='Shor code + MWPM')
        ax.plot(per_arr, per_arr, '--',
                color='gray', lw=1.5, alpha=0.7, label='No correction')

        if scale == 'log':
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title('Log-Log')
        else:
            ax.set_title('Linear')
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

        ax.set_xlabel('PER'); ax.set_ylabel('LER')
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('per_vs_ler_shor_fixed.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def diagnose(p, sim, n=20):
    print(f"\nDiagnostic at p={p}")
    print(f"{'shot':>5}  {'sx':>10}  {'sz':>6}  {'data_bits':>20}  "
          f"{'x_corr':>20}  {'corrected':>20}  {'parity':>8}")

    for i in range(n):
        data_bits, sx, sz = run_one_shot(p, sim, basis='Z')
        x_corr            = matching_x.decode(sx)
        corrected         = (data_bits + x_corr) % 2
        parity            = int(np.sum(corrected) % 2)

        print(f"{i:>5}  "
              f"{''.join(map(str,sx)):>10}  "
              f"{''.join(map(str,sz)):>6}  "
              f"{''.join(map(str,data_bits)):>20}  "
              f"{''.join(map(str,x_corr)):>20}  "
              f"{''.join(map(str,corrected)):>20}  "
              f"{parity:>8}")

diagnose(0.001, AerSimulator())  # very low noise — should almost always succeed
if __name__ == '__main__':
    per_values = [0.0001,0.0005,0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.10]
    shots      = 1000

    ler_values, ci_values = compute_ler(per_values, shots)

    plot_results(per_values, ler_values, ci_values, shots)  
