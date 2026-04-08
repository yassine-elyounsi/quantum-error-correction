# =============================================================================
# benchmark_shor.py
# Fair benchmark: PPO vs MWPM on identical Qiskit circuits
#
# Noise model : Bernoulli depolarising — X/Y/Z each with prob p/3 per qubit
# Syndrome    : extracted from the same Qiskit circuit instance
# Evaluation  : majority-vote logical error (same logic as mwpm_shor.py)
#
# Both decoders receive the SAME syndrome from the SAME shot.
# PPO decodes the 8-bit syndrome vector.
# MWPM decodes sx (6 bits) and sz (2 bits) separately.
#
# Usage:
#   python benchmark_shor.py
# =============================================================================

import numpy as np
import pymatching
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from stable_baselines3 import PPO

# =============================================================================
# CONFIG — edit these
# =============================================================================

MODEL_PATH  = "ppo_shor_wandb"          # path to your saved PPO model (no .zip)
PER_VALUES  = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.10]
SHOTS       = 1000                       # shots per p value
SAVE_FIG    = "benchmark_ppo_vs_mwpm.png"

# =============================================================================
# PARITY CHECK MATRICES
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
# SHARED CIRCUIT — one shot, returns data bits + syndrome for BOTH decoders
# =============================================================================

def run_shared_shot(p, sim, basis='Z'):
    """
    Build and run one Shor code circuit with Bernoulli depolarising noise.
    Returns data_bits, sx (6 bits), sz (2 bits), and the 8-bit syndrome
    vector that the PPO agent expects.

    The syndrome vector layout matches ShorCodeEnv:
        obs[0:6]  = sx  (X-stabiliser syndrome)
        obs[6:8]  = sz  (Z-stabiliser syndrome)

    Parameters
    ----------
    p     : float   physical error rate
    sim   : AerSimulator
    basis : 'Z' (encode |0_L>) or 'X' (encode |+_L>)

    Returns
    -------
    data_bits : np.ndarray  shape (9,)  uint8
    sx        : np.ndarray  shape (6,)  uint8
    sz        : np.ndarray  shape (2,)  uint8
    obs       : np.ndarray  shape (8,)  int8   ← PPO input
    """
    data  = QuantumRegister(9, 'data')
    anc_x = QuantumRegister(6, 'anc_x')
    anc_z = QuantumRegister(2, 'anc_z')
    syn_x = ClassicalRegister(6, 'sx')
    syn_z = ClassicalRegister(2, 'sz')
    c_d   = ClassicalRegister(9, 'out')

    qc = QuantumCircuit(data, anc_x, anc_z, syn_x, syn_z, c_d)

    # ── Encode ────────────────────────────────────────────────────────────────
    if basis == 'X':
        qc.h(data[0])

    qc.cx(data[0], data[3])
    qc.cx(data[0], data[6])
    qc.h(data[0]); qc.h(data[3]); qc.h(data[6])
    qc.cx(data[0], data[1]); qc.cx(data[0], data[2])
    qc.cx(data[3], data[4]); qc.cx(data[3], data[5])
    qc.cx(data[6], data[7]); qc.cx(data[6], data[8])

    qc.barrier()

    # ── Bernoulli depolarising noise (same as mwpm_shor.py) ──────────────────
    for i in range(9):
        r = np.random.rand()
        if   r < p / 3:       qc.x(data[i])
        elif r < 2 * p / 3:   qc.y(data[i])
        elif r < p:            qc.z(data[i])

    qc.barrier()

    # ── X syndrome (detects X and Y errors) ───────────────────────────────────
    qc.cx(data[0], anc_x[0]); qc.cx(data[1], anc_x[0])
    qc.cx(data[1], anc_x[1]); qc.cx(data[2], anc_x[1])
    qc.cx(data[3], anc_x[2]); qc.cx(data[4], anc_x[2])
    qc.cx(data[4], anc_x[3]); qc.cx(data[5], anc_x[3])
    qc.cx(data[6], anc_x[4]); qc.cx(data[7], anc_x[4])
    qc.cx(data[7], anc_x[5]); qc.cx(data[8], anc_x[5])

    # ── Z syndrome (detects Z and Y errors) ───────────────────────────────────
    qc.h(anc_z[0])
    for i in [0, 1, 2, 3, 4, 5]:
        qc.cx(anc_z[0], data[i])
    qc.h(anc_z[0])

    qc.h(anc_z[1])
    for i in [3, 4, 5, 6, 7, 8]:
        qc.cx(anc_z[1], data[i])
    qc.h(anc_z[1])

    qc.measure(anc_x, syn_x)
    qc.measure(anc_z, syn_z)

    if basis == 'X':
        for i in range(9):
            qc.h(data[i])

    qc.measure(data, c_d)

    result  = sim.run(qc, shots=1).result()
    bitstr  = list(result.get_counts().keys())[0]
    bitstr  = bitstr.replace(' ', '')[::-1]

    data_bits = np.array([int(b) for b in bitstr[:9]],  dtype=np.uint8)
    sz        = np.array([int(b) for b in bitstr[9:11]], dtype=np.uint8)
    sx        = np.array([int(b) for b in bitstr[11:]], dtype=np.uint8)

    # 8-bit observation vector for PPO: [sx(6 bits) | sz(2 bits)]
    obs = np.concatenate([sx, sz]).astype(np.int8)

    return data_bits, sx, sz, obs


# =============================================================================
# LOGICAL ERROR CHECK — majority vote (same as mwpm_shor.py)
# =============================================================================

def majority(block):
    return int(np.sum(block) >= 2)


def has_logical_x_error(corrected):
    """True if majority-vote logical bit ≠ 0 after X correction."""
    b0 = majority(corrected[0:3])
    b1 = majority(corrected[3:6])
    b2 = majority(corrected[6:9])
    return majority([b0, b1, b2]) != 0


def has_logical_z_error(corrected):
    """True if majority-vote logical bit ≠ 0 after Z correction."""
    b0 = majority(corrected[0:3])
    b1 = majority(corrected[3:6])
    b2 = majority(corrected[6:9])
    return majority([b0, b1, b2]) != 0


# =============================================================================
# PPO DECODER — maps 8-bit obs → correction vector on 9 qubits
# =============================================================================

def ppo_correction_vectors(action):
    """
    Convert PPO action integer to (x_corr, z_corr) binary vectors of length 9.
    Action space matches ShorCodeEnv:
        0       → no correction
        1–9     → X on qubit i-1
        10–18   → Z on qubit i-10
        19–27   → Y on qubit i-19  (= X and Z)

    Returns
    -------
    x_corr : np.ndarray shape (9,) uint8  — X flips to apply
    z_corr : np.ndarray shape (9,) uint8  — Z flips to apply
    """
    x_corr = np.zeros(9, dtype=np.uint8)
    z_corr = np.zeros(9, dtype=np.uint8)

    if action == 0:
        pass
    elif 1 <= action <= 9:
        x_corr[action - 1] = 1
    elif 10 <= action <= 18:
        z_corr[action - 10] = 1
    elif 19 <= action <= 27:
        q = action - 19
        x_corr[q] = 1
        z_corr[q] = 1

    return x_corr, z_corr


# =============================================================================
# BENCHMARK LOOP
# =============================================================================

def run_benchmark(model, per_values, shots=1000):
    """
    For each p in per_values, run `shots` paired trials.
    Each trial:
      - runs ONE shared Qiskit circuit (Z basis + X basis)
      - feeds syndrome to MWPM → logical error check
      - feeds same syndrome to PPO → logical error check
    Returns LER arrays for both decoders + 95% CI.
    """
    sim = AerSimulator()

    ler_mwpm = []
    ler_ppo  = []
    ci_mwpm  = []
    ci_ppo   = []

    print(f'\n{"─"*72}')
    print(f'  Shor Code — PPO vs MWPM Benchmark')
    print(f'  Noise: Bernoulli depolarising  |  shots per point: {shots}')
    print(f'{"─"*72}')
    print(f'  {"PER":>8}  {"LER_MWPM":>10}  {"LER_PPO":>10}  '
          f'{"CI_MWPM":>9}  {"CI_PPO":>9}  {"Winner":>8}')
    print(f'  {"─"*68}')

    for p in per_values:
        err_mwpm = 0
        err_ppo  = 0

        for _ in range(shots):

            # ── Z-basis shot (tests X-error correction) ───────────────────────
            data_z, sx_z, sz_z, obs_z = run_shared_shot(p, sim, basis='Z')

            # MWPM on Z-basis shot
            x_corr_mwpm         = matching_x.decode(sx_z)
            corrected_z_mwpm    = (data_z + x_corr_mwpm) % 2
            mwpm_x_err          = has_logical_x_error(corrected_z_mwpm)

            # PPO on Z-basis shot
            action_z, _         = model.predict(obs_z, deterministic=True)
            action_z            = int(action_z)
            x_corr_ppo, _       = ppo_correction_vectors(action_z)
            corrected_z_ppo     = (data_z + x_corr_ppo) % 2
            ppo_x_err           = has_logical_x_error(corrected_z_ppo)

            # ── X-basis shot (tests Z-error correction) ───────────────────────
            data_x, sx_x, sz_x, obs_x = run_shared_shot(p, sim, basis='X')

            # MWPM on X-basis shot
            z_corr_mwpm         = matching_z.decode(sz_x)
            corrected_x_mwpm    = (data_x + z_corr_mwpm) % 2
            mwpm_z_err          = has_logical_z_error(corrected_x_mwpm)

            # PPO on X-basis shot
            action_x, _         = model.predict(obs_x, deterministic=True)
            action_x            = int(action_x)
            _, z_corr_ppo       = ppo_correction_vectors(action_x)
            corrected_x_ppo     = (data_x + z_corr_ppo) % 2
            ppo_z_err           = has_logical_z_error(corrected_x_ppo)

            # ── Combine X and Z errors ────────────────────────────────────────
            if mwpm_x_err or mwpm_z_err:
                err_mwpm += 1
            if ppo_x_err or ppo_z_err:
                err_ppo += 1

        ler_m = err_mwpm / shots
        ler_p = err_ppo  / shots
        ci_m  = 1.96 * np.sqrt(ler_m * (1 - ler_m) / shots)
        ci_p  = 1.96 * np.sqrt(ler_p * (1 - ler_p) / shots)

        ler_mwpm.append(ler_m)
        ler_ppo.append(ler_p)
        ci_mwpm.append(ci_m)
        ci_ppo.append(ci_p)

        winner = 'MWPM' if ler_m < ler_p else ('PPO' if ler_p < ler_m else 'TIE')
        print(f'  {p:>8.4f}  {ler_m:>10.4f}  {ler_p:>10.4f}  '
              f'{ci_m:>9.4f}  {ci_p:>9.4f}  {winner:>8}')

    print(f'{"─"*72}')
    return (np.array(ler_mwpm), np.array(ci_mwpm),
            np.array(ler_ppo),  np.array(ci_ppo))


# =============================================================================
# PLOT
# =============================================================================

def plot_benchmark(per_values, ler_mwpm, ci_mwpm, ler_ppo, ci_ppo, shots):
    per_arr = np.array(per_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Shor Code — PPO vs MWPM  |  Bernoulli depolarising noise  '
        f'|  {shots} shots per point',
        fontsize=12, fontweight='bold'
    )

    MWPM_COLOR = '#378ADD'
    PPO_COLOR  = '#E8602C'
    BASE_COLOR = 'gray'

    for ax, scale in zip(axes, ['log', 'linear']):

        # confidence bands
        ax.fill_between(per_arr,
                        np.maximum(ler_mwpm - ci_mwpm, 1e-8),
                        ler_mwpm + ci_mwpm,
                        alpha=0.15, color=MWPM_COLOR)
        ax.fill_between(per_arr,
                        np.maximum(ler_ppo - ci_ppo, 1e-8),
                        ler_ppo + ci_ppo,
                        alpha=0.15, color=PPO_COLOR)

        # main curves
        ax.plot(per_arr, ler_mwpm, 'o-',
                color=MWPM_COLOR, lw=2, ms=7, label='MWPM')
        ax.plot(per_arr, ler_ppo, 's-',
                color=PPO_COLOR,  lw=2, ms=7, label='PPO (RL)')
        ax.plot(per_arr, per_arr, '--',
                color=BASE_COLOR, lw=1.5, alpha=0.7, label='No correction')

        if scale == 'log':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Log–Log Scale')
        else:
            ax.set_title('Linear Scale')
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f'{x*100:.2f}%'))
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

        ax.set_xlabel('Physical Error Rate (PER)')
        ax.set_ylabel('Logical Error Rate (LER)')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_FIG, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved → {SAVE_FIG}')
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print('Loading PPO model...')
    model = PPO.load(MODEL_PATH)
    print(f'  Model loaded from {MODEL_PATH}')

    ler_mwpm, ci_mwpm, ler_ppo, ci_ppo = run_benchmark(
        model, PER_VALUES, shots=SHOTS
    )

    plot_benchmark(PER_VALUES, ler_mwpm, ci_mwpm, ler_ppo, ci_ppo, SHOTS)