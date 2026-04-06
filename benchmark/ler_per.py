# =============================================================================
# ler_vs_per_repetition.py
# LER vs PER Benchmark — 3-Qubit Repetition Code with MWPM Decoding
# =============================================================================
#
# Measures Logical Error Rate (LER) at each Physical Error Rate (PER)
# and produces a publication-quality plot.
#
# What this script does:
#   For each PER in [0.01, 0.03, 0.05, 0.07, 0.10]:
#     1. Generate n_shots noisy syndrome circuits
#     2. Decode each syndrome with MWPM
#     3. Verify correction recovers the logical qubit
#     4. LER = fraction of failed corrections
#
#   Then plots:
#     - LER vs PER for repetition code + MWPM
#     - Reference line LER = PER  (no QEC baseline)
#     - Reference line LER = PER² (ideal distance-3 suppression)
#
# Usage:
#   python benchmarks/ler_vs_per_repetition.py
#   python benchmarks/ler_vs_per_repetition.py --shots 5000
#   python benchmarks/ler_vs_per_repetition.py --shots 2000 --save results/ler.png
# =============================================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

from decoders.mwpm_repetition import MWPMRepetitionDecoder
from src.circuits.repetition_code import (
    run_syndrome_extraction,
    build_noise_model,
    prepare_logical_qubit,
)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


# =============================================================================
# BENCHMARK CORE
# =============================================================================

def run_single_shot(decoder, logical_state, per, noise_type, rng):
    """
    Run one full QEC cycle:
      inject random error → noisy syndrome → MWPM decode → correct → measure

    Parameters
    ----------
    decoder       : MWPMRepetitionDecoder
    logical_state : str    '0' or '1'
    per           : float  physical error rate
    noise_type    : str
    rng           : np.random.Generator

    Returns
    -------
    success : bool   True if logical qubit was recovered correctly
    meta    : dict   details about this shot
    """
    # Step 1 — randomly inject a deterministic error
    error_qubit = rng.choice([None, 0, 1, 2])

    # Step 2 — noisy syndrome extraction
    syndrome_int, syndrome_str, _ = run_syndrome_extraction(
        logical_state=logical_state,
        error_qubit=error_qubit,
        p=per,
        shots=1,
        noise_type=noise_type,
    )

    # Step 3 — MWPM decode
    result           = decoder.decode_from_str(syndrome_str)
    correction_qubit = result['qubit']

    # Step 4 — apply correction and measure logical qubit
    success = _apply_and_measure(
        logical_state=logical_state,
        error_qubit=error_qubit,
        correction_qubit=correction_qubit,
        per=per,
        noise_type=noise_type,
    )

    meta = {
        'error_qubit'      : error_qubit,
        'syndrome_str'     : syndrome_str,
        'syndrome_int'     : syndrome_int,
        'correction_qubit' : correction_qubit,
        'success'          : success,
    }

    return success, meta


def _apply_and_measure(logical_state, error_qubit, correction_qubit,
                        per, noise_type):
    """
    Build and run the correction + decode + measure circuit.

    Returns True if logical qubit measurement matches logical_state.
    """
    data    = QuantumRegister(3, 'data')
    log_reg = ClassicalRegister(1, 'logical')
    qc      = QuantumCircuit(data, log_reg)

    # Encode
    prepare_logical_qubit(qc, data[0], logical_state)
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])

    # Inject deterministic error
    if error_qubit is not None:
        qc.x(data[error_qubit])

    # Apply MWPM correction
    if correction_qubit >= 0:
        qc.x(data[correction_qubit])

    # Decode — reverse encoding
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[1])

    # Measure
    qc.measure(data[0], log_reg[0])

    # Run with noise
    if per > 0.0:
        nm  = build_noise_model(per, noise_type)
        sim = AerSimulator(noise_model=nm)
    else:
        sim = AerSimulator()

    counts  = sim.run(qc, shots=256).result().get_counts()
    outcome = max(counts, key=counts.get).strip()
    return outcome == logical_state


def benchmark(per_values, n_shots=3000, logical_state='0',
              noise_type='depolarising', verbose=True):
    """
    Run the full LER vs PER benchmark for the repetition code.

    Parameters
    ----------
    per_values    : list of float   physical error rates to evaluate
    n_shots       : int             number of shots per PER value
    logical_state : str             '0' or '1'
    noise_type    : str
    verbose       : bool

    Returns
    -------
    results : dict
        'per_values'  : list[float]
        'ler_values'  : list[float]
        'ler_std'     : list[float]   95% confidence interval half-width
        'n_shots'     : int
        'details'     : list[dict]    per-point breakdown
    """
    decoder = MWPMRepetitionDecoder()
    rng     = np.random.default_rng(seed=42)

    ler_values = []
    ler_std    = []
    details    = []

    if verbose:
        print(f'\n{"─"*58}')
        print(f'  LER vs PER Benchmark — Repetition Code + MWPM')
        print(f'{"─"*58}')
        print(f'  Logical state : |{logical_state}⟩')
        print(f'  Noise type    : {noise_type}')
        print(f'  Shots per PER : {n_shots}')
        print(f'{"─"*58}')
        print(f'  {"PER":>8}  {"LER":>10}  {"±95% CI":>10}  '
              f'{"Failures":>10}  {"Status"}')
        print(f'  {"─"*54}')

    for per in per_values:
        failures = 0
        outcomes = []

        for shot in range(n_shots):
            success, _ = run_single_shot(
                decoder, logical_state, per, noise_type, rng
            )
            outcomes.append(1 if success else 0)
            if not success:
                failures += 1

        ler  = failures / n_shots

        # 95% confidence interval using Wilson score approximation
        # For large n, CI ≈ 1.96 * sqrt(p*(1-p)/n)
        ci   = 1.96 * np.sqrt(ler * (1 - ler) / n_shots) if n_shots > 0 else 0.0

        ler_values.append(ler)
        ler_std.append(ci)

        point_details = {
            'per'      : per,
            'ler'      : round(ler, 6),
            'ci'       : round(ci, 6),
            'failures' : failures,
            'n_shots'  : n_shots,
        }
        details.append(point_details)

        # Check whether QEC is helping (LER < PER)
        qec_status = 'QEC helps' if ler < per else 'above threshold'

        if verbose:
            print(f'  {per:>8.3f}  {ler:>10.4f}  {ci:>10.4f}  '
                  f'{failures:>10}  {qec_status}')

    if verbose:
        print(f'{"─"*58}')

    return {
        'per_values'  : per_values,
        'ler_values'  : ler_values,
        'ler_std'     : ler_std,
        'n_shots'     : n_shots,
        'details'     : details,
        'logical_state': logical_state,
        'noise_type'  : noise_type,
    }


# =============================================================================
# PLOT
# =============================================================================

def plot_ler_vs_per(results, save_path=None, show=True):
    """
    Plot LER vs PER curve for the repetition code with MWPM decoding.

    Shows:
      - Measured LER with 95% confidence interval shading
      - Reference: LER = PER  (no correction baseline)
      - Reference: LER ≈ C * PER²  (ideal distance-3 suppression)
      - Threshold annotation

    Parameters
    ----------
    results   : dict   output of benchmark()
    save_path : str or None
    show      : bool
    """
    per_arr = np.array(results['per_values'])
    ler_arr = np.array(results['ler_values'])
    ci_arr  = np.array(results['ler_std'])
    n       = results['n_shots']
    ls      = results['logical_state']

    # Reference curves
    per_ref   = np.linspace(per_arr.min() * 0.5, per_arr.max() * 1.2, 200)
    ler_no_qec = per_ref                            # LER = PER (no correction)

    # Fit C for LER ≈ C * PER²
    # Use least squares on log scale: log(LER) ≈ log(C) + 2*log(PER)
    valid  = ler_arr > 0
    if valid.sum() >= 2:
        log_per  = np.log(per_arr[valid])
        log_ler  = np.log(ler_arr[valid])
        coeffs   = np.polyfit(log_per, log_ler, 1)
        slope    = round(coeffs[0], 2)
        C_fit    = np.exp(coeffs[1])
        ler_sq   = C_fit * per_ref ** 2
    else:
        slope  = 2.0
        ler_sq = 3 * per_ref ** 2

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        '3-Qubit Repetition Code — MWPM Decoding\n'
        f'LER vs Physical Error Rate  |  '
        f'logical state |{ls}⟩  |  {n} shots per point',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Left plot: log-log scale ──────────────────────────────────────────────
    ax = axes[0]

    # Confidence interval shading
    ax.fill_between(
        per_arr,
        np.maximum(ler_arr - ci_arr, 1e-6),
        ler_arr + ci_arr,
        alpha=0.20, color='black', label='95% confidence interval'
    )

    # Measured LER
    ax.plot(per_arr, ler_arr, 'o-',
            color='black', lw=2.5, ms=8, zorder=5,
            label='Repetition code + MWPM')

    # Reference: no QEC
    ax.plot(per_ref, ler_no_qec, '--',
            color='gray', lw=1.5, alpha=0.7,
            label='No QEC baseline  (LER = PER)')

    # Reference: quadratic suppression
    ax.plot(per_ref, ler_sq, ':',
            color='gray', lw=1.5, alpha=0.7,
            label=f'Ideal d=3 suppression  (LER ≈ {C_fit:.1f}·PER²,  slope≈{slope})')

    # Threshold annotation — where LER curve crosses LER=PER line
    crossings = np.where(np.diff(np.sign(ler_arr - per_arr)))[0]
    if len(crossings) > 0:
        idx   = crossings[0]
        p_thr = per_arr[idx]
        ax.axvline(x=p_thr, color='black', linestyle='-.', lw=1.0, alpha=0.5)
        ax.text(p_thr * 1.05, ax.get_ylim()[0] * 2,
                f'threshold\n≈{p_thr:.2f}',
                fontsize=9, color='black', alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical Error Rate  (PER)', fontsize=12)
    ax.set_ylabel('Logical Error Rate  (LER)', fontsize=12)
    ax.set_title('Log-Log Scale', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_xlim(per_arr.min() * 0.7, per_arr.max() * 1.5)

    # ── Right plot: linear scale ──────────────────────────────────────────────
    ax2 = axes[1]

    ax2.fill_between(
        per_arr,
        np.maximum(ler_arr - ci_arr, 0),
        ler_arr + ci_arr,
        alpha=0.20, color='black'
    )

    ax2.plot(per_arr, ler_arr, 'o-',
             color='black', lw=2.5, ms=8, zorder=5,
             label='Repetition code + MWPM')

    ax2.plot(per_arr, per_arr, '--',
             color='gray', lw=1.5, alpha=0.7,
             label='No QEC baseline  (LER = PER)')

    ax2.set_xlabel('Physical Error Rate  (PER)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate  (LER)', fontsize=12)
    ax2.set_title('Linear Scale', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(0, per_arr.max() * 1.1)
    ax2.set_ylim(0, max(ler_arr.max(), per_arr.max()) * 1.1)

    # x-axis as percentage
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x*100:.0f}%')
    )
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x*100:.1f}%')
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nPlot saved to: {save_path}')
    if show:
        plt.show()

    return fig


def print_summary_table(results):
    """Print a clean summary table of the benchmark results."""
    print(f'\n{"═"*58}')
    print(f'  RESULTS SUMMARY — Repetition Code + MWPM')
    print(f'{"═"*58}')
    print(f'  {"PER":>8}  {"LER":>10}  {"LER/PER":>10}  {"Interpretation"}')
    print(f'  {"─"*54}')

    for d in results['details']:
        per   = d['per']
        ler   = d['ler']
        ratio = ler / per if per > 0 else float('nan')
        if ratio < 0.5:
            interp = 'QEC very effective'
        elif ratio < 1.0:
            interp = 'QEC effective'
        elif ratio < 1.2:
            interp = 'near threshold'
        else:
            interp = 'above threshold'

        print(f'  {per:>8.3f}  {ler:>10.4f}  {ratio:>10.3f}  {interp}')

    print(f'{"═"*58}')
    print(f'\n  Interpretation of LER/PER ratio:')
    print(f'    < 1.0  → QEC is reducing errors (below threshold)')
    print(f'    = 1.0  → QEC has no effect (threshold)')
    print(f'    > 1.0  → QEC is making things worse (above threshold)')
    print(f'\n  Expected for distance-3 code:')
    print(f'    LER ≈ C·PER²  at low noise  (quadratic suppression)')
    print(f'    threshold ≈ 0.05–0.10 for depolarising noise')


# =============================================================================
# CLI
# =============================================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        description='LER vs PER benchmark — Repetition Code + MWPM',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python benchmarks/ler_vs_per_repetition.py
  python benchmarks/ler_vs_per_repetition.py --shots 5000
  python benchmarks/ler_vs_per_repetition.py --shots 2000 --save results/ler_rep.png
  python benchmarks/ler_vs_per_repetition.py --state 1 --noise_type x_only
        """
    )
    parser.add_argument(
        '--shots', type=int, default=3000,
        help='Number of shots per PER value  (default: 3000)'
    )
    parser.add_argument(
        '--state', type=str, default='0', choices=['0', '1'],
        help="Logical state to encode  (default: '0')"
    )
    parser.add_argument(
        '--noise_type', type=str, default='depolarising',
        choices=['depolarising', 'x_only'],
        help='Noise model type  (default: depolarising)'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Save plot to this path  e.g. results/ler_rep.png'
    )
    parser.add_argument(
        '--no_plot', action='store_true',
        help='Disable plot display'
    )
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    per_values = [0.01, 0.03, 0.05, 0.07, 0.10]

    print(f'\n{"="*58}')
    print(f'  LER vs PER — Repetition Code + MWPM')
    print(f'{"="*58}')
    print(f'  PER values    : {per_values}')
    print(f'  Shots per PER : {args.shots}')
    print(f'  Logical state : |{args.state}⟩')
    print(f'  Noise type    : {args.noise_type}')

    results = benchmark(
        per_values=per_values,
        n_shots=args.shots,
        logical_state=args.state,
        noise_type=args.noise_type,
        verbose=True
    )

    print_summary_table(results)

    plot_ler_vs_per(
        results,
        save_path=args.save,
        show=not args.no_plot
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    main()