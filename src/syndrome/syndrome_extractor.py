# =============================================================================
# syndrome_extractor.py
# =============================================================================
# Unified syndrome extractor for both repetition and Shor codes.
#
# CLI Usage:
#   python syndrome_extractor.py --code repetition --noise 0.05 --state 0 --shots 1000
#   python syndrome_extractor.py --code shor        --noise 0.03 --state + --shots 500
#   python syndrome_extractor.py --code shor        --noise 0.1  --state "1.57,0.5" --shots 200
#
# Import Usage:
#   from syndrome.syndrome_extractor import SyndromeExtractor
#   ext   = SyndromeExtractor('shor')
#   batch = ext.extract(logical_state='+', noise_rate=0.05, shots=1000)
#   ext.plot_histogram(batch)
# =============================================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from src.circuits.repetition_code import (
    decode_repetition_syndrome,
    run_syndrome_extraction as rep_run_syndrome,
)
from src.circuits.shor_code import (
    decode_shor_syndrome,
    run_syndrome_extraction as shor_run_syndrome,
)


# =============================================================================
# STATE PARSER
# =============================================================================

def parse_logical_state(state_str):
    """
    Parse a logical state string into the correct format.

    Accepts:
        '0'         ->  '0'
        '1'         ->  '1'
        '+'         ->  '+'
        '-'         ->  '-'
        '1.57,0.5'  ->  (1.57, 0.5)   Bloch sphere (theta, phi)

    Parameters
    ----------
    state_str : str

    Returns
    -------
    logical_state : str or tuple
    """
    if state_str in ('0', '1', '+', '-'):
        return state_str

    try:
        parts = state_str.split(',')
        if len(parts) == 2:
            return (float(parts[0].strip()), float(parts[1].strip()))
    except ValueError:
        pass

    raise ValueError(
        f"Cannot parse logical_state {state_str!r}.\n"
        f"Valid: '0', '1', '+', '-', or 'theta,phi' e.g. '1.57,0.5'"
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _extract_one_repetition(logical_state, noise_rate, noise_type):
    """
    Run one shot of repetition code syndrome extraction.
    Randomly injects X error on one qubit (0,1,2) or no error.
    """
    true_error_qubit = np.random.choice([None, 0, 1, 2])
    true_error_type  = 'X' if true_error_qubit is not None else 'None'

    syndrome_int, syndrome_str, _ = rep_run_syndrome(
        logical_state=logical_state,
        error_qubit=true_error_qubit,
        p=noise_rate,
        shots=1,
        noise_type=noise_type
    )

    decoded_type, decoded_qubit, _ = decode_repetition_syndrome(syndrome_int)

    return {
        'syndrome_int'    : syndrome_int,
        'syndrome_str'    : syndrome_str,
        'syndrome_vec'    : [int(b) for b in syndrome_str],
        'decoded_type'    : decoded_type,
        'decoded_qubit'   : decoded_qubit,
        'true_error_qubit': true_error_qubit if true_error_qubit is not None else -1,
        'true_error_type' : true_error_type,
        'logical_state'   : str(logical_state),
        'code'            : 'repetition',
        'noise_rate'      : noise_rate,
    }


def _extract_one_shor(logical_state, noise_rate, noise_type):
    """
    Run one shot of Shor code syndrome extraction.
    Randomly injects X, Z, or Y error on one qubit (0-8) or no error.
    """
    true_error_qubit = np.random.choice([None, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    true_error_type  = (
        np.random.choice(['X', 'Z', 'Y'])
        if true_error_qubit is not None
        else 'None'
    )

    syndrome_int, syndrome_str, _ = shor_run_syndrome(
        logical_state=logical_state,
        error_qubit=true_error_qubit,
        error_type='None',
        p=noise_rate,
        shots=1,
        noise_type=noise_type
    )

    decoded_type, decoded_qubit, correction, description = \
        decode_shor_syndrome(syndrome_int)

    return {
        'syndrome_int'    : syndrome_int,
        'syndrome_str'    : syndrome_str,
        'syndrome_vec'    : [int(b) for b in syndrome_str],
        'decoded_type'    : decoded_type,
        'decoded_qubit'   : decoded_qubit,
        'true_error_qubit': true_error_qubit if true_error_qubit is not None else -1,
        'true_error_type' : true_error_type,
        'logical_state'   : str(logical_state),
        'code'            : 'shor',
        'noise_rate'      : noise_rate,
    }


# =============================================================================
# MAIN CLASS
# =============================================================================

class SyndromeExtractor:
    """
    Unified syndrome extractor for repetition and Shor codes.

    Parameters
    ----------
    code_type : str   'repetition' or 'shor'
    """

    SUPPORTED_CODES = ('repetition', 'shor')

    def __init__(self, code_type='repetition'):
        if code_type not in self.SUPPORTED_CODES:
            raise ValueError(
                f"code_type must be one of {self.SUPPORTED_CODES}. "
                f"Got {code_type!r}"
            )
        self.code_type       = code_type
        self.n_syndrome_bits = 2 if code_type == 'repetition' else 8

    def extract(self, logical_state='0', noise_rate=0.05,
                shots=1000, noise_type='depolarising'):
        """
        Extract a batch of syndrome samples.

        Parameters
        ----------
        logical_state : str or tuple
            '0', '1', '+', '-', or (theta, phi)
        noise_rate    : float   [0.0, 1.0]
        shots         : int     number of samples
        noise_type    : str     'depolarising' or 'x_only'

        Returns
        -------
        batch : list of dict, each containing:
            'syndrome_int'     : int
            'syndrome_str'     : str    binary string
            'syndrome_vec'     : list   ← observation vector for RL agent
            'decoded_type'     : str    decoded error type
            'decoded_qubit'    : int    decoded qubit index
            'true_error_qubit' : int    injected qubit (-1 if none)
            'true_error_type'  : str    injected error type
            'logical_state'    : str
            'code'             : str
            'noise_rate'       : float
        """
        if not 0.0 <= noise_rate <= 1.0:
            raise ValueError(f'noise_rate must be in [0.0, 1.0]. Got {noise_rate}')
        if shots < 1:
            raise ValueError(f'shots must be >= 1. Got {shots}')

        _extractor = (
            _extract_one_repetition
            if self.code_type == 'repetition'
            else _extract_one_shor
        )

        batch = []
        for i in range(shots):
            sample = _extractor(logical_state, noise_rate, noise_type)
            batch.append(sample)
            if shots >= 200 and (i + 1) % max(1, shots // 5) == 0:
                print(f'  [{self.code_type}] {i+1}/{shots} samples done')

        return batch

    def summary(self, batch):
        """
        Compute and print summary statistics from a batch.

        Returns
        -------
        stats : dict
        """
        if not batch:
            print('Empty batch.')
            return {}

        total               = len(batch)
        syndrome_counts     = Counter(s['syndrome_str'] for s in batch)
        decoded_type_counts = Counter(s['decoded_type'] for s in batch)
        no_error_rate       = decoded_type_counts.get('None', 0) / total
        unknown_rate        = decoded_type_counts.get('Unknown', 0) / total

        stats = {
            'total_samples'       : total,
            'syndrome_counts'     : syndrome_counts,
            'decoded_type_counts' : decoded_type_counts,
            'no_error_rate'       : round(no_error_rate, 4),
            'unknown_rate'        : round(unknown_rate, 4),
            'code_type'           : self.code_type,
            'noise_rate'          : batch[0]['noise_rate'],
            'logical_state'       : batch[0]['logical_state'],
        }

        print(f'\n{"─"*50}')
        print(f'  SUMMARY — {self.code_type.upper()} CODE')
        print(f'{"─"*50}')
        print(f'  Logical state   : |{stats["logical_state"]}⟩')
        print(f'  Noise rate      : {stats["noise_rate"]*100:.1f}%')
        print(f'  Total samples   : {total}')
        print(f'  No-error rate   : {no_error_rate:.1%}')
        print(f'  Unknown rate    : {unknown_rate:.1%}')
        print(f'  Decoded types   : {dict(decoded_type_counts)}')
        print(f'  Top 5 syndromes :')
        for syn, cnt in syndrome_counts.most_common(5):
            print(f'    {syn}  →  {cnt:4d} times  ({cnt/total*100:.1f}%)')
        print(f'{"─"*50}\n')

        return stats

    def plot_histogram(self, batch, top_n=15, save_path=None, show=True):
        """
        Plot syndrome frequency histogram (bar chart + pie chart).

        Parameters
        ----------
        batch     : list of dict
        top_n     : int            number of top syndromes to show
        save_path : str or None    saves figure if provided
        show      : bool
        """
        if not batch:
            print('Empty batch — nothing to plot.')
            return

        total         = len(batch)
        noise_rate    = batch[0]['noise_rate']
        logical_state = batch[0]['logical_state']
        syn_counts    = Counter(s['syndrome_str'] for s in batch)
        type_counts   = Counter(s['decoded_type'] for s in batch)

        top_syn = syn_counts.most_common(top_n)
        labels  = [s for s, _ in top_syn]
        counts  = [c for _, c in top_syn]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # ── Left: syndrome frequency bar chart ───────────────────────────────
        ax = axes[0]
        bars = ax.bar(range(len(labels)), counts,
                      color='#2c2c2c', edgecolor='black', linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels, rotation=50, ha='right',
            fontsize=8, fontfamily='monospace'
        )
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Syndrome (binary)', fontsize=12)
        ax.set_title(
            f'Syndrome Frequency — {self.code_type.capitalize()} Code\n'
            f'state = |{logical_state}⟩   '
            f'noise = {noise_rate*100:.1f}%   '
            f'shots = {total}',
            fontsize=12, fontweight='bold'
        )
        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(c), ha='center', va='bottom', fontsize=7
            )

        # ── Right: decoded error type pie chart ───────────────────────────────
        ax2 = axes[1]
        pie_labels = list(type_counts.keys())
        pie_sizes  = list(type_counts.values())
        shades     = ['#1a1a1a', '#555555', '#888888',
                      '#aaaaaa', '#cccccc', '#eeeeee']
        colors     = shades[:len(pie_labels)]
        ax2.pie(
            pie_sizes,
            labels=[f'{l}  ({v})' for l, v in zip(pie_labels, pie_sizes)],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax2.set_title(
            f'Decoded Error Distribution\n'
            f'{self.code_type.capitalize()} — noise = {noise_rate*100:.1f}%',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Histogram saved to: {save_path}')
        if show:
            plt.show()


# =============================================================================
# STANDALONE FUNCTIONAL INTERFACE
# =============================================================================

def extract_syndrome_batch(code_type='repetition', logical_state='0',
                            noise_rate=0.05, shots=1000,
                            noise_type='depolarising'):
    """
    Functional shortcut — same as SyndromeExtractor.extract().
    """
    return SyndromeExtractor(code_type).extract(
        logical_state=logical_state,
        noise_rate=noise_rate,
        shots=shots,
        noise_type=noise_type
    )


# =============================================================================
# CLI
# =============================================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        description='Syndrome Extractor — QEC batch syndrome generation',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python syndrome_extractor.py --code repetition --noise 0.05 --state 0 --shots 1000
  python syndrome_extractor.py --code shor        --noise 0.03 --state + --shots 500
  python syndrome_extractor.py --code repetition  --noise 0.05 --state - --shots 300
  python syndrome_extractor.py --code shor        --noise 0.10 --state "1.57,0.5" --shots 200
  python syndrome_extractor.py --code shor        --noise 0.05 --state 0 --shots 1000 --save hist.png
  python syndrome_extractor.py --code repetition  --noise 0.05 --state 0 --no_plot
        """
    )

    parser.add_argument(
        '--code', '-c',
        type=str,
        required=True,
        choices=['repetition', 'shor'],
        help="Code type: 'repetition' (2-bit syndrome) or 'shor' (8-bit syndrome)"
    )

    parser.add_argument(
        '--noise', '-n',
        type=float,
        required=True,
        metavar='RATE',
        help='Noise rate per gate in [0.0, 1.0],  e.g. 0.05 for 5%%'
    )

    parser.add_argument(
        '--state', '-s',
        type=str,
        required=True,
        metavar='STATE',
        help=(
            "Logical state to encode:\n"
            "  '0'         ->  |0>\n"
            "  '1'         ->  |1>\n"
            "  '+'         ->  |+> = (|0>+|1>)/sqrt(2)\n"
            "  '-'         ->  |-> = (|0>-|1>)/sqrt(2)\n"
            "  'theta,phi' ->  general Bloch sphere,  e.g. '1.57,0.5'"
        )
    )

    parser.add_argument(
        '--shots',
        type=int,
        default=1000,
        metavar='N',
        help='Number of syndrome samples to generate  (default: 1000)'
    )

    parser.add_argument(
        '--noise_type',
        type=str,
        default='depolarising',
        choices=['depolarising', 'x_only'],
        help="Noise model type  (default: depolarising)"
    )

    parser.add_argument(
        '--save',
        type=str,
        default=None,
        metavar='PATH',
        help='Save histogram image to this path,  e.g. output/hist.png'
    )

    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Disable plot display  (useful in headless / server environments)'
    )

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # Parse and validate logical state
    try:
        logical_state = parse_logical_state(args.state)
    except ValueError as e:
        print(f'\nError parsing --state: {e}\n')
        parser.print_help()
        return

    # Validate noise rate
    if not 0.0 <= args.noise <= 1.0:
        print(f'\nError: --noise must be between 0.0 and 1.0. Got {args.noise}\n')
        return

    # Print run configuration
    print(f'\n{"="*50}')
    print(f'  SYNDROME EXTRACTOR')
    print(f'{"="*50}')
    print(f'  Code type     : {args.code}')
    print(f'  Logical state : |{logical_state}⟩')
    print(f'  Noise rate    : {args.noise*100:.1f}%')
    print(f'  Shots         : {args.shots}')
    print(f'  Noise type    : {args.noise_type}')
    print(f'{"="*50}\n')

    # Run extraction
    extractor = SyndromeExtractor(args.code)
    batch     = extractor.extract(
        logical_state=logical_state,
        noise_rate=args.noise,
        shots=args.shots,
        noise_type=args.noise_type
    )

    # Summary
    extractor.summary(batch)

    # Show first 5 samples
    print('First 5 samples:')
    for i, s in enumerate(batch[:5]):
        print(f'  [{i}]  syndrome={s["syndrome_str"]}  '
              f'decoded={s["decoded_type"]} q{s["decoded_qubit"]}  '
              f'true={s["true_error_type"]} q{s["true_error_qubit"]}')

    # Plot
    extractor.plot_histogram(
        batch,
        save_path=args.save,
        show=not args.no_plot
    )


if __name__ == '__main__':
    main()