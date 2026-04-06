from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.quantum_info import Statevector, state_fidelity
import numpy as np
def _build_shor_syndrome_table():
    """Build complete 22-entry syndrome table for the Shor code."""
    table = {}
 
    # No error
    table[0] = ('None', -1, 'None', 'No error detected')
 
    # X and Y error patterns within each group
    x_within = {
        0b01: 0,   # error on first qubit of group
        0b11: 1,   # error on second qubit of group
        0b10: 2,   # error on third qubit of group
    }
    group_starts      = [0, 3, 6]
    group_bit_offsets = [0, 2, 4]
 
    # Z syndrome per group
    z_syn_per_group = {
        0: 0b01000000,   # Group 0
        1: 0b11000000,   # Group 1
        2: 0b10000000,   # Group 2
    }
 
    for grp_idx, (q_start, bit_off) in enumerate(
            zip(group_starts, group_bit_offsets)):
        for syn_pattern, local_q in x_within.items():
            phys_q = q_start + local_q
            x_syn  = syn_pattern << bit_off
            z_syn  = z_syn_per_group[grp_idx]
            y_syn  = x_syn | z_syn
 
            # X error entry
            table[x_syn] = (
                'X', phys_q,
                f'X on q{phys_q}',
                f'X error on qubit {phys_q} (Group {grp_idx}, local q{local_q})'
            )
 
            # Y error entry — combined X and Z syndrome
            table[y_syn] = (
                'Y', phys_q,
                f'Y on q{phys_q}',
                f'Y error on qubit {phys_q} (Group {grp_idx}, local q{local_q})'
            )
 
    # Z error entries — phase-flip across groups
    z_entries = {
        0b01000000: (0, 'Z on q0', 'Z error in Group 0 -> apply Z on q0'),
        0b11000000: (3, 'Z on q3', 'Z error in Group 1 -> apply Z on q3'),
        0b10000000: (6, 'Z on q6', 'Z error in Group 2 -> apply Z on q6'),
    }
    for syn_int, (phys_q, corr, desc) in z_entries.items():
        table[syn_int] = ('Z', phys_q, corr, desc)
 
    return table
SHOR_SYNDROME_TABLE=_build_shor_syndrome_table()
def prepare_logical_qubit(qc, qubit, logical_state='0'):
    """
    Prepare a single qubit in the desired logical state
    before encoding into the Shor code.
 
    Parameters
    ----------
    qc            : QuantumCircuit   circuit to modify in place
    qubit         : Qubit            always data[0]
    logical_state : str or tuple
        '0'            -> |0>  (default, no gate needed)
        '1'            -> |1>  (X gate)
        '+'            -> |+> = (|0> + |1>)/sqrt(2)
        '-'            -> |-> = (|0> - |1>)/sqrt(2)
        (theta, phi)   -> cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
    """
    if logical_state == '0':
        pass
 
    elif logical_state == '1':
        qc.x(qubit)
 
    elif logical_state == '+':
        qc.h(qubit)
 
    elif logical_state == '-':
        qc.x(qubit)
        qc.h(qubit)
 
    elif isinstance(logical_state, (tuple, list)) and len(logical_state) == 2:
        theta, phi = float(logical_state[0]), float(logical_state[1])
        qc.ry(theta, qubit)
        qc.rz(phi,   qubit)
 
    else:
        raise ValueError(
            "logical_state must be '0','1','+','-' or a (theta, phi) tuple.\n"
            f"Got: {logical_state!r}"
        )
def get_ideal_statevector(logical_state='0'):
    """
    Compute the ideal 9-qubit statevector of the encoded logical state
    without any noise or error. Used as reference for fidelity.
 
    Parameters
    ----------
    logical_state : str or tuple
 
    Returns
    -------
    sv : Statevector   9-qubit encoded statevector
    """
    data = QuantumRegister(9, 'q')
    qc   = QuantumCircuit(data)
 
    prepare_logical_qubit(qc, data[0], logical_state)
 
    # Outer encoding
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[6])
    qc.h(data[0]); qc.h(data[3]); qc.h(data[6])
 
    # Inner encoding
    qc.cx(data[0], data[1]); qc.cx(data[0], data[2])
    qc.cx(data[3], data[4]); qc.cx(data[3], data[5])
    qc.cx(data[6], data[7]); qc.cx(data[6], data[8])
 
    return Statevector(qc)
def decode_shor_syndrome(syndrome_int):
    """
    Decode an 8-bit syndrome integer for the 9-qubit Shor code.
 
    Parameters
    ----------
    syndrome_int : int
        Syndrome as integer (0-255).
        Must be computed with bit 0 as LSB (after little-endian reversal).
 
    Returns
    -------
    error_type  : str   'X', 'Z', 'Y', 'None', or 'Unknown'
    qubit       : int   physical qubit to correct (0-8), or -1
    correction  : str   correction gate description
    description : str   human-readable description
    """
    if syndrome_int in SHOR_SYNDROME_TABLE:
        return SHOR_SYNDROME_TABLE[syndrome_int]
 
    return (
        'Unknown', -1, 'None',
        f'Multi-qubit or unknown error (syndrome={syndrome_int:08b})'
    )
def build_encoding_circuit(logical_state='0'):
    """
    Build encoding-only circuit (no error, no syndrome, no measurement).
 
    Parameters
    ----------
    logical_state : str or tuple
 
    Returns
    -------
    qc   : QuantumCircuit   9 data qubits, no classical registers
    data : QuantumRegister
    """
    data = QuantumRegister(9, 'q')
    qc   = QuantumCircuit(data)
 
    prepare_logical_qubit(qc, data[0], logical_state)
    qc.barrier(label='prepared')
 
    # Outer encoding — phase-flip protection
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[6])
    qc.h(data[0]); qc.h(data[3]); qc.h(data[6])
    qc.barrier(label='outer_enc')
 
    # Inner encoding — bit-flip protection
    qc.cx(data[0], data[1]); qc.cx(data[0], data[2])
    qc.cx(data[3], data[4]); qc.cx(data[3], data[5])
    qc.cx(data[6], data[7]); qc.cx(data[6], data[8])
    qc.barrier(label='inner_enc')
 
    return qc, data
def build_shor_circuit(logical_state='0', error_qubit=None, error_type='None'):
    """
    Build the full Shor syndrome extraction circuit:
        Prepare -> Encode -> Error Injection -> 8-bit Syndrome Extraction
 
    Measures only the 8 ancilla qubits.
    Data qubits are NOT measured.
 
    Parameters
    ----------
    logical_state : str or tuple
        '0', '1', '+', '-', or (theta, phi)
    error_qubit   : int or None
        Physical qubit to inject error on (0-8), or None.
    error_type    : str
        'X', 'Z', or 'Y'
 
    Returns
    -------
    qc            : QuantumCircuit
    data          : QuantumRegister
    ancilla       : QuantumRegister
    syndrome_bits : ClassicalRegister
    """
    if error_qubit is not None and error_qubit not in range(9):
        raise ValueError(f'error_qubit must be 0-8. Got {error_qubit}')
    if error_type not in ('X', 'Z', 'Y','None'):
        raise ValueError(f"error_type must be 'X', 'Z', or 'Y'. Got {error_type!r}")
 
    data          = QuantumRegister(9, 'q')
    ancilla       = QuantumRegister(8, 'anc')
    syndrome_bits = ClassicalRegister(8, 'syn')
    qc            = QuantumCircuit(data, ancilla, syndrome_bits)
 
    # ── PREPARE LOGICAL STATE ─────────────────────────────────────────────────
    prepare_logical_qubit(qc, data[0], logical_state)
    qc.barrier(label='prepared')
 
    # ── OUTER ENCODING ────────────────────────────────────────────────────────
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[6])
    qc.h(data[0]); qc.h(data[3]); qc.h(data[6])
 
    # ── INNER ENCODING ────────────────────────────────────────────────────────
    qc.cx(data[0], data[1]); qc.cx(data[0], data[2])
    qc.cx(data[3], data[4]); qc.cx(data[3], data[5])
    qc.cx(data[6], data[7]); qc.cx(data[6], data[8])
    qc.barrier(label='encoded')
 
    # ── ERROR INJECTION ──
    # 
    if error_type is not 'None':
      if error_qubit is not None:
        if error_type == 'X':
            qc.x(data[error_qubit])
        elif error_type == 'Z':
            qc.z(data[error_qubit])
        elif error_type == 'Y':
            qc.x(data[error_qubit])
            qc.z(data[error_qubit])
        qc.barrier(label=f'{error_type}_q{error_qubit}')
 
    # ── X-ERROR SYNDROMES (6 bits) ────────────────────────────────────────────
    # Group 0: parity(q0,q1) -> anc[0],  parity(q1,q2) -> anc[1]
    qc.cx(data[0], ancilla[0]); qc.cx(data[1], ancilla[0])
    qc.cx(data[1], ancilla[1]); qc.cx(data[2], ancilla[1])
 
    # Group 1: parity(q3,q4) -> anc[2],  parity(q4,q5) -> anc[3]
    qc.cx(data[3], ancilla[2]); qc.cx(data[4], ancilla[2])
    qc.cx(data[4], ancilla[3]); qc.cx(data[5], ancilla[3])
 
    # Group 2: parity(q6,q7) -> anc[4],  parity(q7,q8) -> anc[5]
    qc.cx(data[6], ancilla[4]); qc.cx(data[7], ancilla[4])
    qc.cx(data[7], ancilla[5]); qc.cx(data[8], ancilla[5])
    qc.barrier()
 
    # ── Z-ERROR SYNDROMES (2 bits) ────────────────────────────────────────────
    # CORRECT approach: H on ancilla, CX from ancilla to data leaders, H on ancilla
    # This reads phase parity WITHOUT disturbing the encoded data qubits
 
    # anc[6]: phase parity between Group 0 leader (q0) and Group 1 leader (q3)
    # Apply H on the 6 data qubits
    for q in [0,1,2,3,4,5]:
       qc.h(data[q])

    # CNOTs: data → ancilla
    for q in [0,1,2,3,4,5]:
       qc.cx(data[q], ancilla[6])

    # Apply H again
    for q in [0,1,2,3,4,5]:
      qc.h(data[q])
    # anc[7]: phase parity between Group 1 leader (q3) and Group 2 leader (q6)
    # Apply H on the 6 data qubits
    for q in [3,4,5,6,7,8]:
       qc.h(data[q])

    # CNOTs
    for q in [3,4,5,6,7,8]:
      qc.cx(data[q], ancilla[7])

    # Apply H again
    for q in [3,4,5,6,7,8]:
      qc.h(data[q])
      qc.barrier()
 
    # ── MEASURE ALL ANCILLA ───────────────────────────────────────────────────
    for i in range(8):
        qc.measure(ancilla[i], syndrome_bits[i])
 
    return qc, data, ancilla, syndrome_bits 
def build_noise_model(p, noise_type='depolarising'):
    """
    Build a Qiskit Aer noise model.
 
    Parameters
    ----------
    p          : float   error probability per gate [0.0, 1.0]
    noise_type : str
        'depolarising' -> X, Y, Z randomly (realistic)
        'x_only'       -> X only
 
    Returns
    -------
    noise_model : NoiseModel
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'p must be in [0.0, 1.0]. Got {p}')
 
    nm = NoiseModel()
 
    if noise_type == 'depolarising':
        p2 = min(p * 2, 0.99)
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p, 1),  ['x', 'z', 'y', 'h', 'id', 'ry', 'rz']
        )
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), ['cx', 'cz']
        )
 
    elif noise_type == 'x_only':
        p2     = min(p * 2, 0.99)
        single = pauli_error([('X', p), ('I', 1 - p)])
        two    = pauli_error([
            ('IX', p2 / 3),
            ('XI', p2 / 3),
            ('XX', p2 / 3),
            ('II', 1 - p2),
        ])
        nm.add_all_qubit_quantum_error(single, ['x', 'z', 'y', 'h', 'id', 'ry', 'rz'])
        nm.add_all_qubit_quantum_error(two,    ['cx', 'cz'])
 
    else:
        raise ValueError(
            f"noise_type must be 'depolarising' or 'x_only'. Got {noise_type!r}"
        )
 
    return nm
 
 
def _get_simulator(p=0.0, noise_type='depolarising'):
    """Internal — returns noisy or noiseless AerSimulator."""
    if p > 0.0:
        return AerSimulator(noise_model=build_noise_model(p, noise_type))
    return AerSimulator()
 
 
# =============================================================================
# SYNDROME RUNNER
# =============================================================================
 
def run_syndrome_extraction(logical_state='0', error_qubit=None,
                             error_type='None', p=0.0, shots=1,
                             noise_type='depolarising'):
    """
    Run the Shor syndrome extraction circuit and return the measured syndrome.
 
    Parameters
    ----------
    logical_state : str or tuple
    error_qubit   : int or None
    error_type    : str           'X', 'Z', or 'Y'
    p             : float         noise rate
    shots         : int
    noise_type    : str
 
    Returns
    -------
    syndrome_int : int    dominant 8-bit syndrome (little-endian corrected)
    syndrome_str : str    dominant syndrome as binary string e.g. '00000001'
    counts       : dict   full measurement histogram
    """
    qc, _, _, _ = build_shor_circuit(logical_state, error_qubit, error_type)
    sim          = _get_simulator(p, noise_type)
    counts       = sim.run(qc, shots=shots).result().get_counts()
 
    # Dominant syndrome
    raw          = max(counts, key=counts.get)
    # Fix Qiskit little-endian bit ordering
    syndrome_str = raw.replace(' ', '')[::-1]
    syndrome_int = int(syndrome_str, 2)
 
    return syndrome_int, syndrome_str, counts
 