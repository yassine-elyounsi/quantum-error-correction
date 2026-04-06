from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.quantum_info import Statevector, state_fidelity
REPETITION_SYNDROME_TABLE = {
    0b00: ('None', -1, 'No error detected'),
    0b01: ('q0',    0, 'X error on qubit 0'),
    0b11: ('q1',     1, 'X error on qubit 1'),
    0b10: ('q2',     2 , 'X error on qubit 2'),
}
def prepare_logical_qubit(qc, qubit, logical_state='0'):
    """
    Prepare a single qubit in the desired logical state
    before encoding into the repetition code.
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
            "logical_state must be '0', '1', '+', '-' or a (theta, phi) tuple.\n"
            f"Got: {logical_state!r}"
        )

def get_ideal_statevector(logical_state='0'):
    """
    Compute the ideal statevector of the encoded logical state
    without any noise or error. Used as reference for fidelity.
 
    Parameters
    ----------
    logical_state : str or tuple
 
    Returns
    -------
    sv : Statevector   3-qubit encoded statevector
    """
    data = QuantumRegister(3, 'data')
    qc   = QuantumCircuit(data)
 
    prepare_logical_qubit(qc, data[0], logical_state)
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
 
    return Statevector(qc) 
def build_encoding_circuit(logical_state='0'):
    """
    Build encoding-only circuit
 
    Parameters
    ----------
    logical_state : str or tuple
 
    Returns
    -------
    qc   : QuantumCircuit   3 data qubits, no classical registers
    data : QuantumRegister
    """
    data = QuantumRegister(3, 'data')
    qc   = QuantumCircuit(data)
 
    prepare_logical_qubit(qc, data[0], logical_state)
    qc.barrier(label='prepared')
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
 
    return qc, data   
def build_repetition_circuit(logical_state='0', error_qubit=None):
    """
    Build the syndrome extraction circuit:
        Prepare -> Encode -> Error Injection -> Syndrome Extraction
 
    Parameters
    ----------
    logical_state : str or tuple
        '0', '1', '+', '-', or (theta, phi)
    error_qubit   : int or None
        Qubit to inject a deterministic X error on (0, 1, 2), or None.
 
    Returns
    -------
    qc            : QuantumCircuit
    data          : QuantumRegister
    ancilla       : QuantumRegister
    syndrome_bits : ClassicalRegister
    """
    if error_qubit is not None and error_qubit not in [0, 1, 2]:
        raise ValueError(f'error_qubit must be 0, 1, or 2. Got {error_qubit}')
 
    data          = QuantumRegister(3, 'data')
    ancilla       = QuantumRegister(2, 'ancilla')
    syndrome_bits = ClassicalRegister(2, 'syndrome')
    qc            = QuantumCircuit(data, ancilla, syndrome_bits)
 
    # Prepare logical state
    prepare_logical_qubit(qc, data[0], logical_state)
    qc.barrier(label='prepared')
 
    # Encoding
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.barrier(label='encoded')
 
    # Deterministic error injection
    if error_qubit is not None:
        qc.x(data[error_qubit])
        qc.barrier(label=f'X_q{error_qubit}')
 
    # Syndrome extraction
    # ancilla[0] = parity(data[0] XOR data[1])
    qc.cx(data[0], ancilla[0])
    qc.cx(data[1], ancilla[0])
 
    # ancilla[1] = parity(data[1] XOR data[2])
    qc.cx(data[1], ancilla[1])
    qc.cx(data[2], ancilla[1])
 
    # Measure ancilla only
    qc.measure(ancilla[0], syndrome_bits[0])
    qc.measure(ancilla[1], syndrome_bits[1])
 
    return qc, data, ancilla, syndrome_bits
def build_noise_model(p, noise_type='depolarising'):
    """
    Build a Qiskit Aer noise model.
 
    Parameters
    ----------
    p          : float   error probability per gate [0.0, 1.0]
    noise_type : str
        'depolarising' -> X, Y, Z randomly (realistic general noise)
        'x_only'       -> X only (matches what repetition code corrects)
 
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
            depolarizing_error(p, 1),  ['x', 'h', 'id', 'ry', 'rz']
        )
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), ['cx']
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
        nm.add_all_qubit_quantum_error(single, ['x', 'h', 'id', 'ry', 'rz'])
        nm.add_all_qubit_quantum_error(two,    ['cx'])
 
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
def decode_repetition_syndrome(syndrome_int):
    """
    Decode a 2-bit syndrome integer for the 3-qubit repetition code.
 
    Parameters
    ----------
    syndrome_int : int
        Syndrome as integer (0-3).
        
    Returns
    -------
    error_type  : str   'X', 'None', or 'Unknown'
    qubit       : int   physical qubit to correct (0-2), or -1
    description : str
    """
    if syndrome_int in REPETITION_SYNDROME_TABLE:
        return REPETITION_SYNDROME_TABLE[syndrome_int]
 
    return ('Unknown', -1, f'Unexpected syndrome: {syndrome_int:02b}')
def run_syndrome_extraction(logical_state='0', error_qubit=None,
                             p=0.0, shots=1, noise_type='depolarising'):
    """
    Run the syndrome extraction circuit and return the measured syndrome.
 
    Parameters
    ----------
    logical_state : str or tuple
    error_qubit   : int or None   deterministic error qubit
    p             : float         noise rate (0.0 = no noise)
    shots         : int           number of circuit executions
    noise_type    : str
 
    Returns
    -------
    syndrome_int : int    dominant syndrome (little-endian corrected)
    syndrome_str : str    dominant syndrome as binary string e.g. '01'
    counts       : dict   full measurement histogram
    """
    qc, _, _, _ = build_repetition_circuit(logical_state, error_qubit)
    sim          = _get_simulator(p, noise_type)
    counts       = sim.run(qc, shots=shots).result().get_counts()
 
    # Get dominant syndrome from counts
    raw          = max(counts, key=counts.get)
    # Fix Qiskit little-endian bit ordering by reversing the string
    syndrome_str = raw.replace(' ', '')[::-1]
    syndrome_int = int(syndrome_str, 2)
 
    return syndrome_int, syndrome_str, counts

def run_full_pipeline(logical_state='0', error_qubit=None,
                      p=0.0, shots=1024, noise_type='depolarising'):
    """
    Run the complete QEC pipeline:
        Syndrome extraction -> decode -> correction -> decode -> measure
 
    Fidelity strategy:
        '0' or '1'     -> measurement-based fidelity (fast)
        superposition  -> statevector-based fidelity (exact)
 
    Parameters
    ----------
    logical_state : str or tuple
    error_qubit   : int or None
    p             : float         noise rate
    shots         : int           used only for classical states
    noise_type    : str
 
    Returns
    -------
   
    syndrome_int     : int
    error_type       : str
   
    """
    # Step 1 — extract syndrome (1 shot)
    syndrome_int, _, _ = run_syndrome_extraction(
        logical_state, error_qubit,
        p=p, shots=1, noise_type=noise_type
    )
 
    # Step 2 — decode syndrome
    error_type, correction_qubit, description = decode_repetition_syndrome(
        syndrome_int
    )
    
    return syndrome_int, error_type, correction_qubit