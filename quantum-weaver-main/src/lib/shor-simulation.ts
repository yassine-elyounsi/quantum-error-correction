// Shor 9-qubit code simulation engine
// Matches the PPO agent: obs=MultiBinary(8), action=Discrete(28)

export type PauliError = 'I' | 'X' | 'Y' | 'Z';
export type QubitState = { id: number; error: PauliError; correction: PauliError };
export type Phase = 'idle' | 'encode' | 'noise' | 'syndrome' | 'rl_decode' | 'correct' | 'measure';

export interface SimulationState {
  phase: Phase;
  qubits: QubitState[];
  syndrome: { sx: number[]; sz: number[] };
  agentAction: number;
  logicalPreserved: boolean | null;
  noiseP: number;
  shots: number;
  successes: number;
  failures: number;
}

export function createInitialState(): SimulationState {
  return {
    phase: 'idle',
    qubits: Array.from({ length: 9 }, (_, i) => ({ id: i, error: 'I', correction: 'I' })),
    syndrome: { sx: [0, 0, 0, 0, 0, 0], sz: [0, 0] },
    agentAction: 27, // identity
    logicalPreserved: null,
    noiseP: 0.2,
    shots: 0,
    successes: 0,
    failures: 0,
  };
}

function applyBernoulliNoise(p: number): PauliError {
  const r = Math.random();
  if (r < p / 3) return 'X';
  if (r < (2 * p) / 3) return 'Y';
  if (r < p) return 'Z';
  return 'I';
}

// Shor code X-syndrome: compare pairs within each group of 3
// Group 0: q0,q1,q2 → sx0 = q0⊕q1, sx1 = q1⊕q2
// Group 1: q3,q4,q5 → sx2 = q3⊕q4, sx3 = q4⊕q5
// Group 2: q6,q7,q8 → sx4 = q6⊕q7, sx5 = q7⊕q8
function hasXComponent(e: PauliError): number {
  return e === 'X' || e === 'Y' ? 1 : 0;
}

function hasZComponent(e: PauliError): number {
  return e === 'Z' || e === 'Y' ? 1 : 0;
}

function computeSyndrome(qubits: QubitState[]): { sx: number[]; sz: number[] } {
  const x = qubits.map(q => hasXComponent(q.error));
  const z = qubits.map(q => hasZComponent(q.error));

  const sx = [
    x[0] ^ x[1], x[1] ^ x[2],
    x[3] ^ x[4], x[4] ^ x[5],
    x[6] ^ x[7], x[7] ^ x[8],
  ];

  // Z syndrome: compare group-level Z parities
  // Each group's effective Z = XOR of its 3 qubits' Z components
  const g0z = z[0] ^ z[1] ^ z[2];
  const g1z = z[3] ^ z[4] ^ z[5];
  const g2z = z[6] ^ z[7] ^ z[8];
  const sz = [g0z ^ g1z, g1z ^ g2z];

  return { sx, sz };
}

// Decode action index to correction
// 0-8: X on qubit i, 9-17: Z on qubit i, 18-26: Y on qubit i, 27: identity
export function decodeAction(action: number): { qubit: number; pauli: PauliError } | null {
  if (action === 27) return null;
  if (action < 9) return { qubit: action, pauli: 'X' };
  if (action < 18) return { qubit: action - 9, pauli: 'Z' };
  if (action < 27) return { qubit: action - 18, pauli: 'Y' };
  return null;
}

// Simple majority-vote decoder (mimics what the trained PPO learned)
function rlDecode(sx: number[], sz: number[]): number {
  // X error correction per group
  for (let g = 0; g < 3; g++) {
    const s0 = sx[g * 2];
    const s1 = sx[g * 2 + 1];
    if (s0 && s1) return g * 3 + 1; // X on middle qubit
    if (s0) return g * 3; // X on first qubit
    if (s1) return g * 3 + 2; // X on third qubit
  }

  // Z error correction across groups
  if (sz[0] && sz[1]) {
    return 9 + 3; // Z on group 1 representative (q3)
  }
  if (sz[0]) return 9; // Z on group 0 representative (q0)
  if (sz[1]) return 9 + 6; // Z on group 2 representative (q6)

  return 27; // identity
}

// Check if logical qubit is preserved after correction
function checkLogical(qubits: QubitState[]): boolean {
  // After correction, check if residual errors form a logical operator
  // For simplicity: no residual X/Y/Z errors means preserved
  for (const q of qubits) {
    const eX = hasXComponent(q.error) ^ hasXComponent(q.correction);
    const eZ = hasZComponent(q.error) ^ hasZComponent(q.correction);
    if (eX || eZ) {
      // Check if uncorrected — but for Shor code the logical ops span all 9 qubits
      // Simplified: single residual errors that weren't corrected = failure
      return false;
    }
  }
  return true;
}

export function runShot(noiseP: number): {
  qubits: QubitState[];
  syndrome: { sx: number[]; sz: number[] };
  agentAction: number;
  logicalPreserved: boolean;
} {
  // Apply noise
  const qubits: QubitState[] = Array.from({ length: 9 }, (_, i) => ({
    id: i,
    error: applyBernoulliNoise(noiseP),
    correction: 'I',
  }));

  // Compute syndrome
  const syndrome = computeSyndrome(qubits);

  // RL decode
  const agentAction = rlDecode(syndrome.sx, syndrome.sz);
  const decoded = decodeAction(agentAction);
  if (decoded) {
    qubits[decoded.qubit].correction = decoded.pauli;
  }

  // Check logical
  const logicalPreserved = checkLogical(qubits);

  return { qubits, syndrome, agentAction, logicalPreserved };
}
