import { type QubitState, type Phase } from '@/lib/shor-simulation';

interface ShorCircuitProps {
  qubits: QubitState[];
  phase: Phase;
}

const PHASES = ['encode', 'noise', 'syndrome', 'rl_decode', 'correct', 'measure'] as const;
const PHASE_LABELS = ['Encode', 'Noise', 'Syndrome', 'RL Decode', 'Correct', 'Measure'];

function PhaseIndicator({ phase }: { phase: Phase }) {
  return (
    <div className="flex gap-2 mb-4">
      {PHASES.map((p, i) => {
        const active = PHASES.indexOf(phase as any) >= i;
        return (
          <span
            key={p}
            className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
              active
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary text-muted-foreground'
            }`}
          >
            {PHASE_LABELS[i]}
          </span>
        );
      })}
    </div>
  );
}

function GateBox({ label, variant = 'default' }: { label: string; variant?: 'default' | 'hadamard' | 'cnot-ctrl' | 'cnot-target' | 'measure' | 'error' | 'correction' }) {
  const styles: Record<string, string> = {
    default: 'bg-secondary text-foreground border-border',
    hadamard: 'bg-primary text-primary-foreground border-primary',
    'cnot-ctrl': 'text-accent',
    'cnot-target': 'text-accent',
    measure: 'bg-secondary text-foreground border-border',
    error: 'bg-destructive text-accent-foreground border-destructive',
    correction: 'bg-success text-success-foreground border-success',
  };

  if (variant === 'cnot-ctrl') {
    return <span className="text-accent font-mono text-xs font-medium">●</span>;
  }
  if (variant === 'cnot-target') {
    return <span className="text-accent font-mono text-xs font-medium">⊕</span>;
  }

  return (
    <span className={`inline-flex items-center justify-center px-1.5 py-0.5 rounded-sm border text-[10px] font-mono font-medium ${styles[variant]}`}>
      {label}
    </span>
  );
}

function QubitLine({ qubit, index, groupIndex }: { qubit: QubitState; index: number; groupIndex: number }) {
  const isGroupLeader = index % 3 === 0;
  const subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈'];

  return (
    <div className="flex items-center gap-1 font-mono text-xs text-muted-foreground h-6">
      <span className="w-6 text-right text-foreground font-medium">q{subscripts[index]}</span>
      <span className="text-border">───</span>
      {isGroupLeader ? (
        <GateBox label="H" variant="hadamard" />
      ) : (
        <span className="w-5 text-center text-border">·</span>
      )}
      <span className="text-border">──</span>
      {isGroupLeader ? (
        <GateBox label="" variant="cnot-ctrl" />
      ) : index % 3 === 1 ? (
        <GateBox label="" variant="cnot-target" />
      ) : (
        <span className="w-3 text-center text-border">·</span>
      )}
      <span className="text-border">──</span>
      {isGroupLeader ? (
        <GateBox label="" variant="cnot-ctrl" />
      ) : index % 3 === 2 ? (
        <GateBox label="" variant="cnot-target" />
      ) : (
        <span className="w-3 text-center text-border">·</span>
      )}
      <span className="text-border">──</span>
      {qubit.error !== 'I' ? (
        <GateBox label={qubit.error} variant="error" />
      ) : (
        <span className="w-5 text-center text-border">──</span>
      )}
      <span className="text-border">──</span>
      {qubit.correction !== 'I' ? (
        <GateBox label={qubit.correction} variant="correction" />
      ) : (
        <span className="w-5 text-center text-border">──</span>
      )}
      <span className="text-border">──</span>
      <GateBox label="M" variant="measure" />
    </div>
  );
}

export default function ShorCircuit({ qubits, phase }: ShorCircuitProps) {
  return (
    <div className="bg-card border-2 border-border rounded-lg p-6">
      <h2 className="font-serif text-lg font-semibold text-foreground mb-4">
        §1 — Circuit Diagram
      </h2>
      <PhaseIndicator phase={phase} />
      <div className="bg-secondary/50 border border-border rounded-md p-4 space-y-0">
        {[0, 1, 2].map((group) => (
          <div key={group}>
            {group > 0 && <div className="border-t border-dashed border-border my-2" />}
            <div className="text-[10px] text-muted-foreground font-mono mb-1">Group {group}</div>
            {[0, 1, 2].map((qi) => {
              const idx = group * 3 + qi;
              return <QubitLine key={idx} qubit={qubits[idx]} index={idx} groupIndex={group} />;
            })}
          </div>
        ))}
      </div>
      <div className="flex gap-4 mt-4 text-[10px] font-mono text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-destructive" /> Error
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-success" /> Correction
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-muted" /> Idle
        </span>
      </div>
    </div>
  );
}
