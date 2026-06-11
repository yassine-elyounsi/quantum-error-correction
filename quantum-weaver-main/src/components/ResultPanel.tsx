import { decodeAction } from '@/lib/shor-simulation';

interface ResultPanelProps {
  logicalPreserved: boolean | null;
  agentAction: number;
  shots: number;
  successes: number;
  failures: number;
}

export default function ResultPanel({
  logicalPreserved,
  agentAction,
  shots,
  successes,
  failures,
}: ResultPanelProps) {
  const ler = shots > 0 ? ((failures / shots) * 100).toFixed(1) : '—';
  const decoded = decodeAction(agentAction);
  const actionLabel = decoded
    ? `${decoded.pauli} on q${decoded.qubit}`
    : 'Identity (no correction)';

  return (
    <div className="space-y-4">
      {/* Result indicator */}
      <div
        className={`border-2 rounded-lg p-5 text-center transition-colors ${
          logicalPreserved === null
            ? 'bg-secondary border-border'
            : logicalPreserved
            ? 'bg-success-bg border-success-border'
            : 'bg-syndrome-active-bg border-syndrome-active-border'
        }`}
      >
        <div className="text-2xl mb-1">
          {logicalPreserved === null ? '—' : logicalPreserved ? '✓' : '✗'}
        </div>
        <div
          className={`text-sm font-semibold font-serif ${
            logicalPreserved === null
              ? 'text-muted-foreground'
              : logicalPreserved
              ? 'text-success'
              : 'text-accent'
          }`}
        >
          {logicalPreserved === null
            ? 'AWAITING SHOT'
            : logicalPreserved
            ? 'QUBIT PRESERVED'
            : 'LOGICAL ERROR'}
        </div>
        <div className="text-[11px] text-muted-foreground mt-1 font-mono">
          {logicalPreserved === null ? '' : `agent action: ${actionLabel}`}
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2">
        <StatBox label="SHOTS" value={shots.toString()} />
        <StatBox label="LER" value={`${ler}%`} variant={failures > 0 ? 'accent' : 'default'} />
        <StatBox label="SUCCESSES" value={successes.toString()} variant="success" />
        <StatBox label="FAILURES" value={failures.toString()} variant={failures > 0 ? 'accent' : 'default'} />
      </div>

      {/* Agent info */}
      <div className="bg-card border-2 border-border rounded-lg p-4">
        <h4 className="font-serif text-sm font-semibold mb-2 text-foreground">Agent Info</h4>
        <div className="space-y-1.5 text-xs">
          <InfoRow label="Algorithm" value="PPO" />
          <InfoRow label="Network" value="MLP [64, 64]" />
          <InfoRow label="Obs space" value="MultiBinary(8)" />
          <InfoRow label="Action space" value="Discrete(28)" />
          <InfoRow label="Timesteps" value="200,704" />
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value, variant = 'default' }: { label: string; value: string; variant?: 'default' | 'accent' | 'success' }) {
  const textColor = variant === 'accent' ? 'text-accent' : variant === 'success' ? 'text-success' : 'text-foreground';
  return (
    <div className="bg-card border-2 border-border rounded-lg p-3 text-center">
      <div className={`text-xl font-bold font-mono ${textColor}`}>{value}</div>
      <div className="text-[10px] text-muted-foreground tracking-wider">{label}</div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono font-medium text-foreground">{value}</span>
    </div>
  );
}
