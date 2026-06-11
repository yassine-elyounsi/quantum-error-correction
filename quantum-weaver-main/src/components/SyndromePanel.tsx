interface SyndromePanelProps {
  sx: number[];
  sz: number[];
}

function SyndromeBit({ value, label }: { value: number; label: string }) {
  const active = value === 1;
  return (
    <div
      className={`flex flex-col items-center justify-center px-3 py-2 rounded border-2 font-mono text-sm font-medium transition-colors ${
        active
          ? 'bg-syndrome-active-bg border-syndrome-active-border text-syndrome-active-text'
          : 'bg-syndrome-idle-bg border-syndrome-idle-border text-foreground'
      }`}
    >
      <span className="text-[9px] text-muted-foreground mb-0.5">{label}</span>
      <span className="text-base">{value}</span>
    </div>
  );
}

export default function SyndromePanel({ sx, sz }: SyndromePanelProps) {
  return (
    <div className="space-y-4">
      <div className="bg-card border-2 border-border rounded-lg p-5">
        <h3 className="font-serif text-base font-semibold mb-1 text-foreground">
          §2 — X-Syndrome (6 bits)
        </h3>
        <p className="text-xs text-muted-foreground mb-3 font-mono">
          detects bit-flip errors
        </p>
        <div className="flex gap-2">
          {sx.map((v, i) => (
            <SyndromeBit key={i} value={v} label={`sx[${i}]`} />
          ))}
        </div>
      </div>
      <div className="bg-card border-2 border-border rounded-lg p-5">
        <h3 className="font-serif text-base font-semibold mb-1 text-foreground">
          §3 — Z-Syndrome (2 bits)
        </h3>
        <p className="text-xs text-muted-foreground mb-3 font-mono">
          detects phase-flip errors
        </p>
        <div className="flex gap-2">
          {sz.map((v, i) => (
            <SyndromeBit key={i} value={v} label={`sz[${i}]`} />
          ))}
        </div>
      </div>
    </div>
  );
}
