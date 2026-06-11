import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

interface ControlsPanelProps {
  noiseP: number;
  onNoiseChange: (p: number) => void;
  onRunShot: () => void;
  onRun100: () => void;
  onReset: () => void;
  isRunning: boolean;
}

export default function ControlsPanel({
  noiseP,
  onNoiseChange,
  onRunShot,
  onRun100,
  onReset,
  isRunning,
}: ControlsPanelProps) {
  return (
    <div className="bg-card border-2 border-border rounded-lg p-5 space-y-5">
      <h3 className="font-serif text-base font-semibold text-foreground">Controls</h3>

      <div>
        <div className="flex justify-between text-sm mb-2">
          <span className="text-muted-foreground">Noise p</span>
          <span className="font-mono font-medium text-accent">
            {(noiseP * 100).toFixed(1)}%
          </span>
        </div>
        <Slider
          value={[noiseP * 100]}
          min={0}
          max={50}
          step={1}
          onValueChange={([v]) => onNoiseChange(v / 100)}
          className="accent-accent"
        />
      </div>

      <Button
        onClick={onRunShot}
        disabled={isRunning}
        className="w-full bg-primary text-primary-foreground hover:bg-primary/90 font-medium"
      >
        ▶ Run Shot
      </Button>

      <Button
        onClick={onRun100}
        disabled={isRunning}
        variant="outline"
        className="w-full"
      >
        ↻ Run 100 Shots
      </Button>

      <Button
        onClick={onReset}
        variant="outline"
        className="w-full"
      >
        ↺ Reset
      </Button>
    </div>
  );
}
