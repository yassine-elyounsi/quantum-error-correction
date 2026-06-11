import { useState, useCallback } from 'react';
import ShorCircuit from '@/components/ShorCircuit';
import SyndromePanel from '@/components/SyndromePanel';
import ControlsPanel from '@/components/ControlsPanel';
import ResultPanel from '@/components/ResultPanel';
import { createInitialState, runShot, type SimulationState, type Phase } from '@/lib/shor-simulation';

const PHASE_SEQUENCE: Phase[] = ['encode', 'noise', 'syndrome', 'rl_decode', 'correct', 'measure'];
const PHASE_DELAY = 400;

export default function Index() {
  const [state, setState] = useState<SimulationState>(createInitialState());
  const [isRunning, setIsRunning] = useState(false);

  const animateShot = useCallback((noiseP: number, prevState: SimulationState) => {
    return new Promise<SimulationState>((resolve) => {
      const result = runShot(noiseP);
      let phaseIdx = 0;

      const tick = () => {
        const phase = PHASE_SEQUENCE[phaseIdx];
        setState((s) => ({
          ...s,
          phase,
          qubits: phaseIdx >= 1 ? result.qubits : s.qubits,
          syndrome: phaseIdx >= 2 ? result.syndrome : s.syndrome,
          agentAction: phaseIdx >= 3 ? result.agentAction : 27,
          logicalPreserved: phaseIdx >= 5 ? result.logicalPreserved : null,
        }));

        phaseIdx++;
        if (phaseIdx < PHASE_SEQUENCE.length) {
          setTimeout(tick, PHASE_DELAY);
        } else {
          const newState: SimulationState = {
            phase: 'measure',
            qubits: result.qubits,
            syndrome: result.syndrome,
            agentAction: result.agentAction,
            logicalPreserved: result.logicalPreserved,
            noiseP,
            shots: prevState.shots + 1,
            successes: prevState.successes + (result.logicalPreserved ? 1 : 0),
            failures: prevState.failures + (result.logicalPreserved ? 0 : 1),
          };
          setState(newState);
          resolve(newState);
        }
      };

      tick();
    });
  }, []);

  const handleRunShot = useCallback(async () => {
    setIsRunning(true);
    await animateShot(state.noiseP, state);
    setIsRunning(false);
  }, [state, animateShot]);

  const handleRun100 = useCallback(async () => {
    setIsRunning(true);
    let current = state;
    for (let i = 0; i < 100; i++) {
      const result = runShot(current.noiseP);
      current = {
        ...current,
        phase: 'measure',
        qubits: result.qubits,
        syndrome: result.syndrome,
        agentAction: result.agentAction,
        logicalPreserved: result.logicalPreserved,
        shots: current.shots + 1,
        successes: current.successes + (result.logicalPreserved ? 1 : 0),
        failures: current.failures + (result.logicalPreserved ? 0 : 1),
      };
    }
    setState(current);
    setIsRunning(false);
  }, [state]);

  const handleReset = useCallback(() => {
    setState(createInitialState());
  }, []);

  const handleNoiseChange = useCallback((p: number) => {
    setState((s) => ({ ...s, noiseP: p }));
  }, []);

  return (
    <div className="min-h-screen bg-background p-6 md:p-10">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="mb-8 border-b-2 border-foreground pb-5">
          <h1 className="font-serif text-2xl md:text-3xl font-bold text-foreground">
            Shor Code — Reinforcement Learning Decoder
          </h1>
          <p className="text-sm text-muted-foreground mt-2 font-mono">
            PPO Agent · Transfer Learning · 9-qubit · Bernoulli depolarizing noise
          </p>
        </header>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-6">
          {/* Left column */}
          <div className="space-y-6">
            <ShorCircuit qubits={state.qubits} phase={state.phase} />
            <SyndromePanel sx={state.syndrome.sx} sz={state.syndrome.sz} />
          </div>

          {/* Right column */}
          <div className="space-y-4">
            <ControlsPanel
              noiseP={state.noiseP}
              onNoiseChange={handleNoiseChange}
              onRunShot={handleRunShot}
              onRun100={handleRun100}
              onReset={handleReset}
              isRunning={isRunning}
            />
            <ResultPanel
              logicalPreserved={state.logicalPreserved}
              agentAction={state.agentAction}
              shots={state.shots}
              successes={state.successes}
              failures={state.failures}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
