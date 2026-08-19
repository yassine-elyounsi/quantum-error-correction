# Quantum Error Correction with Reinforcement Learning

A research project exploring **reinforcement learning-based decoders for Quantum
Error Correction (QEC)**. The work progresses from foundational QEC codes
(repetition and Shor codes) to a full RL decoding pipeline on the **surface
code** — first at distance 3 with a discrete-task decoder, then at distance 5,
including a version trained under **continuous, per-qubit noise** with a
different reward structure and architecture.

The goal throughout is the same one motivating QEC research generally: design
agents that can infer the right correction from noisy syndrome measurements,
improving error-correction strategies for future fault-tolerant quantum
computers.

---

## Project structure & exploration path

The repository is organized to follow the actual learning path of the project —
starting from simpler, well-understood codes and building up to the harder,
more realistic surface-code setting.

### 1. Repetition Code & Shor Code — foundations

**`notebooks/Repetition_Code.ipynb`**, **`notebooks/Shor_Code.ipynb`**

These notebooks cover the foundational QEC background before any RL is
introduced: the repetition code as the simplest possible error-correcting code
(protecting against a single type of noise), and the Shor code as the first
code capable of protecting against arbitrary single-qubit errors by combining
bit-flip and phase-flip protection. They establish the stabilizer formalism,
syndrome extraction, and correction logic that the later, more complex surface
code experiments build on.

### 2. Surface Code, distance 3 — first RL decoder

**`notebooks/surface_code.ipynb`**, **`reinforcement/surface/`**

The surface code is introduced at **distance 3** (the smallest non-trivial
instance), using **Stim** for syndrome extraction and **PyMatching** (Minimum
Weight Perfect Matching, MWPM) as the classical baseline decoder.

- `reinforcement/surface/ddqn_agent.py` — a **Dueling Double DQN (DDQN)** agent
  with a CNN-based architecture that takes the syndrome measurements as input
  and learns to output corrections directly, trained to beat the MWPM baseline.
- The environment is built with **Gymnasium**, and earlier iterations of this
  stage also explored MLP and LSTM decoder architectures before settling on the
  CNN-based dueling design.

This is the stage where the RL decoder is shown to outperform the classical
MWPM baseline on the distance-3 code.

### 3. Surface Code, distance 5 — scaling up

**`reinforcement/surface5_realistic/`** (discrete variant), **`checkpoints_d5_continuous*/`**

Scaling the same problem to **distance 5** increases the syndrome space
substantially. This stage exposed a fundamental challenge: **syndrome
ambiguity** — at higher distances, different physical error configurations can
produce identical or near-identical syndromes, making the decoding problem
inherently harder and limiting what any decoder (classical or learned) can
resolve from the syndrome alone.

### 4. Surface Code, distance 5 — continuous, realistic noise

**`reinforcement/surface5_realistic/`**, **`src/environments/surface5_realistic_env.py`**

This is the most advanced setting in the project. Unlike the distance-3 and
early distance-5 experiments — which treat error correction as a **discrete,
single-shot task** (one round of noise, one correction) — this version models
**continuous noise**: errors accumulate over time, qubit by qubit, and the
agent must act repeatedly as the system evolves.

Key differences from the earlier stages:

- **Continuous noise process**, rather than a single discrete error round —
  the environment (`surface5_realistic_env.py`) models noise accumulating over
  time on each qubit individually.
- **Per-qubit reward structure**, rather than a single episode-level reward —
  the agent receives feedback tied to individual qubits, encouraging it to
  learn a more fine-grained correction policy rather than a single global
  action.
- **A different agent architecture** (`Ddqn_agent_multi.py`) adapted to this
  continuous, per-qubit setting rather than the single-shot CNN decoder used
  at distance 3.
- Training is run via `train_d5_realistic.py`, with an additional experimental
  variant, `train_d5_realistic_ppr.py`, exploring a PPR-based training
  approach.

### 5. Encoding agent

**`Encoders/`**

A separate agent focused on the **encoding** side of the QEC pipeline, based on
the approach in Olle et al. (2024). It includes a **circuit pruner**
(`circuit_pruner.py`) that reduces the number of gates in the resulting
encoding circuits by **20–33%**, and a `meta_env.py` environment used to train
the encoding agent (`Encoding_agent.py`).

### 6. Dashboard

**`dashboard/`**

A Streamlit-based interactive dashboard (`App.py`, `Demo_env.py`,
`Surface_render.py`) for visualizing and demoing the trained decoders on the
surface code environment.

### 7. Benchmarking

**`benchmark/`**, **`benchmark_results/`**

Scripts and results comparing the trained RL decoders against the classical
MWPM baseline across the different code distances and settings explored above.

---

## Tech stack

Python · Stim · PyMatching · PyTorch · Stable-Baselines3 · Gymnasium ·
Streamlit · Weights & Biases

## Repository structure

```
quantum-error-correction/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Repetition_Code.ipynb
│   ├── Shor_Code.ipynb
│   ├── surface_code.ipynb
│   └── rlsurface.ipynb
├── reinforcement/
│   ├── surface/                   # distance-3 DDQN decoder
│   └── surface5_realistic/        # distance-5, continuous noise, per-qubit reward
├── src/
│   └── environments/
│       └── surface5_realistic_env.py
├── Encoders/                      # encoding agent + circuit pruner
├── decoders/
├── dashboard/                     # Streamlit visualization
├── benchmark/
└── benchmark_results/
```

## Running it

```bash
git clone https://github.com/yassine-elyounsi/quantum-error-correction.git
cd quantum-error-correction
python -m venv venv
venv\Scripts\activate      # or source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

Explore the foundational notebooks first (`Repetition_Code.ipynb`,
`Shor_Code.ipynb`, `surface_code.ipynb`), then run training scripts under
`reinforcement/` for the distance-3 and distance-5 decoders.

> Training runs are logged with Weights & Biases — set your own W&B API key
> (e.g. via a local `.env` file, which is gitignored) before running the
> training scripts.
