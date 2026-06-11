"""
Probabilistic Policy Reuse (PPR)
==================================
Three components:

  1. PolicyLibrary   — stores frozen snapshots of trained DDQN agents
  2. ReuseScheduler  — manages ψ (reuse probability) over training
  3. PPRAgent        — wraps DDQNAgent, queries library with prob ψ

How it fits into training
--------------------------

  Stage 1: train DDQN at d=3  →  save snapshot into PolicyLibrary
  Stage 2: train DDQN at d=5  →  wrap with PPRAgent using d=3 library

  At every step in Stage 2:

    sample u ~ Uniform(0, 1)

    if u < ψ:
        action = library_policy.argmax_Q(state)   ← reuse d=3 knowledge
    else:
        action = ddqn_agent.select_action(state)  ← own ε-greedy policy

    ψ decays linearly:  ψ_0 → ψ_min  over decay_steps

Key property
------------
  Reused transitions are still stored in the replay buffer and used
  for training — the agent learns FROM the library's behaviour as
  well as its own.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional

from reinforcement.surface.Dueling_cnn import DuelingCNN
from reinforcement.surface.ddqn_agent  import DDQNAgent


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1 — Policy Entry (one frozen policy)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyEntry:
    """
    A single frozen policy stored in the library.

    Holds a snapshot of a trained DuelingCNN's weights.
    The network is loaded lazily on first use.

    Fields
    ------
    name         : human-readable label e.g. "d3_p001"
    state_dict   : frozen network weights (deep copy)
    distance     : code distance it was trained on
    noise        : noise rate it was trained on
    success_rate : benchmark success rate (used for library selection)
    in_channels  : network input channels  (= 2k)
    grid_size    : network grid size       (= 2d+1)
    n_actions    : network output size     (= 2d²+1)
    """
    name:         str
    state_dict:   dict
    distance:     int
    noise:        float
    success_rate: float
    in_channels:  int
    grid_size:    int
    n_actions:    int
    metadata:     dict = field(default_factory=dict)

    # Lazy-loaded network (not serialised)
    _net: object = field(default=None, init=False, repr=False)

    def get_network(self, device: str = "cpu") -> nn.Module:
        """Load the frozen network on first call, cache it."""
        if self._net is None:
            net = DuelingCNN(self.in_channels, self.grid_size, self.n_actions)
            net.load_state_dict(self.state_dict)
            net.eval()
            net.to(torch.device(device))
            self._net = net
        return self._net

    def get_q_values(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Compute Q-values for all actions from this frozen policy.

        If the state spatial size doesn't match this policy's grid_size
        (e.g. querying a d=3 policy with a d=5 state), we centre-crop
        or zero-pad automatically.

        Parameters
        ----------
        state  : np.ndarray  shape (C, H, W)
        device : str

        Returns
        -------
        q_vals : np.ndarray  shape (n_actions,)
        """
        net = self.get_network(device)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1,C,H,W)
            state_t = self._adapt_spatial(state_t)
            q_vals  = net(state_t).squeeze(0).cpu().numpy()              # (n_actions,)
        return q_vals

    def best_action(self, state: np.ndarray, device: str = "cpu") -> int:
        """Return greedy action from this frozen policy."""
        return int(np.argmax(self.get_q_values(state, device)))

    def _adapt_spatial(self, state_t: torch.Tensor) -> torch.Tensor:
        """
        Adapt state spatial dimensions to match this policy's grid_size.
        Uses centre-crop if state is larger, zero-pad if smaller.
        """
        _, C, H, W = state_t.shape
        target = self.grid_size
        if H == target and W == target:
            return state_t
        elif H > target:
            # Centre crop
            sh = (H - target) // 2
            sw = (W - target) // 2
            return state_t[:, :, sh:sh+target, sw:sw+target]
        else:
            # Zero pad
            ph = (target - H) // 2
            pw = (target - W) // 2
            return nn.functional.pad(state_t, (pw, pw, ph, ph))


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2 — Policy Library
# ══════════════════════════════════════════════════════════════════════════════

class PolicyLibrary:
    """
    Stores and retrieves frozen policies from previous training stages.

    Usage
    -----
    # After training d=3 agent:
    library = PolicyLibrary()
    library.add_from_agent(agent_d3, name="d3", distance=3, noise=0.01,
                           success_rate=0.85)

    # When training d=5 agent:
    ppr = PPRAgent(agent_d5, library, scheduler)
    """

    def __init__(self):
        self.policies: List[PolicyEntry] = []

    # ── Adding policies ───────────────────────────────────────────────────────

    def add_from_agent(
        self,
        agent:        DDQNAgent,
        name:         str,
        distance:     int,
        noise:        float,
        success_rate: float,
        metadata:     dict = None,
    ) -> None:
        """
        Snapshot a trained DDQNAgent and add it to the library.

        Parameters
        ----------
        agent        : trained DDQNAgent
        name         : label e.g. "d3_noise001"
        distance     : code distance the agent was trained on
        noise        : noise rate
        success_rate : fraction of episodes agent solved successfully
        """
        # Infer grid_size from the network's advantage stream input size
        # flat_size = 64 * grid_size²  →  grid_size = sqrt(flat_size / 64)
        flat_size = agent.online_net.advantage_stream[0].in_features
        grid_size = int(np.sqrt(flat_size // 64))
        in_channels = agent.online_net.conv[0].in_channels

        entry = PolicyEntry(
            name         = name,
            state_dict   = agent.get_policy_snapshot(),   # deep copy
            distance     = distance,
            noise        = noise,
            success_rate = success_rate,
            in_channels  = in_channels,
            grid_size    = grid_size,
            n_actions    = agent.n_actions,
            metadata     = metadata or {},
        )
        self.policies.append(entry)
        print(f"[Library] Added '{name}'  d={distance}  "
              f"noise={noise:.4f}  success={success_rate:.3f}")

    def add_from_checkpoint(
        self,
        path:         str,
        name:         str,
        distance:     int,
        noise:        float,
        success_rate: float,
        in_channels:  int,
        grid_size:    int,
        n_actions:    int,
        device:       str  = "cpu",
        metadata:     dict = None,
    ) -> None:
        """Load a policy snapshot from a .pt checkpoint file."""
        ckpt  = torch.load(path, map_location=device)
        entry = PolicyEntry(
            name         = name,
            state_dict   = ckpt["online_net"],
            distance     = distance,
            noise        = noise,
            success_rate = success_rate,
            in_channels  = in_channels,
            grid_size    = grid_size,
            n_actions    = n_actions,
            metadata     = metadata or {},
        )
        self.policies.append(entry)
        print(f"[Library] Loaded '{name}' from {path}")

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_best_action(self, state: np.ndarray, device: str = "cpu") -> int:
        """
        Query the best policy in the library for an action.

        Selects the policy with the highest success_rate, then returns
        its greedy action for the given state.
        """
        best_policy = max(self.policies, key=lambda p: p.success_rate)
        return best_policy.best_action(state, device)

    def get_best_policy(self) -> Optional[PolicyEntry]:
        """Return the policy with the highest success_rate."""
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p.success_rate)

    def get_all_policies(self) -> List[PolicyEntry]:
        return self.policies

    def __len__(self) -> int:
        return len(self.policies)

    def summary(self) -> None:
        print(f"\n{'═'*50}")
        print(f"  Policy Library  ({len(self.policies)} policies)")
        print(f"{'═'*50}")
        for p in self.policies:
            print(f"  [{p.name}]  d={p.distance}  "
                  f"noise={p.noise:.4f}  success={p.success_rate:.3f}  "
                  f"actions={p.n_actions}")
        print(f"{'═'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3 — Reuse Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class ReuseScheduler:
    """
    Manages the reuse probability ψ over training.

    ψ decays linearly from ψ_start → ψ_min over decay_steps.

    Intuition
    ---------
    Early in training the new agent knows nothing, so we reuse the
    library policy often (high ψ).  As the agent improves, we rely
    on the library less and less (ψ → ψ_min).

    Parameters
    ----------
    psi_start   : initial reuse probability  (default 0.5)
    psi_min     : minimum reuse probability  (default 0.05)
    decay_steps : steps to decay from start → min
    """

    def __init__(
        self,
        psi_start:   float = 0.5,
        psi_min:     float = 0.05,
        decay_steps: int   = 50_000,
    ):
        self.psi_start   = psi_start
        self.psi_min     = psi_min
        self.decay_steps = decay_steps

        self.psi     = psi_start    # current reuse probability
        self._step   = 0            # number of decay steps taken

    def step(self) -> None:
        """Advance the scheduler by one step (call once per env step)."""
        self._step += 1
        self.psi = max(
            self.psi_min,
            self.psi_start - (self.psi_start - self.psi_min)
            * (self._step / self.decay_steps)
        )

    def should_reuse(self) -> bool:
        """
        Sample whether to reuse the library policy this step.
        Returns True with probability ψ.
        """
        return np.random.random() < self.psi

    @property
    def current(self) -> float:
        return self.psi

    def summary(self) -> None:
        print(f"\n{'═'*50}")
        print(f"  Reuse Scheduler")
        print(f"{'═'*50}")
        print(f"  psi_start   : {self.psi_start}")
        print(f"  psi_min     : {self.psi_min}")
        print(f"  decay_steps : {self.decay_steps:,}")
        print(f"  current ψ   : {self.psi:.4f}")
        print(f"  steps done  : {self._step:,}")
        print(f"{'═'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4 — PPR Agent (the wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class PPRAgent:
    """
    Wraps a DDQNAgent with Probabilistic Policy Reuse.

    Action selection at each step:

        u ~ Uniform(0,1)
        if u < ψ  and  library is not empty:
            action ← library.get_best_action(state)   ← REUSE
        else:
            action ← ddqn_agent.select_action(state)  ← OWN POLICY

    The reused transition is still pushed to the replay buffer
    and used for training — the agent learns from library behaviour.

    Parameters
    ----------
    agent     : DDQNAgent  the agent being trained
    library   : PolicyLibrary  frozen policies from previous stages
    scheduler : ReuseScheduler  manages ψ decay
    device    : str
    """

    def __init__(
        self,
        agent:     DDQNAgent,
        library:   PolicyLibrary,
        scheduler: ReuseScheduler,
        device:    str = "cpu",
    ):
        self.agent     = agent
        self.library   = library
        self.scheduler = scheduler
        self.device    = device

        # ── Statistics ────────────────────────────────────────────────────────
        self._total_steps  = 0
        self._reuse_steps  = 0
        self._own_steps    = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(
        self,
        state:  np.ndarray,
        greedy: bool = False,
    ):
        """
        Select an action using PPR.

        Parameters
        ----------
        state  : np.ndarray  shape (C, H, W)
        greedy : bool        if True → always use own policy (eval mode)

        Returns
        -------
        action : int
        source : str   'library' or 'own'
        """
        self._total_steps += 1

        # In greedy/eval mode always use own policy
        if greedy:
            action = self.agent.select_action(state, greedy=True)
            self._own_steps += 1
            return action, "own"

        # PPR decision: reuse or own?
        if self.scheduler.should_reuse() and len(self.library) > 0:
            action = self.library.get_best_action(state, self.device)
            self._reuse_steps += 1
            source = "library"
        else:
            action = self.agent.select_action(state)
            self._own_steps += 1
            source = "own"

        # Advance scheduler after each step
        self.scheduler.step()

        return action, source

    # ── Delegate to underlying agent ──────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Store transition in replay buffer.
        Called regardless of whether action came from library or own policy.
        """
        self.agent.push(state, action, reward, next_state, done)

    def train_step(self):
        """Forward training to the underlying DDQN agent."""
        return self.agent.train_step()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self.agent.epsilon

    @property
    def psi(self) -> float:
        return self.scheduler.current

    @property
    def buffer(self):
        return self.agent.buffer

    # ── Statistics ────────────────────────────────────────────────────────────

    def reuse_ratio(self) -> float:
        """Fraction of steps that used the library policy."""
        if self._total_steps == 0:
            return 0.0
        return self._reuse_steps / self._total_steps

    def stats(self) -> dict:
        return {
            "total_steps":  self._total_steps,
            "reuse_steps":  self._reuse_steps,
            "own_steps":    self._own_steps,
            "reuse_ratio":  round(self.reuse_ratio(), 4),
            "psi":          round(self.psi, 4),
            "epsilon":      round(self.epsilon, 4),
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        print(f"\n{'═'*50}")
        print(f"  PPR Agent")
        print(f"{'═'*50}")
        print(f"  ψ (reuse prob)  : {self.psi:.4f}")
        print(f"  ε (explore)     : {self.epsilon:.4f}")
        print(f"  total steps     : {self._total_steps:,}")
        print(f"  reuse steps     : {self._reuse_steps:,}  "
              f"({self.reuse_ratio():.1%})")
        print(f"  own steps       : {self._own_steps:,}")
        print(f"  library size    : {len(self.library)}")
        print(f"  buffer size     : {len(self.agent.buffer):,}")
        print(f"{'═'*50}\n")