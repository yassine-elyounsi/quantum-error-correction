"""
DDQN Agent for Surface Code Decoding
======================================

Combines:
  - Dueling CNN       → computes Q(s,a) = V(s) + A(s,a) - mean(A)
  - Replay Buffer     → stores and samples transitions
  - Double DQN update → online net selects, target net evaluates
  - Epsilon-greedy    → exploration that decays over training

Training step (Double DQN):
  1. Sample mini-batch from replay buffer
  2. Online net  → picks best next action:  a* = argmax_a Q_online(s', a)
  3. Target net  → evaluates that action:   y  = r + γ · Q_target(s', a*)
  4. Loss        → Huber( Q_online(s,a) - y )
  5. Backprop    → update online net weights
  6. Every C steps → copy online weights → target net
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Optional

from reinforcement.surface.Dueling_cnn import DuelingCNN
from reinforcement.surface.replay_buffer import ReplayBuffer


class DDQNAgent:
    """
    Double DQN Agent with Dueling CNN architecture.

    Parameters
    ----------
    in_channels     : number of input channels        (= 2k = 6  for d=3)
    grid_size       : spatial size of syndrome grid   (= 2d+1 = 7 for d=3)
    n_actions       : size of discrete action space   (= 19       for d=3)
    lr              : learning rate                   (default 1e-4)
    gamma           : discount factor                 (default 0.99)
    epsilon_start   : initial exploration rate        (default 1.0)
    epsilon_end     : minimum exploration rate        (default 0.05)
    epsilon_decay   : number of steps to decay ε      (default 50_000)
    batch_size      : mini-batch size                 (default 64)
    target_update   : steps between target net syncs  (default 1_000)
    buffer_capacity : replay buffer size              (default 50_000)
    device          : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        in_channels:     int,
        grid_size:       int,
        n_actions:       int,
        lr:              float = 1e-4,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   int   = 50_000,
        batch_size:      int   = 64,
        target_update:   int   = 1_000,
        buffer_capacity: int   = 50_000,
        device:          str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        # ── Exploration schedule ──────────────────────────────────────────────
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._steps_done   = 0      # total steps (drives decay + target sync)

        # ── Two networks ──────────────────────────────────────────────────────
        # Online network : trains every step, makes decisions
        self.online_net = DuelingCNN(in_channels, grid_size, n_actions).to(self.device)

        # Target network : frozen copy, only synced every target_update steps
        self.target_net = DuelingCNN(in_channels, grid_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()      # target net is NEVER trained directly

        # ── Optimiser & loss ──────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss

        # ── Replay buffer ─────────────────────────────────────────────────────
        state_shape = (in_channels, grid_size, grid_size)
        self.buffer = ReplayBuffer(
            capacity    = buffer_capacity,
            state_shape = state_shape,
            device      = device,
        )

        # ── Training statistics ───────────────────────────────────────────────
        self.losses      = []       # loss value at each train step
        self.q_means     = []       # mean Q-value at each train step
        self._train_steps = 0       # counts only training steps (not action steps)

    # ═════════════════════════════════════════════════════════════════════════
    # ACTION SELECTION
    # ═════════════════════════════════════════════════════════════════════════

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        With probability ε     → random action  (explore)
        With probability (1-ε) → best Q action  (exploit)

        ε decays linearly from epsilon_start → epsilon_end
        over epsilon_decay steps.

        Parameters
        ----------
        state  : np.ndarray  shape (in_channels, grid_size, grid_size)
        greedy : bool        if True → always exploit  (for evaluation)

        Returns
        -------
        action : int
        """
        # Decay epsilon
        self._steps_done += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (self._steps_done / self.epsilon_decay) * (1.0 - self.epsilon_end)
        )

        # Explore: random action
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        # Exploit: ask online network
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t)           # (1, n_actions)
        self.online_net.train()

        return int(q_vals.argmax(dim=1).item())

    # ═════════════════════════════════════════════════════════════════════════
    # STORE TRANSITION
    # ═════════════════════════════════════════════════════════════════════════

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store one transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ═════════════════════════════════════════════════════════════════════════
    # TRAINING STEP  ← the core DDQN update
    # ═════════════════════════════════════════════════════════════════════════

    def train_step(self) -> Optional[float]:
        """
        One Double DQN gradient update.

        Steps
        -----
        1. Sample mini-batch from replay buffer
        2. Compute current Q-values:    Q_online(s, a)
        3. Compute DDQN target:         y = r + γ · Q_target(s', a*)
                                        where a* = argmax_a Q_online(s', a)
        4. Compute Huber loss:          L = SmoothL1( Q_online(s,a) - y )
        5. Backprop + gradient clip
        6. Sync target net if needed

        Returns
        -------
        loss : float or None
        """
        if not self.buffer.is_ready_for(self.batch_size):
            return None

        # ── 1. Sample ─────────────────────────────────────────────────────────
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        #   states      : (64, 6, 7, 7)
        #   actions     : (64,)
        #   rewards     : (64,)
        #   next_states : (64, 6, 7, 7)
        #   dones       : (64,)   1.0=terminal  0.0=not terminal

        # ── 2. Current Q-values ───────────────────────────────────────────────
        # Get Q-values for ALL actions, then pick the one actually taken
        q_all     = self.online_net(states)              # (64, 19)
        q_current = q_all.gather(                        # (64,)
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        # ── 3. DDQN Target ────────────────────────────────────────────────────
        with torch.no_grad():

            # Online net picks the best action in next state
            # "which action looks best right now?"
            q_next_online = self.online_net(next_states)             # (64, 19)
            best_actions  = q_next_online.argmax(dim=1, keepdim=True) # (64, 1)

            # Target net evaluates that action
            # "how good is that action really?" (stable estimate)
            q_next_target = self.target_net(next_states)             # (64, 19)
            q_next_best   = q_next_target.gather(
                dim=1, index=best_actions
            ).squeeze(1)                                              # (64,)

            # y = r + γ · Q_target(s', a*) · (1 - done)
            # at terminal states done=1.0 so future term disappears
            td_target = rewards + self.gamma * q_next_best * (1.0 - dones)

        # ── 4. Loss ───────────────────────────────────────────────────────────
        loss = self.loss_fn(q_current, td_target)

        # ── 5. Backprop ───────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── 6. Sync target net ────────────────────────────────────────────────
        self._train_steps += 1
        if self._train_steps % self.target_update == 0:
            self._sync_target()
            
        # ── 7. Record stats ───────────────────────────────────────────────────
        loss_val = loss.item()
        self.losses.append(loss_val)
        self.q_means.append(q_current.mean().item())

        return loss_val

    # ═════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _sync_target(self) -> None:
        """Hard copy: online weights → target weights."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ═════════════════════════════════════════════════════════════════════════
    # PPR INTERFACE  ← these methods are called by the PPR layer
    # ═════════════════════════════════════════════════════════════════════════

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Return Q-values for all actions given a state.
        PPR uses this to query and compare policies.

        Returns
        -------
        q_vals : np.ndarray  shape (n_actions,)
        """
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t).squeeze(0).cpu().numpy()
        self.online_net.train()
        return q_vals

    def get_policy_snapshot(self) -> dict:
        """
        Frozen deep copy of online network weights.
        PPR calls this to save this policy into the library
        after training is complete at distance d.
        """
        return copy.deepcopy(self.online_net.state_dict())

    # ═════════════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ═════════════════════════════════════════════════════════════════════════

    def save(self, path: str) -> None:
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "steps_done":  self._steps_done,
        }, path)
        print(f"[DDQN] Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon     = ckpt["epsilon"]
        self._steps_done = ckpt["steps_done"]
        print(f"[DDQN] Loaded ← {path}")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    def summary(self) -> None:
        print(f"\n{'═'*50}")
        print(f"  DDQN Agent")
        print(f"{'═'*50}")
        print(f"  n_actions      : {self.n_actions}")
        print(f"  gamma          : {self.gamma}")
        print(f"  epsilon        : {self.epsilon:.4f}")
        print(f"  epsilon_end    : {self.epsilon_end}")
        print(f"  epsilon_decay  : {self.epsilon_decay:,} steps")
        print(f"  batch_size     : {self.batch_size}")
        print(f"  target_update  : every {self.target_update:,} steps")
        print(f"  steps_done     : {self._steps_done:,}")
        print(f"{'─'*50}")
        print(f"  buffer         : {len(self.buffer):,} / {self.buffer.capacity:,}")
        print(f"  buffer ready   : {self.buffer.is_ready_for(self.batch_size)}")
        if self.losses:
            print(f"  last loss      : {self.losses[-1]:.6f}")
            print(f"  mean Q         : {self.q_means[-1]:.4f}")
        print(f"{'─'*50}")
        total = sum(p.numel() for p in self.online_net.parameters())
        print(f"  network params : {total:,}")
        print(f"  device         : {self.device}")
        print(f"{'═'*50}\n")