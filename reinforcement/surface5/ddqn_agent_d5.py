"""
DDQN Agent for Surface Code Decoding — Distance 5 variant
===========================================================
Identical to the d=3 DDQNAgent, with ONE addition:

  Differential learning rate for transfer learning.

When warm-starting d=5 from d=3 conv weights, Conv1 contains:
  - channels 0:6   copied from d=3   (preserve this knowledge → slow lr)
  - channels 6:10  zero-padded       (learn from scratch     → slow lr)

We give Conv1 a SEPARATE (slower) learning rate than the rest of the
network so the transferred d=3 knowledge is preserved while the FC layers
adapt quickly to the d=5 scale.

  lr_conv1 → Conv1.weight only   (slow, e.g. 1e-5)
  lr       → everything else     (normal, e.g. 3e-5)

If lr_conv1 is None, behaves exactly like the standard single-lr agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Optional

from reinforcement.surface.Dueling_cnn   import DuelingCNN
from reinforcement.surface.replay_buffer import ReplayBuffer


class DDQNAgentD5:
    """
    Double DQN Agent (Dueling CNN) with optional differential learning rate.

    Extra parameter vs the d=3 agent
    --------------------------------
    lr_conv1 : float or None
        If provided, Conv1.weight uses this (slower) learning rate while
        all other parameters use `lr`. Used for transfer-learning stability.
        If None, all parameters use `lr` (standard behaviour).
    grad_clip : float
        Max gradient norm for clipping (default 1.0 — tighter than d=3's
        original 10.0, needed for d=5 stability).
    """

    def __init__(
        self,
        in_channels:     int,
        grid_size:       int,
        n_actions:       int,
        lr:              float = 3e-5,
        lr_conv1:        Optional[float] = None,
        grad_clip:       float = 1.0,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   int   = 100_000,
        batch_size:      int   = 64,
        target_update:   int   = 1_000,
        buffer_capacity: int   = 100_000,
        device:          str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.grad_clip     = grad_clip
        self.device        = torch.device(device)

        # ── Exploration schedule ──────────────────────────────────────────────
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._steps_done   = 0

        # ── Two networks ──────────────────────────────────────────────────────
        self.online_net = DuelingCNN(in_channels, grid_size, n_actions).to(self.device)
        self.target_net = DuelingCNN(in_channels, grid_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ── Optimiser — single or differential ────────────────────────────────
        self.lr        = lr
        self.lr_conv1  = lr_conv1

        if lr_conv1 is not None:
            # Two parameter groups:
            #   Group 1 → Conv1.weight only   (slow lr)
            #   Group 2 → everything else      (normal lr)
            conv1_weight = self.online_net.conv[0].weight
            conv1_id     = id(conv1_weight)
            rest_params  = [p for p in self.online_net.parameters()
                            if id(p) != conv1_id]

            self.optimizer = optim.Adam([
                {"params": [conv1_weight], "lr": lr_conv1},
                {"params": rest_params,    "lr": lr},
            ])
            print(f"[DDQN-D5] Differential lr: Conv1={lr_conv1:.1e}  rest={lr:.1e}")
        else:
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
            print(f"[DDQN-D5] Single lr: {lr:.1e}")

        self.loss_fn = nn.SmoothL1Loss()   # Huber

        # ── Replay buffer ─────────────────────────────────────────────────────
        state_shape = (in_channels, grid_size, grid_size)
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity, state_shape=state_shape, device=device,
        )

        # ── Stats ─────────────────────────────────────────────────────────────
        self.losses       = []
        self.q_means      = []
        self._train_steps = 0

    # ═════════════════════════════════════════════════════════════════════════
    # ACTION SELECTION
    # ═════════════════════════════════════════════════════════════════════════

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        self._steps_done += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (self._steps_done / self.epsilon_decay) * (1.0 - self.epsilon_end)
        )

        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t)
        self.online_net.train()
        return int(q_vals.argmax(dim=1).item())

    # ═════════════════════════════════════════════════════════════════════════
    # STORE TRANSITION
    # ═════════════════════════════════════════════════════════════════════════

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ═════════════════════════════════════════════════════════════════════════
    # TRAINING STEP
    # ═════════════════════════════════════════════════════════════════════════

    def train_step(self) -> Optional[float]:
        if not self.buffer.is_ready_for(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q-values
        q_all     = self.online_net(states)
        q_current = q_all.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # DDQN target
        with torch.no_grad():
            q_next_online = self.online_net(next_states)
            best_actions  = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.target_net(next_states)
            q_next_best   = q_next_target.gather(dim=1, index=best_actions).squeeze(1)
            td_target     = rewards + self.gamma * q_next_best * (1.0 - dones)

        # Loss
        loss = self.loss_fn(q_current, td_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        # Sync target net
        self._train_steps += 1
        if self._train_steps % self.target_update == 0:
            self._sync_target()

        # Stats
        loss_val = loss.item()
        self.losses.append(loss_val)
        self.q_means.append(q_current.mean().item())
        return loss_val

    # ═════════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═════════════════════════════════════════════════════════════════════════

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ═════════════════════════════════════════════════════════════════════════
    # PPR / QUERY INTERFACE
    # ═════════════════════════════════════════════════════════════════════════

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t).squeeze(0).cpu().numpy()
        self.online_net.train()
        return q_vals

    def get_policy_snapshot(self) -> dict:
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
        print(f"[DDQN-D5] Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        # Optimizer state may not match if structure changed — load safely
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"[DDQN-D5] Optimizer state not loaded ({e}); using fresh optimizer")
        self.epsilon     = ckpt["epsilon"]
        self._steps_done = ckpt["steps_done"]
        print(f"[DDQN-D5] Loaded ← {path}")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    def summary(self) -> None:
        print(f"\n{'═'*50}")
        print(f"  DDQN Agent (D5 — differential lr)")
        print(f"{'═'*50}")
        print(f"  n_actions      : {self.n_actions}")
        print(f"  lr (rest)      : {self.lr:.1e}")
        print(f"  lr_conv1       : {self.lr_conv1 if self.lr_conv1 else 'same as rest'}")
        print(f"  grad_clip      : {self.grad_clip}")
        print(f"  gamma          : {self.gamma}")
        print(f"  epsilon        : {self.epsilon:.4f}")
        print(f"  epsilon_decay  : {self.epsilon_decay:,} steps")
        print(f"  batch_size     : {self.batch_size}")
        print(f"  target_update  : every {self.target_update:,} steps")
        print(f"  steps_done     : {self._steps_done:,}")
        print(f"{'─'*50}")
        print(f"  buffer         : {len(self.buffer):,} / {self.buffer.capacity:,}")
        total = sum(p.numel() for p in self.online_net.parameters())
        print(f"  network params : {total:,}")
        print(f"  device         : {self.device}")
        print(f"{'═'*50}\n")