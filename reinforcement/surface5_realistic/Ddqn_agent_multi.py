"""
Multi-Discrete DDQN Agent — realistic d=5 surface code
========================================================
Double DQN with a Dueling CNN that has 25 independent qubit heads.

Differences from the single-action DDQN agent
----------------------------------------------
  • Action is a VECTOR of shape (n_qubits,), each entry in {0,1,2,3}.
  • Reward is a VECTOR of shape (n_qubits,) — per-qubit credit assignment
    from the environment (sum over qubits = global Δweight + terminal).
  • The network outputs (B, n_qubits, 4) Q-values.
  • TD loss is computed per head and averaged over all heads.

Replay buffer stores vector actions and vector rewards.

Update (Double DQN, per head)
-----------------------------
  q_current[b,i]  = Q_online(s)[b, i, a[b,i]]
  best_a[b,i]     = argmax_a Q_online(s')[b, i, a]
  q_next[b,i]     = Q_target(s')[b, i, best_a[b,i]]
  td_target[b,i]  = r[b,i] + γ · q_next[b,i] · (1 - done[b])
  loss            = Huber(q_current, td_target)   averaged over (B, n_qubits)

Exploration
-----------
  ε-greedy applied INDEPENDENTLY per qubit head:
    with prob ε  → that qubit picks a random action in {0,1,2,3}
    else         → that qubit picks argmax of its head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Optional

from reinforcement.surface5_realistic.Dueling_cnn_multi import DuelingCNNMultiDiscrete


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER  (vector actions + vector rewards)
# ══════════════════════════════════════════════════════════════════════════════

class MultiDiscreteReplayBuffer:
    """
    Stores transitions where action and reward are per-qubit vectors.

    state       (in_channels, grid, grid)  float32
    action      (n_qubits,)                int64    in {0,1,2,3}
    reward      (n_qubits,)                float32  per-qubit reward
    next_state  (in_channels, grid, grid)  float32
    done        scalar                     float32
    """

    def __init__(self, capacity, state_shape, n_qubits, device="cpu"):
        self.capacity = capacity
        self.n_qubits = n_qubits
        self.device   = torch.device(device)
        self.ptr      = 0
        self.size     = 0

        self.states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions     = np.zeros((capacity, n_qubits),     dtype=np.int64)
        self.rewards     = np.zeros((capacity, n_qubits),     dtype=np.float32)
        self.dones       = np.zeros(capacity,                 dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.ptr
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.LongTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def is_ready_for(self, batch_size):
        return self.size >= batch_size

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-DISCRETE DDQN AGENT
# ══════════════════════════════════════════════════════════════════════════════

class MultiDiscreteDDQNAgent:
    """
    Double DQN with Dueling CNN multi-discrete heads.

    Parameters
    ----------
    in_channels, grid_size, n_qubits, n_per_qubit : network dims
    lr            : learning rate (all layers except Conv1)
    lr_conv1      : separate (slow) lr for Conv1.weight (None → use lr)
    grad_clip     : max gradient norm
    gamma, epsilon_start/end/decay, batch_size, target_update, buffer_capacity
    device        : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        in_channels:     int   = 15,
        grid_size:       int   = 11,
        n_qubits:        int   = 25,
        n_per_qubit:     int   = 4,
        lr:              float = 3e-5,
        lr_conv1:        Optional[float] = None,
        grad_clip:       float = 1.0,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   int   = 200_000,
        batch_size:      int   = 64,
        target_update:   int   = 1_000,
        buffer_capacity: int   = 100_000,
        identity_bias:   float = 0.9,
        device:          str   = "cpu",
    ):
        self.n_qubits     = n_qubits
        self.n_per_qubit  = n_per_qubit
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update= target_update
        self.grad_clip    = grad_clip
        self.device       = torch.device(device)

        # Exploration
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._steps_done   = 0

        # Identity-biased exploration (critical for multi-discrete action spaces).
        # When a head explores, it picks Identity with prob `identity_bias`,
        # else a uniform non-Identity Pauli. Without this, a random policy flips
        # ~half of all 25 qubits every step and destroys the code instantly.
        self.identity_bias = identity_bias
        self._explore_probs = np.array(
            [identity_bias] + [(1.0 - identity_bias) / (n_per_qubit - 1)]
            * (n_per_qubit - 1)
        )

        # Networks
        self.online_net = DuelingCNNMultiDiscrete(
            in_channels, grid_size, n_qubits, n_per_qubit).to(self.device)
        self.target_net = DuelingCNNMultiDiscrete(
            in_channels, grid_size, n_qubits, n_per_qubit).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer — single or differential lr
        self.lr       = lr
        self.lr_conv1 = lr_conv1
        if lr_conv1 is not None:
            conv1_weight = self.online_net.conv[0].weight
            conv1_id     = id(conv1_weight)
            rest = [p for p in self.online_net.parameters() if id(p) != conv1_id]
            self.optimizer = optim.Adam([
                {"params": [conv1_weight], "lr": lr_conv1},
                {"params": rest,           "lr": lr},
            ])
            print(f"[MD-DDQN] Differential lr: Conv1={lr_conv1:.1e}  rest={lr:.1e}")
        else:
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
            print(f"[MD-DDQN] Single lr: {lr:.1e}")

        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        state_shape = (in_channels, grid_size, grid_size)
        self.buffer = MultiDiscreteReplayBuffer(
            buffer_capacity, state_shape, n_qubits, device)

        # Stats
        self.losses       = []
        self.q_means      = []
        self._train_steps = 0

    # ──────────────────────────────────────────────────────────────────────────
    #  ACTION SELECTION  (ε-greedy per head)
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Returns a joint action vector of shape (n_qubits,), each in {0,1,2,3}.

        ε-greedy applied independently per qubit head:
          with prob ε  → that head picks a uniform random action
          else         → that head picks argmax of its Q-values
        """
        self._steps_done += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (self._steps_done / self.epsilon_decay) * (1.0 - self.epsilon_end)
        )

        # Greedy Q-based action
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(state_t).squeeze(0)        # (n_qubits, 4)
            greedy_a = q.argmax(dim=1).cpu().numpy()       # (n_qubits,)
        self.online_net.train()

        if greedy:
            return greedy_a.astype(np.int64)

        # Per-head ε-greedy: replace some heads with random actions
        explore_mask = np.random.random(self.n_qubits) < self.epsilon
        # Identity-biased random actions (not uniform) — see __init__ note
        random_a     = np.random.choice(
            self.n_per_qubit, size=self.n_qubits, p=self._explore_probs)
        action       = np.where(explore_mask, random_a, greedy_a)
        return action.astype(np.int64)

    # ──────────────────────────────────────────────────────────────────────────
    #  STORE
    # ──────────────────────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        """reward is a per-qubit vector (n_qubits,); action is (n_qubits,)."""
        self.buffer.push(state, action, reward, next_state, done)

    # ──────────────────────────────────────────────────────────────────────────
    #  TRAIN STEP  (Double DQN, per head)
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        if not self.buffer.is_ready_for(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        #   states      (B, 15, 11, 11)
        #   actions     (B, 25)
        #   rewards     (B, 25)
        #   next_states (B, 15, 11, 11)
        #   dones       (B,)

        B = states.size(0)

        # Current Q for taken action, per head
        q_all     = self.online_net(states)                       # (B, 25, 4)
        q_current = q_all.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, 25)

        # DDQN target, per head
        with torch.no_grad():
            q_next_online = self.online_net(next_states)          # (B, 25, 4)
            best_a        = q_next_online.argmax(dim=2, keepdim=True)  # (B, 25, 1)

            q_next_target = self.target_net(next_states)          # (B, 25, 4)
            q_next_best   = q_next_target.gather(2, best_a).squeeze(2)  # (B, 25)

            done_b   = dones.unsqueeze(1)                          # (B, 1) → broadcast
            td_target = rewards + self.gamma * q_next_best * (1.0 - done_b)  # (B, 25)

        # Loss over all heads
        loss = self.loss_fn(q_current, td_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        # Sync target
        self._train_steps += 1
        if self._train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.q_means.append(q_current.mean().item())
        return loss_val

    # ──────────────────────────────────────────────────────────────────────────
    #  SAVE / LOAD
    # ──────────────────────────────────────────────────────────────────────────

    # def save(self, path: str) -> None:
    #     torch.save({
    #         "online_net":  self.online_net.state_dict(),
    #         "target_net":  self.target_net.state_dict(),
    #         "optimizer":   self.optimizer.state_dict(),
    #         "epsilon":     self.epsilon,
    #         "steps_done":  self._steps_done,
    #     }, path)
    #     print(f"[MD-DDQN] Saved → {path}")
    def save(self, path: str) -> None:
     import os
     tmp = path + ".tmp"
     torch.save({
        "online_net":  self.online_net.state_dict(),
        "target_net":  self.target_net.state_dict(),
        "optimizer":   self.optimizer.state_dict(),
        "epsilon":     self.epsilon,
        "steps_done":  self._steps_done,
    }, tmp)
     os.replace(tmp, path)   # atomic rename; avoids partial/locked writes    

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"[MD-DDQN] Optimizer state not loaded ({e}); fresh optimizer")
        self.epsilon     = ckpt["epsilon"]
        self._steps_done = ckpt["steps_done"]
        print(f"[MD-DDQN] Loaded ← {path}")

    # ──────────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ──────────────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        total = sum(p.numel() for p in self.online_net.parameters())
        print(f"\n{'═'*52}")
        print(f"  Multi-Discrete DDQN Agent")
        print(f"{'═'*52}")
        print(f"  heads (qubits)   : {self.n_qubits}")
        print(f"  actions per head : {self.n_per_qubit}")
        print(f"  total Q-values   : {self.n_qubits * self.n_per_qubit}")
        print(f"  lr (rest)        : {self.lr:.1e}")
        print(f"  lr_conv1         : {self.lr_conv1 if self.lr_conv1 else 'same as rest'}")
        print(f"  grad_clip        : {self.grad_clip}")
        print(f"  gamma            : {self.gamma}")
        print(f"  epsilon          : {self.epsilon:.4f}")
        print(f"  epsilon_decay    : {self.epsilon_decay:,} steps")
        print(f"  batch_size       : {self.batch_size}")
        print(f"  target_update    : every {self.target_update:,} steps")
        print(f"  buffer           : {len(self.buffer):,} / {self.buffer.capacity:,}")
        print(f"  network params   : {total:,}")
        print(f"  device           : {self.device}")
        print(f"{'═'*52}\n")