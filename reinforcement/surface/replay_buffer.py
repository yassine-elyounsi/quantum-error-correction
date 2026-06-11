"""
Replay Buffer for DDQN
=======================

Stores transitions (state, action, reward, next_state, done)
and serves random mini-batches for training.

Key design decisions
--------------------
- Circular buffer  : fixed capacity, oldest transition overwritten when full
- NumPy storage    : states pre-allocated as a single contiguous array
                     → much faster than a list of tensors
- Lazy conversion  : numpy → torch only happens at sample time
                     → no GPU memory wasted on stored transitions
- Float32 states   : matches the environment observation dtype exactly

Memory calculation for capacity=50,000, d=3:
  states      : 50000 × (6,7,7) × 4 bytes =  58.8 MB
  next_states : 50000 × (6,7,7) × 4 bytes =  58.8 MB
  actions     : 50000 × 4 bytes            =   0.2 MB
  rewards     : 50000 × 4 bytes            =   0.2 MB
  dones       : 50000 × 4 bytes            =   0.2 MB
  ─────────────────────────────────────────────────────
  Total                                    ~ 118 MB
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Circular replay buffer with pre-allocated NumPy arrays.

    Parameters
    ----------
    capacity    : int    maximum number of transitions to store
    state_shape : tuple  shape of a single observation  e.g. (6, 7, 7)
    device      : str    torch device for sampled tensors ('cpu' or 'cuda')
    """

    def __init__(
        self,
        capacity:    int,
        state_shape: tuple,
        device:      str = "cpu",
    ):
        self.capacity    = capacity
        self.state_shape = state_shape
        self.device      = torch.device(device)

        # ── Pre-allocate storage ──────────────────────────────────────────────
        self._states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions     = np.zeros((capacity,),              dtype=np.int64)
        self._rewards     = np.zeros((capacity,),              dtype=np.float32)
        self._dones       = np.zeros((capacity,),              dtype=np.float32)

        # ── Circular buffer pointers ──────────────────────────────────────────
        self._ptr  = 0      # next write position
        self._size = 0      # current number of stored transitions

    # ── Core API ──────────────────────────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Store one transition.

        Parameters
        ----------
        state      : np.ndarray  shape state_shape   e.g. (6, 7, 7)
        action     : int         action index        e.g. 0..18
        reward     : float       shaped reward
        next_state : np.ndarray  shape state_shape
        done       : bool        True if episode ended
        """
        self._states     [self._ptr] = state
        self._next_states[self._ptr] = next_state
        self._actions    [self._ptr] = action
        self._rewards    [self._ptr] = reward
        self._dones      [self._ptr] = float(done)

        # Advance pointer — wrap around when full
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor,   # states
        torch.Tensor,   # actions
        torch.Tensor,   # rewards
        torch.Tensor,   # next_states
        torch.Tensor,   # dones
    ]:
        """
        Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int   number of transitions to sample

        Returns
        -------
        states      : torch.FloatTensor   (batch, *state_shape)
        actions     : torch.LongTensor    (batch,)
        rewards     : torch.FloatTensor   (batch,)
        next_states : torch.FloatTensor   (batch, *state_shape)
        dones       : torch.FloatTensor   (batch,)   1.0=done  0.0=not done
        """
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions, "
            f"cannot sample {batch_size}"
        )

        # Random indices without replacement
        idx = np.random.choice(self._size, size=batch_size, replace=False)

        states      = torch.FloatTensor(self._states     [idx]).to(self.device)
        actions     = torch.LongTensor (self._actions    [idx]).to(self.device)
        rewards     = torch.FloatTensor(self._rewards    [idx]).to(self.device)
        next_states = torch.FloatTensor(self._next_states[idx]).to(self.device)
        dones       = torch.FloatTensor(self._dones      [idx]).to(self.device)

        return states, actions, rewards, next_states, dones

    # ── Properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self, batch_size: int = 64) -> bool:
        """True if buffer has enough transitions to start training."""
        return self._size >= batch_size

    def is_ready_for(self, batch_size: int) -> bool:
        """True if buffer has at least batch_size transitions."""
        return self._size >= batch_size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def fill_ratio(self) -> float:
        """How full the buffer is, from 0.0 to 1.0."""
        return self._size / self.capacity

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        state_bytes = 2 * np.prod(self.state_shape) * self.capacity * 4
        total_mb    = (state_bytes + self.capacity * 3 * 4) / 1024 ** 2

        print(f"\n{'═'*45}")
        print(f"  Replay Buffer")
        print(f"{'═'*45}")
        print(f"  capacity    : {self.capacity:,}")
        print(f"  state_shape : {self.state_shape}")
        print(f"  device      : {self.device}")
        print(f"  memory      : ~{total_mb:.1f} MB")
        print(f"{'─'*45}")
        print(f"  stored      : {self._size:,} / {self.capacity:,}")
        print(f"  fill ratio  : {self.fill_ratio:.1%}")
        print(f"  ptr         : {self._ptr}")
        print(f"{'═'*45}\n")