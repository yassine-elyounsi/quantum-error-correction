"""
Dueling CNN Q-Network for Surface Code Decoding
=================================================

Input  : state tensor  shape (2k, 2d+1, 2d+1)
                             (6,  7,    7   )   for d=3, k=3

Output : Q-values      shape (n_actions,)
                             (19,)              for d=3

Architecture
------------

(6, 7, 7)
    │
    ▼
┌─────────────────────────────────┐
│  Conv2d(6  → 32, k=3, pad=1)   │  ← detect local syndrome patterns
│  BatchNorm2d(32)                │
│  ReLU                           │
├─────────────────────────────────┤
│  Conv2d(32 → 64, k=3, pad=1)   │  ← combine neighbouring patterns
│  BatchNorm2d(64)                │
│  ReLU                           │
├─────────────────────────────────┤
│  Conv2d(64 → 64, k=3, pad=1)   │  ← deep abstract features
│  BatchNorm2d(64)                │
│  ReLU                           │
└─────────────────────────────────┘
    │
    ▼
Flatten  →  (64 × 7 × 7 = 3136)
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
┌──────────────────────┐              ┌──────────────────────────┐
│   Value  stream      │              │   Advantage  stream      │
│   FC(3136 → 256)     │              │   FC(3136 → 256)         │
│   ReLU               │              │   ReLU                   │
│   FC(256  → 1)       │              │   FC(256  → n_actions)   │
│                      │              │                          │
│   V(s)  scalar       │              │   A(s,a)  vector         │
└──────────────────────┘              └──────────────────────────┘
    │                                              │
    └──────────────────┬───────────────────────────┘
                       ▼
         Q(s,a) = V(s) + A(s,a) - mean_a[ A(s,a) ]

Why dueling?
------------
Many steps the syndrome weight does NOT change regardless of which
correction is applied.  The Value stream learns "how good is this
syndrome state" independently of the action.  The Advantage stream
learns "which action is relatively better".  This separation makes
learning faster and more stable for our environment.

Why BatchNorm?
--------------
Syndrome grids are sparse (mostly zeros, few ±1 values).
BatchNorm prevents the activations from collapsing during early
training when the agent is exploring randomly.
"""

import torch
import torch.nn as nn


class DuelingCNN(nn.Module):
    """
    Dueling CNN Q-Network.

    Parameters
    ----------
    in_channels : int   number of input channels  (= 2k = 6 for d=3)
    grid_size   : int   spatial size of the grid  (= 2d+1 = 7 for d=3)
    n_actions   : int   number of discrete actions (= 2d²+1 = 19 for d=3)
    """

    def __init__(self, in_channels: int, grid_size: int, n_actions: int):
        super().__init__()

        self.in_channels = in_channels
        self.grid_size   = grid_size
        self.n_actions   = n_actions

        # ── Convolutional backbone ────────────────────────────────────────────
        # padding=1 keeps spatial size unchanged (7→7→7)
        # so the flattened size is always 64 × grid_size × grid_size
        self.conv = nn.Sequential(

            # Layer 1 — local pattern detection
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2 — combine neighbouring patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3 — deep abstract features
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Flattened feature size: 64 channels × grid_size × grid_size
        flat_size = 64 * grid_size * grid_size   # = 3136 for d=3

        # ── Value stream  V(s) → scalar ──────────────────────────────────────
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),               # single scalar
        )

        # ── Advantage stream  A(s,a) → vector of size n_actions ──────────────
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),       # one value per action
        )

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, in_channels, grid_size, grid_size)
                                (B,     6,           7,         7        )

        Returns
        -------
        q : torch.Tensor  shape (batch, n_actions)
                                (B,     19         )
        """
        # 1. Conv backbone
        features = self.conv(x)                          # (B, 64, 7, 7)

        # 2. Flatten
        features = features.view(features.size(0), -1)  # (B, 3136)

        # 3. Two streams
        V = self.value_stream(features)                  # (B, 1)
        A = self.advantage_stream(features)              # (B, 19)

        # 4. Dueling combination
        #    Q(s,a) = V(s) + A(s,a) - mean_a[ A(s,a) ]
        #    subtracting the mean makes A identifiable (zero-mean advantage)
        Q = V + A - A.mean(dim=1, keepdim=True)         # (B, 19)

        return Q

    # ── Convenience methods ───────────────────────────────────────────────────

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for a single state (no batch dimension needed).

        Parameters
        ----------
        state : torch.Tensor  shape (in_channels, grid_size, grid_size)

        Returns
        -------
        q : torch.Tensor  shape (n_actions,)
        """
        with torch.no_grad():
            return self.forward(state.unsqueeze(0)).squeeze(0)

    def get_action(self, state: torch.Tensor) -> int:
        """
        Greedy action selection for a single state.

        Parameters
        ----------
        state : torch.Tensor  shape (in_channels, grid_size, grid_size)

        Returns
        -------
        action : int
        """
        q = self.get_q_values(state)
        return int(q.argmax().item())

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        """
        He initialisation for ReLU layers.
        Zero-initialise the final advantage layer bias so Q-values
        start near zero rather than random large values.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print a readable summary of shapes and parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        flat_size = 64 * self.grid_size * self.grid_size

        print(f"\n{'═'*52}")
        print(f"  Dueling CNN Q-Network")
        print(f"{'═'*52}")
        print(f"  Input  : ({self.in_channels}, {self.grid_size}, {self.grid_size})")
        print(f"{'─'*52}")
        print(f"  Conv1  : ({self.in_channels}→32,  k=3, pad=1)  → (32, {self.grid_size}, {self.grid_size})")
        print(f"  Conv2  : (32→64, k=3, pad=1)  → (64, {self.grid_size}, {self.grid_size})")
        print(f"  Conv3  : (64→64, k=3, pad=1)  → (64, {self.grid_size}, {self.grid_size})")
        print(f"  Flatten: {flat_size}")
        print(f"{'─'*52}")
        print(f"  Value     : {flat_size}→256→1")
        print(f"  Advantage : {flat_size}→256→{self.n_actions}")
        print(f"{'─'*52}")
        print(f"  Output : Q(s,a)  shape ({self.n_actions},)")
        print(f"{'─'*52}")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"{'═'*52}\n")