"""
Dueling CNN — Multi-Discrete variant for realistic d=5 surface code
=====================================================================
Same convolutional backbone as the single-action DuelingCNN, but the output
is restructured for a MultiDiscrete([4]*n_qubits) action space.

Architecture
------------
    Input: (B, 3k, 11, 11)  = (B, 15, 11, 11)   for d=5, k=5
       │
    Conv1 (15 → 32, 3x3, pad 1) + BN + ReLU
    Conv2 (32 → 64, 3x3, pad 1) + BN + ReLU
    Conv3 (64 → 64, 3x3, pad 1) + BN + ReLU
       │
    Flatten → 64 * 11 * 11 = 7744
       │
    Shared FC: 7744 → 512 → 256   (ReLU)       ← single feature trunk
       │
       ├── Value head:     256 → 1                       V(s)
       │
       └── Advantage head: 256 → n_qubits * 4            A(s, ·)
                           reshaped to (B, n_qubits, 4)

Dueling combination, applied PER QUBIT HEAD independently:
    Q[:, i, :] = V(s) + A_i(s, ·) - mean_a A_i(s, a)
    output shape: (B, n_qubits, 4)

Total Q-values per state: n_qubits * 4 = 100 for d=5.

Action selection (greedy):
    action[i] = argmax_a Q[:, i, a]      for each qubit i
    → joint action vector of shape (n_qubits,)
"""

import torch
import torch.nn as nn


class DuelingCNNMultiDiscrete(nn.Module):
    """
    Dueling CNN with one independent advantage head per qubit.

    Parameters
    ----------
    in_channels : int   number of input channels (= 3k = 15 for d=5, k=5)
    grid_size   : int   spatial size (= 2d+1 = 11 for d=5)
    n_qubits    : int   number of data qubits (= d² = 25 for d=5)
    n_per_qubit : int   actions per qubit (= 4: Identity, X, Z, Y)
    """

    def __init__(
        self,
        in_channels: int = 15,
        grid_size:   int = 11,
        n_qubits:    int = 25,
        n_per_qubit: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size   = grid_size
        self.n_qubits    = n_qubits
        self.n_per_qubit = n_per_qubit

        # ── Convolutional backbone (same as single-action version) ────────────
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        flat = 64 * grid_size * grid_size       # 64 * 11 * 11 = 7744

        # ── Shared feature trunk ──────────────────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # ── Value head: single scalar V(s) ────────────────────────────────────
        self.value_head = nn.Linear(256, 1)

        # ── Advantage head: n_qubits * n_per_qubit, reshaped to (n_qubits, 4) ──
        self.advantage_head = nn.Linear(256, n_qubits * n_per_qubit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, grid, grid)

        Returns
        -------
        q : (B, n_qubits, n_per_qubit)   Q-values per qubit per action
        """
        B = x.size(0)

        features = self.conv(x).flatten(1)          # (B, 7744)
        h        = self.shared_fc(features)         # (B, 256)

        value     = self.value_head(h)              # (B, 1)
        advantage = self.advantage_head(h)          # (B, n_qubits * 4)
        advantage = advantage.view(B, self.n_qubits, self.n_per_qubit)  # (B, 25, 4)

        # Dueling combination, per qubit head:
        #   Q_i = V + A_i - mean_a(A_i)
        adv_mean = advantage.mean(dim=2, keepdim=True)          # (B, 25, 1)
        q = value.unsqueeze(1) + advantage - adv_mean           # (B, 25, 4)

        return q

    # ──────────────────────────────────────────────────────────────────────────
    #  CONVENIENCE
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Greedy joint action: independent argmax per qubit head.

        Parameters
        ----------
        x : (B, in_channels, grid, grid)

        Returns
        -------
        action : (B, n_qubits)  int64, each entry in {0,1,2,3}
        """
        q = self.forward(x)                  # (B, 25, 4)
        return q.argmax(dim=2)               # (B, 25)