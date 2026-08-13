"""
surface_render.py — clean, reference-style rendering for the d=3 dashboard.

draw_lattice(rs)            -> matplotlib Figure (the lattice)
draw_qvalues(q, chosen, n) -> matplotlib Figure (Q-values over 19 actions)

Style: light, airy, rounded-square stabilizers, numbered qubit circles,
errors color-coded, corrected qubits turn green.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

# palette (matches the reference legend) ──────────────────────────────────────
X_BLUE   = "#3b82f6"     # X error  /  X-stab fired
Z_ORANGE = "#f59e0b"     # Z error  /  Z-stab fired
Y_PURPLE = "#8b5cf6"     # Y error
GREEN    = "#22c55e"     # corrected
PINK     = "#ec4899"     # last action
QGREY    = "#d4dae3"     # idle qubit
QEDGE    = "#aeb8c7"
PLQ_IDLE = "#eef1f6"
PLQ_EDGE = "#dde3ec"
NUM_DARK = "#475569"
LATTICE  = "#e2e8f0"


def _xy(pos):
    """(row,col) on the (2d+1) grid -> tidy plot coords (col/2, -row/2)."""
    r, c = pos
    return (c / 2.0, -r / 2.0)


def draw_lattice(rs, figsize=(5.6, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0); ax.set_facecolor("none")

    data_pos = rs["data_pos"]
    dxy = [_xy(p) for p in data_pos]
    nq = rs["n_qubits"]

    # faint lattice lines between orthogonally adjacent data qubits
    for i in range(nq):
        for j in range(i + 1, nq):
            (xi, yi), (xj, yj) = dxy[i], dxy[j]
            if abs(xi - xj) + abs(yi - yj) == 1.0:
                ax.plot([xi, xj], [yi, yj], color=LATTICE, lw=2.0, zorder=0)

    # stabilizer plaquettes — uniform rounded squares centred on the ancilla
    def plaquette(pos, fired, color):
        x, y = _xy(pos)
        s = 0.62
        fc = color if fired else PLQ_IDLE
        ec = color if fired else PLQ_EDGE
        ax.add_patch(FancyBboxPatch(
            (x - s / 2, y - s / 2), s, s,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.4,
            alpha=0.95 if fired else 0.9, zorder=1, mutation_aspect=1))

    for pos, fired in zip(rs["real_z_pos"], rs["z_fired"]):
        plaquette(pos, fired, Z_ORANGE)
    for pos, fired in zip(rs["real_x_pos"], rs["x_fired"]):
        plaquette(pos, fired, X_BLUE)

    # data qubits with error / corrected colouring
    for i, (x, y) in enumerate(dxy):
        ex, ez = int(rs["x_error"][i]), int(rs["z_error"][i])
        cx, cz = int(rs["x_corr"][i]), int(rs["z_corr"][i])
        rx, rz = ex ^ cx, ez ^ cz                      # residual after correction
        if (cx or cz) and rx == 0 and rz == 0:
            fc, txt = GREEN, "white"                   # corrected
        elif rx and rz:
            fc, txt = Y_PURPLE, "white"
        elif rx:
            fc, txt = X_BLUE, "white"
        elif rz:
            fc, txt = Z_ORANGE, "white"
        else:
            fc, txt = QGREY, NUM_DARK
        ax.add_patch(Circle((x, y), 0.205, facecolor=fc, edgecolor=QEDGE,
                            linewidth=1.6, zorder=4))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=txt, zorder=5)

    # last action pulse ring
    a = rs["last_action"]
    if a is not None and a < 2 * nq:
        q = a if a < nq else a - nq
        x, y = dxy[q]
        ax.add_patch(Circle((x, y), 0.31, fill=False, edgecolor=PINK,
                            linewidth=2.4, alpha=0.85, zorder=6))

    xs = [p[0] for p in dxy]; ys = [p[1] for p in dxy]
    ax.set_xlim(min(xs) - 0.85, max(xs) + 0.85)
    ax.set_ylim(min(ys) - 0.85, max(ys) + 0.85)
    ax.set_aspect("equal"); ax.axis("off")

    legend = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor=X_BLUE,   markersize=11, label='X error'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=Z_ORANGE, markersize=11, label='Z error'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=Y_PURPLE, markersize=11, label='Y error'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=GREEN,    markersize=11, label='corrected'),
        Line2D([0],[0], marker='s', color='none', markerfacecolor=X_BLUE,   markersize=11, label='X-stab fired'),
        Line2D([0],[0], marker='s', color='none', markerfacecolor=Z_ORANGE, markersize=11, label='Z-stab fired'),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=3, frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.3)
    fig.tight_layout()
    return fig


def draw_qvalues(q, chosen, n_qubits=9, figsize=(7.4, 2.5)):
    q = np.asarray(q, dtype=float)
    labels = [f"X{i}" for i in range(n_qubits)] + [f"Z{i}" for i in range(n_qubits)] + ["I"]
    colors = [X_BLUE] * n_qubits + [Z_ORANGE] * n_qubits + ["#94a3b8"]
    if chosen is not None:
        colors[chosen] = PINK
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.bar(range(len(q)), q, color=colors, edgecolor="none")
    ax.set_xticks(range(len(q))); ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color="#cbd5e1", lw=0.8)
    ax.set_ylabel("Q-value", fontsize=9, color=NUM_DARK)
    if chosen is not None:
        ax.text(chosen, q[chosen], " \u25c4", color=PINK, fontsize=10,
                va="center", ha="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cbd5e1")
    ax.tick_params(length=0, colors=NUM_DARK)
    fig.tight_layout()
    return fig