"""
circuit_pruner.py
=================
Remove redundant gates from a discovered encoding circuit while keeping it
valid (still detects exactly the same error set under Knill-Laflamme).

The RL agent often pads a working circuit with extra gates that do not change
the final code (e.g. repeated H(0), or gates that cancel out). The paper does
this pruning manually; here we automate it.

Two strategies, applied in order:

  1. Greedy single-gate removal
     Try deleting each gate one at a time. If the circuit still detects all
     target errors without it, keep it deleted. Repeat until a full pass
     removes nothing.

  2. Greedy contiguous-block removal (optional, stronger)
     Try deleting short consecutive runs of gates (length 2, 3) in case a
     pair cancels but neither is removable alone.

Usage
-----
    from circuit_pruner import prune_circuit

    short = prune_circuit(n, k, circuit, error_strings)
    # short is a new gate list, guaranteed to detect the same errors
"""

from Encoders.Clifford_sim import validate_circuit


def _is_valid(n, k, circuit, error_strings):
    """True iff the circuit detects ALL target errors (exact KL)."""
    if len(circuit) == 0:
        return False
    res = validate_circuit(n, k, circuit, error_strings)
    return res["success"]


def prune_circuit(n, k, circuit, error_strings, block_lengths=(1, 2, 3),
                  verbose=False):
    """
    Return a pruned copy of `circuit` that still detects all `error_strings`.

    Parameters
    ----------
    n, k          : code parameters.
    circuit       : list of gate tuples, e.g. [('H',0),('CNOT',0,1),...].
    error_strings : list of Pauli strings the code must detect.
    block_lengths : run lengths to attempt removing. (1,) = single gates only.
                    (1,2,3) also tries removing consecutive pairs/triples.
    verbose       : print what gets removed.

    Returns
    -------
    pruned : a new, shorter (or equal) gate list, still valid.
    """
    # sanity: only prune if the input is actually valid to begin with
    if not _is_valid(n, k, circuit, error_strings):
        if verbose:
            print("  input circuit is not valid; returning unchanged")
        return list(circuit)

    pruned = list(circuit)
    changed = True
    while changed:
        changed = False
        for blen in block_lengths:
            i = 0
            while i + blen <= len(pruned):
                candidate = pruned[:i] + pruned[i + blen:]
                if _is_valid(n, k, candidate, error_strings):
                    if verbose:
                        removed = pruned[i:i + blen]
                        print(f"  removed {removed}  -> {len(candidate)} gates")
                    pruned = candidate
                    changed = True
                    # don't advance i; the next gate shifted into this slot
                else:
                    i += 1
            if changed:
                # restart from single-gate passes after any change
                break
    return pruned


def prune_report(n, k, circuit, error_strings):
    """
    Prune and print a before/after summary. Returns the pruned circuit.
    """
    before = len(circuit)
    pruned = prune_circuit(n, k, circuit, error_strings)
    after = len(pruned)

    def fmt(c):
        return "  ".join(
            f"CNOT({g[1]}->{g[2]})" if g[0] in ("CNOT", "CX")
            else f"{g[0]}({g[1]})"
            for g in c
        )

    res = validate_circuit(n, k, pruned, error_strings)
    print(f"[[{n},{k}]]  {before} gates -> {after} gates "
          f"({before - after} removed, {100*(before-after)/before:.0f}% shorter)")
    print(f"  pruned circuit : {fmt(pruned)}")
    print(f"  still valid    : {res['success']}  "
          f"({res['num_detected']}/{res['num_total']} errors)")
    print(f"  generators     : {res['generators']}")
    return pruned


if __name__ == "__main__":
    # Demo on the circuits discovered during training
    from meta_env import build_error_set

    circuits = {
        (3, 1, 2): [('CNOT', 0, 2)],
        (5, 1, 3): [('CNOT',0,1),('H',1),('H',0),('CNOT',3,4),('H',0),
                    ('CNOT',0,4),('CNOT',0,2),('CNOT',1,3),('H',0),
                    ('CNOT',0,2),('H',0),('CNOT',0,1),('CNOT',2,3),
                    ('CNOT',3,4),('CNOT',2,4)],
        (7, 1, 3): [('CNOT',2,4),('CNOT',0,1),('CNOT',0,2),('H',0),
                    ('CNOT',0,1),('CNOT',0,4),('CNOT',0,2),('CNOT',4,5),
                    ('H',0),('CNOT',3,4),('CNOT',0,6),('CNOT',0,1),
                    ('CNOT',0,2),('H',5),('CNOT',0,3)],
        (9, 1, 3): [('H',0),('CNOT',0,1),('H',8),('CNOT',6,7),('CNOT',0,1),
                    ('CNOT',0,1),('CNOT',0,6),('H',0),('CNOT',0,1),
                    ('CNOT',1,3),('CNOT',0,2),('H',0),('CNOT',0,1),
                    ('CNOT',0,2),('CNOT',4,6),('CNOT',0,6),('H',0),('CNOT',0,2)],
    }

    print("CIRCUIT PRUNING — before vs after")
    print("=" * 64)
    for (n, k, d), circ in circuits.items():
        errs = build_error_set(n, k, d)
        print()
        prune_report(n, k, circ, errs)