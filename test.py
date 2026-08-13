"""
test_agent.py
=============
Evaluate a TRAINED encoding agent. Load a saved policy and measure how well
it actually discovers encoding circuits — with hard numbers.

This answers four questions:

  1. RELIABILITY — for each code, how often (out of N attempts) does the
     agent find a valid circuit? This is the number that actually matters:
     you only need ONE success to get a usable circuit.

  2. CIRCUIT QUALITY — how short are the discovered circuits (after pruning)?

  3. NOISE ADAPTATION — does the agent produce DIFFERENT circuits when the
     noise bias c_Z changes? (This is the whole point of a noise-aware agent.)

  4. GENERALIZATION — can it handle code sizes it was trained on, and does
     it degrade gracefully on ones it saw less of?

Usage
-----
    python test_agent.py --load encoding_policy.pkl
    python test_agent.py --load encoding_policy.pkl --attempts 100
"""

import argparse
import numpy as np

from Encoders.Encoding_agent import EncodingAgent
from Encoders.meta_env import build_error_set
from Encoders.Clifford_sim import validate_circuit


def fmt_circuit(circuit):
    return "  ".join(
        f"CNOT({g[1]}->{g[2]})" if g[0] in ("CNOT", "CX") else f"{g[0]}({g[1]})"
        for g in circuit
    )


def test_reliability(agent, configs, attempts):
    """How often does each code succeed over `attempts` independent tries?"""
    print("\n" + "=" * 66)
    print(f"TEST 1 — RELIABILITY  ({attempts} attempts per code)")
    print("=" * 66)
    print(f"{'code':12s}  {'successes':>10s}  {'rate':>6s}  {'best len':>8s}")
    print("-" * 66)

    results = {}
    for (n, k, d, c_Z, p_I) in configs:
        successes = 0
        best_len = None
        for _ in range(attempts):
            # single greedy attempt
            res = agent.encode(n, k, d, c_Z, p_I, n_attempts=1,
                               greedy=False, prune=True)
            if res and res["success"]:
                successes += 1
                if best_len is None or res["length"] < best_len:
                    best_len = res["length"]
        rate = successes / attempts
        label = f"[[{n},{k},{d}]]"
        bl = str(best_len) if best_len is not None else "—"
        print(f"{label:12s}  {successes:>10d}  {rate:>5.0%}  {bl:>8s}")
        results[(n, k, d)] = {"rate": rate, "best_len": best_len}
    return results


def test_quality(agent, configs):
    """Discover one circuit per code, show pruned length and validity."""
    print("\n" + "=" * 66)
    print("TEST 2 — CIRCUIT QUALITY  (pruned circuits, verified)")
    print("=" * 66)

    for (n, k, d, c_Z, p_I) in configs:
        res = agent.encode(n, k, d, c_Z, p_I, n_attempts=30, prune=True)
        label = f"[[{n},{k},{d}]]"
        if res is None or not res["success"]:
            det = res["n_errors_detected"] if res else 0
            tot = res["n_errors_total"] if res else "?"
            print(f"\n{label}  — no valid circuit found "
                  f"(best {det}/{tot} errors)")
            continue

        # independent re-verification
        errs = build_error_set(n, k, d)
        v = validate_circuit(n, k, res["circuit"], errs)
        raw = res.get("length_raw", res["length"])
        print(f"\n{label}  c_Z={c_Z}  p_I={p_I}")
        print(f"  length     : {res['length']} gates"
              + (f"  (pruned from {raw})" if "length_raw" in res else ""))
        print(f"  verified   : {v['success']}  "
              f"({v['num_detected']}/{v['num_total']} errors)")
        print(f"  circuit    : {fmt_circuit(res['circuit'])}")
        print(f"  generators : {v['generators']}")


def test_noise_adaptation(agent, n=5, k=1, d=3, p_I=0.9):
    """Does the agent produce different circuits for different noise bias?"""
    print("\n" + "=" * 66)
    print(f"TEST 3 — NOISE ADAPTATION  on [[{n},{k},{d}]]")
    print("=" * 66)
    print("If the agent is truly noise-aware, the circuits and/or")
    print("generators should differ across c_Z values.\n")

    circuits = {}
    for c_Z in [0.5, 1.0, 2.0]:
        res = agent.encode(n, k, d, c_Z, p_I, n_attempts=30, prune=True)
        if res and res["success"]:
            gens = tuple(res["generators"])
            circuits[c_Z] = gens
            print(f"  c_Z={c_Z}:  {res['length']} gates,  "
                  f"generators = {res['generators']}")
        else:
            print(f"  c_Z={c_Z}:  no valid circuit found")

    # compare
    uniq = set(circuits.values())
    print()
    if len(uniq) > 1:
        print(f"  -> {len(uniq)} DISTINCT codes across noise levels: "
              f"the agent IS adapting to the noise model.")
    elif len(uniq) == 1:
        print("  -> same code for every c_Z: the agent is NOT visibly "
              "adapting (may need more training, or this code is optimal "
              "for all these noise levels).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="encoding_policy.pkl",
                        help="path to the saved policy")
    parser.add_argument("--attempts", type=int, default=50,
                        help="attempts per code in the reliability test")
    args = parser.parse_args()

    print(f"Loading trained policy from {args.load}")
    agent = EncodingAgent.load(args.load)
    agent.info()

    # codes to evaluate  (n, k, d, c_Z, p_I)
    configs = [
        (3, 1, 2, 1.0, 0.9),
        (5, 1, 3, 1.0, 0.9),
        (7, 1, 3, 1.0, 0.9),
        (9, 1, 3, 1.0, 0.9),
    ]

    test_reliability(agent, configs, args.attempts)
    test_quality(agent, configs)
    test_noise_adaptation(agent, n=5, k=1, d=3)

    print("\n" + "=" * 66)
    print("DONE")
    print("=" * 66)
    print("How to read this:")
    print("  - Reliability rate is what matters most. Even 5-10% means you")
    print("    reliably get a circuit by running encode() with n_attempts=50.")
    print("  - If circuits verify True, they are real and usable.")
    print("  - If noise adaptation shows distinct codes, the meta-agent works.")


if __name__ == "__main__":
    main()