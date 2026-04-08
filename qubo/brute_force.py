"""
brute_force.py  —  Day 4: Solve QUBO by checking EVERY possible portfolio
===========================================================================

WHY BRUTE FORCE?
----------------
With n stocks and binary choices (buy / skip), there are 2^n possible portfolios.
For small n (≤ 20), we can just CHECK ALL OF THEM and keep the best.

This gives us the GROUND TRUTH — the mathematically guaranteed optimal answer.
We will later compare QAOA against this to see how good the quantum solution is.

For n = 5 stocks  →  2^5 = 32 portfolios   (trivial)
For n = 8 stocks  →  2^8 = 256 portfolios  (still trivial for a computer)
For n = 30 stocks →  2^30 ≈ 1 billion      (brute force too slow → need QAOA or similar)

That's exactly why quantum computing matters for larger problems!
"""

import numpy as np
from itertools import product


def brute_force_solve(Q, n):
    """
    Find the QUBO minimum by exhaustively checking all 2^n binary vectors.

    Steps:
        1. Generate every binary string of length n  (e.g. 000, 001, 010, 011, ...)
        2. For each, compute  f(x) = x^T Q x
        3. Return the x that gives the smallest f(x)

    Parameters
    ----------
    Q : ndarray (n, n) — the QUBO Q matrix
    n : int            — number of assets (= number of bits)

    Returns
    -------
    best_x   : ndarray (n,) — the optimal binary vector
    best_obj : float        — the minimum objective value achieved
    """
    best_x   = None
    best_obj = float("inf")   # start with +infinity so any real value is better

    # product([0,1], repeat=n) generates all 2^n binary combinations
    for bits in product([0, 1], repeat=n):
        x   = np.array(bits, dtype=int)
        obj = float(x @ Q @ x)      # f(x) = x^T Q x

        if obj < best_obj:
            best_obj = obj
            best_x   = x.copy()

    return best_x, best_obj


def enumerate_all_solutions(Q, n, top_k=5):
    """
    Return the top_k solutions sorted by objective value (best first).
    Useful for understanding the QUBO landscape — not just the single winner.

    Parameters
    ----------
    Q     : ndarray (n, n)
    n     : int
    top_k : int — how many solutions to return

    Returns
    -------
    list of (bitstring, objective_value) tuples, sorted ascending by objective
    """
    results = []

    for bits in product([0, 1], repeat=n):
        x   = np.array(bits, dtype=int)
        obj = float(x @ Q @ x)
        results.append((x.copy(), obj))

    # Sort by objective value, smallest first
    results.sort(key=lambda pair: pair[1])

    return results[:top_k]
