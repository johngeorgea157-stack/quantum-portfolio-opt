"""
qubo_builder.py  —  Day 4: Convert Portfolio Problem → QUBO
============================================================

WHAT IS A QUBO?
---------------
QUBO = Quadratic Unconstrained Binary Optimization.
It is a way to rewrite any optimization problem so that:
  1. Variables are binary  → each x_i ∈ {0, 1}
  2. The problem is "unconstrained" → constraints are ADDED INTO the objective
     as penalty terms (they become extra costs when violated)
  3. The objective is at most quadratic → x_i * x_j terms (no cubes etc.)

STANDARD QUBO FORM:
    Minimize:  f(x) = x^T Q x = Σ_i Σ_j  Q_ij * x_i * x_j

where Q is an n×n matrix (called the Q matrix).

OUR PORTFOLIO PROBLEM:
----------------------
We have n stocks.  x_i = 1 means "buy stock i", x_i = 0 means "skip it".
We want to:
  - MAXIMIZE  expected return  Σ_i μ_i * x_i          ← bigger is better
  - MINIMIZE  risk (variance)  Σ_i Σ_j σ_ij * x_i * x_j  ← smaller is better
  - SATISFY   cardinality constraint: exactly k stocks selected

MATHEMATICAL FORMULATION:
--------------------------
Minimize:
    f(x) = -Σ_i μ_i * x_i                    ← return (negated, so minimising = maximising return)
           + q * Σ_i Σ_j σ_ij * x_i * x_j    ← risk (q = risk aversion weight)
           + λ * (Σ_i x_i  -  k)^2            ← PENALTY: punish if ≠ k stocks chosen

Breaking down the PENALTY term:
    (Σ_i x_i - k)^2 = (Σ_i x_i)^2 - 2k(Σ_i x_i) + k^2

    Expanding (Σ_i x_i)^2:
        = Σ_i x_i^2  +  2 Σ_{i<j} x_i * x_j
        = Σ_i x_i    +  2 Σ_{i<j} x_i * x_j    ← because x_i^2 = x_i for binary

So the penalty adds to Q as:
    Q_ii  +=  λ * (1 - 2k)          ← diagonal terms (from x_i^2 and -2k*x_i)
    Q_ij  +=  λ                     ← off-diagonal terms (from the cross products), i≠j

PUTTING IT ALL TOGETHER — THE Q MATRIX:
    Q_ii  =  -μ_i  +  q * σ_ii  +  λ * (1 - 2k)
    Q_ij  =   q * σ_ij  +  λ             for i ≠ j

That's it!  One matrix Q captures EVERYTHING: return, risk, and the constraint.
"""

import numpy as np


def build_Q_matrix(returns, cov, penalty, k, risk_weight=1.0):
    """
    Build the QUBO Q matrix for the portfolio problem.

    Parameters
    ----------
    returns     : array of shape (n,)   — expected return μ_i for each stock
    cov         : array of shape (n,n)  — covariance matrix σ_ij
    penalty     : float  λ — how hard to enforce "exactly k stocks" constraint
                  (rule of thumb: set λ ≈ 10 × max(|returns|) so the constraint
                   always dominates over the objective when violated)
    k           : int   — how many stocks to select
    risk_weight : float q — how much to penalise risk vs return (default 1.0)

    Returns
    -------
    Q : ndarray of shape (n, n)  — the symmetric QUBO matrix
    """
    n = len(returns)
    Q = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # DIAGONAL: return contribution + risk (self-variance) + penalty diagonal
                Q[i][i] = (
                    -returns[i]  # maximise return  (negated)
                    + risk_weight * cov[i][i]  # risk: own variance
                    + penalty * (1 - 2 * k)  # penalty: from (x_i - k)^2 expansion
                )
            else:
                # OFF-DIAGONAL: risk (covariance) + penalty cross terms
                Q[i][j] = (
                    risk_weight * cov[i][j]  # risk: covariance between assets i and j
                    + penalty  # penalty: from Σ_i Σ_j x_i*x_j cross terms
                )

    # Q must be symmetric for a valid QUBO
    # (it already is by construction since cov is symmetric, but let's be safe)
    Q = (Q + Q.T) / 2

    return Q


def compute_objective(Q, x):
    """
    Compute the QUBO objective: f(x) = x^T Q x = Σ_i Σ_j Q_ij * x_i * x_j

    Parameters
    ----------
    Q : ndarray (n, n) — the Q matrix
    x : array-like (n,) — binary solution vector  (each entry 0 or 1)

    Returns
    -------
    float — objective value (lower is better for minimisation)
    """
    x = np.array(x)
    return float(x @ Q @ x)  # dot product shorthand for x^T Q x


def decode_bitstring(x, tickers):
    """
    Helper: map a binary vector back to stock names.

    Parameters
    ----------
    x       : array-like (n,) — binary solution
    tickers : list of str     — stock names in the same order as the Q matrix

    Returns
    -------
    list of selected stock names
    """
    return [tickers[i] for i in range(len(x)) if x[i] == 1]
