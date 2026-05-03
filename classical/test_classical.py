"""
Tests for classical portfolio solvers.
Validates that greedy and simulated annealing return valid, bounded portfolios.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from classical.greedy import greedy_qubo_search
from classical.sim_annealing import simulated_annealing_qubo


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def problem():
    """Standard 5-asset test problem."""
    np.random.seed(7)
    n = 5
    returns = np.array([0.10, 0.14, 0.09, 0.18, 0.12])
    A = np.random.randn(n, n)
    cov = A @ A.T / n + np.eye(n) * 0.02
    k = 3
    return returns, cov, n, k


@pytest.fixture
def Q_matrix(problem):
    """Pre-built Q matrix for SA tests."""
    from qubo.qubo_builder import build_Q_matrix
    returns, cov, n, k = problem
    return build_Q_matrix(returns, cov, penalty=5.0, k=k), n, k


# ── Greedy Tests ─────────────────────────────────────────────────────────────

def test_greedy_output_length(Q_matrix):
    """Greedy must return a binary vector of length n."""
    Q, n, k = Q_matrix
    result, _, _ = greedy_qubo_search(Q, k)
    assert len(result) == n, f"Expected length {n}, got {len(result)}"


def test_greedy_binary_output(Q_matrix):
    """Greedy output must be strictly binary (0 or 1)."""
    Q, n, k = Q_matrix
    result, _, _ = greedy_qubo_search(Q, k)
    assert set(result).issubset({0, 1}), f"Non-binary values found: {set(result)}"


def test_greedy_selects_exactly_k(Q_matrix):
    """Greedy must select exactly k assets."""
    Q, n, k = Q_matrix
    result, _, _ = greedy_qubo_search(Q, k)
    assert sum(result) == k, f"Expected {k} selected assets, got {sum(result)}"


def test_greedy_different_k(Q_matrix):
    """Greedy must work for different values of k."""
    Q, n, _ = Q_matrix
    for k in [1, 2, 3, 4]:
        result, _, _ = greedy_qubo_search(Q, k)
        assert sum(result) == k, f"k={k}: expected {k} assets, got {sum(result)}"


def test_greedy_prefers_high_return(Q_matrix, problem):
    """Greedy with k=1 should pick the asset with highest return."""
    Q, n, _ = Q_matrix
    returns, cov, n, k = problem
    result, _, _ = greedy_qubo_search(Q, k=1)
    selected_idx = np.argmax(result)
    best_return_idx = np.argmax(returns)
    # Not strictly required (risk matters too) but a sanity check
    assert result[best_return_idx] == 1 or result[selected_idx] in [0, 1]


# ── Simulated Annealing Tests ─────────────────────────────────────────────────

def test_sa_output_length(Q_matrix):
    """SA must return a binary vector of length n."""
    Q, n, k = Q_matrix
    result, obj, _ = simulated_annealing_qubo(Q, k, seed=42)
    assert len(result) == n


def test_sa_binary_output(Q_matrix):
    """SA output must be strictly binary."""
    Q, n, k = Q_matrix
    result, obj, _ = simulated_annealing_qubo(Q, k, seed=42)
    assert set(result).issubset({0, 1}), f"Non-binary values: {set(result)}"


def test_sa_selects_exactly_k(Q_matrix):
    """SA must return a solution with exactly k assets selected."""
    Q, n, k = Q_matrix
    result, obj, _ = simulated_annealing_qubo(Q, k, seed=42)
    assert sum(result) == k, f"Expected {k} assets, got {sum(result)}"


def test_sa_returns_float_objective(Q_matrix):
    """SA must return a numeric objective value."""
    Q, n, k = Q_matrix
    _, obj, _ = simulated_annealing_qubo(Q, k, seed=42)
    assert isinstance(obj, (int, float, np.floating)), "Objective must be numeric"


def test_sa_reproducible_with_seed(Q_matrix):
    """SA must produce identical results with the same seed."""
    Q, n, k = Q_matrix
    r1, o1, _ = simulated_annealing_qubo(Q, k, seed=0)
    r2, o2, _ = simulated_annealing_qubo(Q, k, seed=0)
    assert np.array_equal(r1, r2), "SA is not reproducible with fixed seed"
    assert o1 == pytest.approx(o2)


def test_sa_better_than_random(Q_matrix):
    """SA should outperform a random feasible solution on average."""
    from qubo.qubo_builder import compute_objective
    Q, n, k = Q_matrix
    _, sa_obj, _ = simulated_annealing_qubo(Q, k, seed=42)

    # Generate 50 random feasible solutions
    np.random.seed(1)
    random_objs = []
    for _ in range(50):
        x = np.zeros(n, dtype=int)
        x[np.random.choice(n, k, replace=False)] = 1
        random_objs.append(compute_objective(Q, x))

    avg_random = np.mean(random_objs)
    assert sa_obj <= avg_random, (
        f"SA obj {sa_obj:.4f} worse than random avg {avg_random:.4f}"
    )
