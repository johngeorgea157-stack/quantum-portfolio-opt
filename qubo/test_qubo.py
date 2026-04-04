"""
Tests for QUBO construction and brute-force solver.
Validates Q matrix correctness and objective function values.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qubo.qubo_builder import build_Q_matrix, compute_objective
from qubo.brute_force import brute_force_solve


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Small 4-asset problem with known properties."""
    np.random.seed(42)
    n = 4
    returns = np.array([0.12, 0.08, 0.15, 0.10])
    # Positive definite covariance
    A = np.random.randn(n, n)
    cov = A @ A.T / n + np.eye(n) * 0.01
    return returns, cov, n


# ── Q Matrix Tests ───────────────────────────────────────────────────────────

def test_Q_matrix_shape(sample_data):
    """Q matrix must be square with size = number of assets."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    assert Q.shape == (n, n), f"Expected ({n},{n}), got {Q.shape}"


def test_Q_matrix_symmetric(sample_data):
    """Q matrix must be symmetric for valid QUBO."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    assert np.allclose(Q, Q.T, atol=1e-8), "Q matrix is not symmetric"


def test_Q_matrix_no_nan(sample_data):
    """Q matrix must contain no NaN or Inf values."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    assert not np.any(np.isnan(Q)), "Q matrix contains NaN"
    assert not np.any(np.isinf(Q)), "Q matrix contains Inf"


def test_Q_matrix_penalty_effect(sample_data):
    """Higher penalty should increase diagonal dominance."""
    returns, cov, n = sample_data
    Q_low  = build_Q_matrix(returns, cov, penalty=0.1, k=2)
    Q_high = build_Q_matrix(returns, cov, penalty=10.0, k=2)
    diag_low  = np.sum(np.abs(np.diag(Q_low)))
    diag_high = np.sum(np.abs(np.diag(Q_high)))
    assert diag_high > diag_low, "Higher penalty should increase diagonal magnitude"


# ── Objective Function Tests ─────────────────────────────────────────────────

def test_objective_all_zeros(sample_data):
    """All-zero bitstring should yield objective = 0."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    x = np.zeros(n, dtype=int)
    obj = compute_objective(Q, x)
    assert obj == pytest.approx(0.0, abs=1e-8)


def test_objective_known_bitstring(sample_data):
    """Manually verify objective for a known bitstring: x^T Q x."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    x = np.array([1, 0, 1, 0])
    expected = x @ Q @ x
    result = compute_objective(Q, x)
    assert result == pytest.approx(expected, rel=1e-6)


# ── Brute Force Solver Tests ─────────────────────────────────────────────────

def test_brute_force_returns_valid_bitstring(sample_data):
    """Brute force must return a binary vector of correct length."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    best_x, best_obj = brute_force_solve(Q, n)
    assert len(best_x) == n
    assert set(best_x).issubset({0, 1}), "Solution must be binary"


def test_brute_force_respects_k_constraint(sample_data):
    """Optimal bitstring should select exactly k assets."""
    returns, cov, n = sample_data
    k = 2
    Q = build_Q_matrix(returns, cov, penalty=100.0, k=k)
    best_x, _ = brute_force_solve(Q, n)
    assert sum(best_x) == k, f"Expected {k} assets selected, got {sum(best_x)}"


def test_brute_force_is_optimal(sample_data):
    """Brute force objective must be ≤ all other feasible solutions."""
    returns, cov, n = sample_data
    Q = build_Q_matrix(returns, cov, penalty=1.0, k=2)
    best_x, best_obj = brute_force_solve(Q, n)

    # Check against 20 random bitstrings
    np.random.seed(0)
    for _ in range(20):
        x = np.random.randint(0, 2, n)
        obj = compute_objective(Q, x)
        assert best_obj <= obj + 1e-8, "Brute force did not find global optimum"
