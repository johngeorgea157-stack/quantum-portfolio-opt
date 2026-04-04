"""
Tests for QAOA circuit construction.
Validates circuit structure, parameter count, and output format.
NOTE: Hardware runs (run_hardware.py) are excluded from CI.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qaoa.qaoa_circuit import build_qaoa_circuit, get_parameter_count


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_Q():
    """Tiny 3-asset Q matrix for fast circuit tests."""
    return np.array([
        [-1.0,  0.5,  0.3],
        [ 0.5, -0.8,  0.2],
        [ 0.3,  0.2, -1.2],
    ])


# ── Circuit Structure Tests ──────────────────────────────────────────────────

def test_circuit_qubit_count(simple_Q):
    """Circuit must have exactly n qubits (one per asset)."""
    n = simple_Q.shape[0]
    circuit = build_qaoa_circuit(simple_Q, p=1)
    assert circuit.num_qubits == n, f"Expected {n} qubits, got {circuit.num_qubits}"


def test_circuit_parameter_count_p1(simple_Q):
    """For p=1 QAOA: expect 2 parameters (1 gamma + 1 beta)."""
    circuit = build_qaoa_circuit(simple_Q, p=1)
    assert circuit.num_parameters == 2, (
        f"p=1 should have 2 parameters, got {circuit.num_parameters}"
    )


def test_circuit_parameter_count_p2(simple_Q):
    """For p=2 QAOA: expect 4 parameters (2 gamma + 2 beta)."""
    circuit = build_qaoa_circuit(simple_Q, p=2)
    assert circuit.num_parameters == 4, (
        f"p=2 should have 4 parameters, got {circuit.num_parameters}"
    )


def test_circuit_parameter_count_general(simple_Q):
    """Parameter count must always equal 2*p."""
    for p in [1, 2, 3]:
        circuit = build_qaoa_circuit(simple_Q, p=p)
        expected = 2 * p
        assert circuit.num_parameters == expected, (
            f"p={p}: expected {expected} params, got {circuit.num_parameters}"
        )


def test_circuit_depth_increases_with_p(simple_Q):
    """Deeper p should produce a deeper circuit."""
    c1 = build_qaoa_circuit(simple_Q, p=1)
    c2 = build_qaoa_circuit(simple_Q, p=2)
    assert c2.depth() > c1.depth(), "p=2 circuit should be deeper than p=1"


def test_circuit_has_measurements(simple_Q):
    """Circuit must include measurement operations."""
    circuit = build_qaoa_circuit(simple_Q, p=1)
    ops = [inst.operation.name for inst in circuit.data]
    assert "measure" in ops, "Circuit must include measurement gates"


def test_circuit_starts_with_hadamard(simple_Q):
    """First layer must be Hadamard gates (uniform superposition)."""
    circuit = build_qaoa_circuit(simple_Q, p=1)
    first_gates = [inst.operation.name for inst in circuit.data[:simple_Q.shape[0]]]
    assert all(g == "h" for g in first_gates), (
        "Circuit must start with Hadamard gates on all qubits"
    )


def test_get_parameter_count(simple_Q):
    """Utility function must return 2*p."""
    for p in [1, 2, 3, 4]:
        assert get_parameter_count(p) == 2 * p
