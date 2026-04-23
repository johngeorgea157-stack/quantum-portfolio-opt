"""
Core QAOA Circuit definitions.
This acts as the single source of truth for the quantum circuit construction.
"""
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
import numpy as np

def qubo_to_ising(Q):
    """
    Convert a QUBO Q-matrix into an Ising Hamiltonian (SparsePauliOp).
    Returns the Hamiltonian operator and a constant offset.
    """
    n = len(Q)
    pauli_list = []
    offset = 0.0

    # --- Diagonal terms: Q_ii * x_i ---
    for i in range(n):
        offset += Q[i][i] / 2.0
        pauli_str = ['I'] * n
        pauli_str[n - 1 - i] = 'Z'  # Qiskit orders strings right to left
        pauli_list.append((''.join(pauli_str), -Q[i][i] / 2.0))

    # --- Off-diagonal terms: Q_ij * x_i * x_j (i < j) ---
    for i in range(n):
        for j in range(i + 1, n):
            c = Q[i][j] + Q[j][i]
            offset += c / 4.0

            # Single Z_i
            p_i = ['I'] * n
            p_i[n - 1 - i] = 'Z'
            pauli_list.append((''.join(p_i), -c / 4.0))

            # Single Z_j
            p_j = ['I'] * n
            p_j[n - 1 - j] = 'Z'
            pauli_list.append((''.join(p_j), -c / 4.0))

            # Z_i Z_j interaction
            p_ij = ['I'] * n
            p_ij[n - 1 - i] = 'Z'
            p_ij[n - 1 - j] = 'Z'
            pauli_list.append((''.join(p_ij), c / 4.0))

    # Simplify strings
    hamiltonian = SparsePauliOp.from_list(pauli_list).simplify()

    # Cast coefficients to real since QAOAAnsatz rejects complex numbers
    real_coeffs = hamiltonian.coeffs.real
    hamiltonian = SparsePauliOp(hamiltonian.paulis, coeffs=real_coeffs)

    return hamiltonian, offset

def create_qaoa_circuit(Q, p=1):
    """
    Generates the fully parameterised QAOA circuit based on the QUBO matrix.
    
    Returns:
        circuit: QAOAAnsatz
        hamiltonian: SparsePauliOp
        offset: float
    """
    hamiltonian, offset = qubo_to_ising(Q)
    qaoa_circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p)
    return qaoa_circuit, hamiltonian, offset
