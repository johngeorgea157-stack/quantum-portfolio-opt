"""
Core QAOA Circuit definitions.
This acts as the single source of truth for the quantum circuit construction.
"""
from qiskit import QuantumCircuit

def create_qaoa_circuit():
    """
    Creates a simple initial test circuit in equal superposition.
    Later, this will be expanded into the full QAOA algorithm.
    """
    # Create a Quantum Circuit acting on 1 qubit and 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Apply the Hadamard (H) gate to put the qubit into equal superposition
    qc.h(0)

    # Measure the qubit and store the result in classical bit 0
    qc.measure(0, 0)

    return qc

if __name__ == "__main__":
    qc = create_qaoa_circuit()
    print("Circuit drawn successfully. You can preview it using qc.draw()")
