"""
Runs the QAOA circuit on a local CPU simulator.
Lightning fast, perfect for debugging, costs nothing.
"""
from qiskit.primitives import StatevectorSampler as Sampler
from qaoa_circuit import create_qaoa_circuit

def main():
    print("Building quantum circuit...")
    qc = create_qaoa_circuit()
    
    print("\nInitializing local Statevector Simulator...")
    sampler = Sampler()
    
    print("Running simulation (shots=1000)...")
    job = sampler.run([qc], shots=1000)
    result = job.result()
    
    # Extract the counts from the classical register 'c'
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()
    
    print("\n--- SIMULATION RESULTS ---")
    print(counts)
    print("You can plot this using qiskit.visualization.plot_histogram(counts)")

if __name__ == "__main__":
    main()
