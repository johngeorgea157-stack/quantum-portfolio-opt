"""
Runs the QAOA circuit on a local CPU simulator.
Lightning fast, perfect for debugging, costs nothing.
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# Make sure we can import from the 'qubo' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qubo.qubo_builder import build_Q_matrix
from qaoa_circuit import create_qaoa_circuit

def main():
    # --- 1. Data Fetching ---
    tickers = ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS']
    print(f"Fetching live data for {tickers}...")
    close_prices = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, period='1y', auto_adjust=True, progress=False)
        close_prices[t] = df['Close']
    
    close_prices = close_prices.dropna()
    daily_returns = close_prices.pct_change(fill_method=None).dropna()
    mu = daily_returns.mean().values * 252
    sigma = daily_returns.cov().values * 252

    # --- 2. QUBO Formulation ---
    k = 2  # Pick exactly 2 stocks
    penalty = 10.0
    Q = build_Q_matrix(mu, sigma, penalty=penalty, k=k)

    # --- 3. Circuit Creation ---
    p = 4
    print(f"\nBuilding QAOA circuit with depth p={p}...")
    qaoa_circuit, cost_hamiltonian, offset = create_qaoa_circuit(Q, p)

    # --- 4. Parameter Optimization ---
    estimator = StatevectorEstimator()

    def evaluate_expectation(x):
        pub = (qaoa_circuit, cost_hamiltonian, x)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    # Dynamic initial point sizing based on p
    np.random.seed(42)
    initial_point = np.random.rand(qaoa_circuit.num_parameters) * np.pi

    print("\nRunning COBYLA optimizer (Simulator)...")
    res = minimize(evaluate_expectation, initial_point, method='COBYLA')
    optimal_params = res.x
    print(f"Optimal Parameters: {np.round(optimal_params, 4)}")
    print(f"Lowest Energy Found: {res.fun:.4f}")

    # --- 5. Circuit Execution & Sampling ---
    print("\nExecuting final optimized circuit...")
    optimized_circuit = qaoa_circuit.assign_parameters(optimal_params)
    optimized_circuit.measure_all()

    sampler = StatevectorSampler()
    job = sampler.run([optimized_circuit])
    result = job.result()[0]

    raw_counts = result.data.meas.get_counts()
    
    # Fix Qiskit Endianness
    counts = {k[::-1]: v for k, v in raw_counts.items()}
    total_shots = sum(counts.values())
    sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}

    print("\n--- SIMULATION RESULTS ---")
    print(f"{'Bitstring':<10} | {'QUBO Value':<12} | {'Probability':<12} | Portfolio")
    print("-" * 75)
    
    for bitstring, count in list(sorted_counts.items())[:5]:
        x = np.array(list(bitstring), dtype=int)
        qubo_val = float(x @ Q @ x)
        selected = [tickers[i] for i in range(len(tickers)) if x[i] == 1]
        prob = count / total_shots
        print(f"{bitstring:<10} | {qubo_val:>12.4f} | {prob:>12.1%} | {selected}")

if __name__ == "__main__":
    main()
