"""
Authenticates with IBM Quantum and submits the final optimal QAOA circuit to real hardware.
WARNING: Execution is subject to real-world cloud queue times.
"""

import sys
import os
import json
import getpass
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RealSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator

# Make sure we can import from the 'qubo' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qubo.qubo_builder import build_Q_matrix
from qaoa_circuit import create_qaoa_circuit


def authenticate():
    print("Authenticating with IBM Quantum...")
    possible_paths = ["../apikey.json", "apikey.json"]
    api_key_path = None

    for path in possible_paths:
        if os.path.exists(path):
            api_key_path = path
            break

    if api_key_path:
        with open(api_key_path, "r") as f:
            token_data = json.load(f)
            ibm_token = token_data.get("apikey")
    else:
        ibm_token = getpass.getpass("Please paste your IBM Quantum API Token here: ")

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=ibm_token,
        set_as_default=True,
        overwrite=True,
    )
    return QiskitRuntimeService()


def main():
    service = authenticate()

    # --- 1. Data Fetching ---
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
    print(f"\nFetching live data for {tickers}...")
    close_prices = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, period="1y", auto_adjust=True, progress=False)
        close_prices[t] = df["Close"]

    close_prices = close_prices.dropna()
    daily_returns = close_prices.pct_change(fill_method=None).dropna()
    mu = daily_returns.mean().values * 252
    sigma = daily_returns.cov().values * 252

    # --- 2. QUBO Formulation ---
    k = 2
    penalty = 10.0
    Q = build_Q_matrix(mu, sigma, penalty=penalty, k=k)

    # --- 3. Circuit Creation ---
    p = 4
    print(f"\nBuilding QAOA circuit with depth p={p}...")
    qaoa_circuit, cost_hamiltonian, _ = create_qaoa_circuit(Q, p)

    # --- 4. Parameter Optimization (Local CPU) ---
    print("\nRunning COBYLA optimizer locally to save IBM Queue time...")
    estimator = StatevectorEstimator()

    def evaluate_expectation(x):
        pub = (qaoa_circuit, cost_hamiltonian, x)
        result = estimator.run([pub]).result()[0]
        return result.data.evs

    np.random.seed(42)
    initial_point = np.random.rand(qaoa_circuit.num_parameters) * np.pi
    res = minimize(evaluate_expectation, initial_point, method="COBYLA")
    optimal_params = res.x
    print("✅ Local Optimization Complete! Parameters ready.")

    # --- 5. Real Hardware Execution ---
    optimized_circuit = qaoa_circuit.assign_parameters(optimal_params)
    optimized_circuit.measure_all()

    print("\nFinding the least busy quantum computer...")
    backend = service.least_busy(simulator=False, operational=True)
    print(f"✅ Selected Backend: {backend.name}")

    print("\nTranspiling circuit for target hardware (Optimization Level 3)...")
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuit = pass_manager.run(optimized_circuit)

    print("\nSubmitting job to real hardware queue...")
    real_sampler = RealSampler(mode=backend)
    job = real_sampler.run([isa_circuit], shots=1000)

    print("\n🚀 Job successfully submitted!")
    print(f"Job ID: {job.job_id()}")
    print("Dashboard: https://quantum.ibm.com/jobs")
    print(
        "\n(Note: Hardware queues may take hours. "
        "Run 'job = service.job(\"JOB_ID\")' later to retrieve!)"
    )


if __name__ == "__main__":
    main()
