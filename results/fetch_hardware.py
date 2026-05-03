"""
Fetches the completed job from IBM Quantum and calculates portfolio metrics.
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qubo.qubo_builder import build_Q_matrix


def fetch_and_evaluate(job_id):
    print(f"Connecting to IBM Quantum to fetch Job {job_id}...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    job = service.job(job_id)
    
    status = job.status()
    print(f"Job Status: {status}")
    if str(status) != "DONE":
        print("Job is not finished yet. Please try again later.")
        return

    # Extract the result
    result = job.result()
    pub_result = result[0]
    
    # Get the raw counts
    counts = pub_result.data.meas.get_counts()
    
    # Qiskit SamplerV2 returns bitstrings in standard Qiskit order. 
    # For N=6, we get 6-bit strings.
    best_bs = max(counts, key=counts.get)
    best_bs_reversed = best_bs[::-1]  # Reverse to match our QUBO ordering
    
    print(f"\n✅ Most frequent bitstring from IBM Hardware: {best_bs_reversed}")
    
    # --- Fetch Data & Compute Metrics ---
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
    # We used 3 tickers in run_hardware.py (which means N=3)
    
    print(f"\nFetching live data to evaluate {tickers}...")
    close_prices = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, period="1y", auto_adjust=True, progress=False)
        close_prices[t] = df["Close"]

    close_prices = close_prices.dropna()
    daily_returns = close_prices.pct_change(fill_method=None).dropna()
    mu = daily_returns.mean().values * 252
    sigma = daily_returns.cov().values * 252

    x_vec = np.array(list(best_bs_reversed), dtype=int)
    
    if len(x_vec) != len(mu):
        print(f"Error: Bitstring length {len(x_vec)} does not match number of assets {len(mu)}.")
        print("Make sure you are running with the exact same stocks used in run_hardware.py!")
        return

    ret = float(x_vec @ mu)
    risk = np.sqrt(float(x_vec @ sigma @ x_vec))
    
    print("\n" + "="*50)
    print("📈 IBM Quantum Hardware Results")
    print("="*50)
    print(f"Portfolio Vector : {x_vec}")
    print(f"Expected Return  : {ret*100:.2f}%")
    print(f"Portfolio Risk   : {risk*100:.2f}%")
    print("="*50)
    print("\nUpdate your README.md table with these values!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_hardware.py <JOB_ID>")
        sys.exit(1)
        
    job_id = sys.argv[1]
    fetch_and_evaluate(job_id)
