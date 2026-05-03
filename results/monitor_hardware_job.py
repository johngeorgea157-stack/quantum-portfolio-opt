"""
Monitors the status of a submitted IBM Quantum hardware job and fetches results when ready.
Usage: python monitor_hardware_job.py d7rg2nqudops7395m6ig
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qubo.qubo_builder import build_Q_matrix


def monitor_and_fetch(job_id, check_interval=60):
    """
    Continuously monitors job status and fetches results when complete.
    
    Parameters
    ----------
    job_id : str
        IBM Quantum job ID
    check_interval : int
        Time in seconds between status checks (default: 60)
    """
    print(f"\n{'='*60}")
    print(f"IBM Quantum Job Monitor")
    print(f"{'='*60}")
    print(f"Job ID: {job_id}")
    print(f"Check Interval: {check_interval}s")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    
    while True:
        try:
            job = service.job(job_id)
            status = job.status()
            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Status: {status}")
            
            if str(status) == "DONE":
                print("\n✅ Job Complete! Fetching results...\n")
                return fetch_and_evaluate(job)
            elif str(status) in ["CANCELLED", "ERROR"]:
                print(f"\n❌ Job {status}. Cannot retrieve results.")
                return None
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\n⏸️  Monitoring paused. Run this script again with the same Job ID to resume.")
            return None
        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(check_interval)


def fetch_and_evaluate(job):
    """
    Extracts results from completed job and calculates portfolio metrics.
    """
    try:
        result = job.result()
        pub_result = result[0]
        
        # Get the raw counts from measurement
        counts = pub_result.data.meas.get_counts()
        
        # Find most frequent bitstring
        best_bs = max(counts, key=counts.get)
        best_bs_reversed = best_bs[::-1]  # Reverse to match QUBO ordering
        best_count = counts[best_bs]
        total_shots = sum(counts.values())
        prob = best_count / total_shots
        
        print(f"Most Frequent Bitstring: {best_bs_reversed}")
        print(f"Occurrence: {best_count}/{total_shots} shots ({prob*100:.1f}%)")
        print()
        
        # --- Fetch Data & Compute Metrics ---
        tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
        
        print(f"Fetching live data for {tickers}...")
        close_prices = pd.DataFrame()
        for t in tickers:
            df = yf.download(t, period="1y", auto_adjust=True, progress=False)
            close_prices[t] = df["Close"]

        close_prices = close_prices.dropna()
        daily_returns = close_prices.pct_change(fill_method=None).dropna()
        mu = daily_returns.mean().values * 252
        sigma = daily_returns.cov().values * 252

        # Build QUBO to get penalty info
        k = 2
        penalty = 10.0
        Q = build_Q_matrix(mu, sigma, penalty=penalty, k=k)
        
        x_vec = np.array(list(best_bs_reversed), dtype=int)
        selected_assets = sum(x_vec)
        
        # Calculate metrics
        ret = float(x_vec @ mu)
        risk = np.sqrt(float(x_vec @ sigma @ x_vec))
        qubo_val = float(x_vec @ Q @ x_vec)
        
        selected_tickers = [tickers[i] for i in range(len(tickers)) if x_vec[i] == 1]
        
        print("\n" + "="*60)
        print("📊 IBM QUANTUM HARDWARE RESULTS")
        print("="*60)
        print(f"Job ID                : {job.job_id()}")
        print(f"Backend               : {job.backend}")
        print(f"Portfolio Vector      : {x_vec}")
        print(f"Selected Assets       : {selected_tickers}")
        print(f"Number of Assets      : {selected_assets}")
        print(f"Expected Return (μ)   : {ret*100:>8.2f}%")
        print(f"Portfolio Risk (σ)    : {risk*100:>8.2f}%")
        print(f"Sharpe Ratio          : {ret/risk if risk > 0 else 0:>8.4f}")
        print(f"QUBO Objective Value  : {qubo_val:>8.4f}")
        print("="*60)
        
        # Print top bitstrings observed
        print("\nTop 5 Bitstrings Observed:")
        print(f"{'Bitstring':<12} | {'Count':<6} | {'Prob':<8} | Selected Assets")
        print("-" * 65)
        
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for bs, count in sorted_counts[:5]:
            bs_reversed = bs[::-1]
            prob_pct = (count / total_shots) * 100
            x = np.array(list(bs_reversed), dtype=int)
            selected = [tickers[i] for i in range(len(tickers)) if x[i] == 1]
            print(f"{bs_reversed:<12} | {count:<6} | {prob_pct:>6.1f}% | {selected}")
        
        print("\n" + "="*60)
        print("✅ Results ready to add to README.md!")
        print("="*60)
        
        return {
            "job_id": job.job_id(),
            "bitstring": best_bs_reversed,
            "probability": prob,
            "assets": selected_tickers,
            "return": ret,
            "risk": risk,
            "sharpe": ret/risk if risk > 0 else 0,
            "qubo_value": qubo_val
        }
        
    except Exception as e:
        print(f"❌ Error fetching results: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_hardware_job.py <JOB_ID>")
        print("\nExample: python monitor_hardware_job.py d7rg2nqudops7395m6ig")
        sys.exit(1)
    
    job_id = sys.argv[1]
    monitor_and_fetch(job_id)
