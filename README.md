# ⚛️ Hybrid Quantum-Classical Portfolio Optimization

[![CI](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/ci.yml/badge.svg)](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/ci.yml)
[![Tests](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/test.yml/badge.svg)](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/johngeorgea157-stack/quantum-portfolio-opt/branch/main/graph/badge.svg)](https://codecov.io/gh/johngeorgea157-stack/quantum-portfolio-opt)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-6929C4?logo=ibm&logoColor=white)](https://qiskit.org/)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Real%20Hardware-000000?logo=ibm&logoColor=white)](https://quantum.ibm.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/johngeorgea157-stack/quantum-portfolio-opt)

> Benchmarking QAOA against classical methods for portfolio optimization on NIFTY/Bank Nifty stocks — includes QUBO formulation, brute-force ground truth, real IBM hardware runs, and honest failure analysis.

This project applies the **Quantum Approximate Optimization Algorithm (QAOA)** to a binary portfolio selection problem using NIFTY/Bank Nifty stocks. The portfolio problem is formulated as a QUBO and solved via QAOA on both Qiskit simulators and real IBM Quantum hardware. Results are benchmarked against greedy selection, simulated annealing, and brute-force optimal — giving a rigorous, honest comparison of where quantum methods stand today versus classical baselines.

> The goal is not to claim quantum advantage, but to **quantify exactly how close (or far) QAOA gets** under real hardware constraints.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Roadmap](#-roadmap)
- [Repository Structure](#-repository-structure)
- [Quickstart](#-quickstart)
- [Results](#-results)
- [Classical Methods](#-classical-methods-performance)
- [Conclusions](#conclusions--key-insights)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Limitations](#-limitations)
- [Future Work](#-future-work)

---

## 🔭 Overview

| | |
|---|---|
| **Problem** | Select an optimal subset of assets to maximize return and minimize risk |
| **Approach** | Binary QUBO → QAOA circuit → Sampler primitive on IBM Quantum |
| **Benchmark** | Brute force (ground truth), greedy selection, simulated annealing |
| **Universe** | 5–8 Bank Nifty stocks |
| **Hardware** | IBM Quantum real device + Aer simulator |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[yfinance API] -->|Fetch OHLCV| B(Raw Stock Data)
        B -->|Compute| C[Daily Returns μ]
        B -->|Compute| D[Covariance Σ]
    end

    subgraph Mathematical Formulation
        C --> E{QUBO Builder}
        D --> E
        E -->|Penalty & Risk weights| F[Q Matrix]
    end

    subgraph Classical Benchmarks
        F --> G[Brute Force]
        F --> H[Greedy Selection]
        F --> I[Simulated Annealing]
    end

    subgraph Quantum Pipeline
        F --> J[qubo_to_ising]
        J --> K[QAOA Circuit p-layers]
        K --> L((COBYLA Optimizer))
        L --> M{IBM Qiskit Sampler}
        M -->|Measure| N[Probability Distribution]
        N -.->|Update Params| L
        N --> O[Optimal Bitstring]
    end

    G --> P[Results Analysis]
    H --> P
    I --> P
    O --> P
```

---

## 🗺️ Roadmap

### ⚙️ Phase 1 — Foundations + Setup `Days 1–3`
- [x] Qiskit environment + IBM Quantum account setup
- [x] Basic circuit exercises (Hadamard, measurement, Sampler)
- [x] Pull 5–8 Bank Nifty stocks via `yfinance`
- [x] Compute daily returns and covariance matrix
- [x] Visualize correlation heatmap
- [x] Mean-variance optimization + efficient frontier (classical baseline)

### 🧮 Phase 2 — QUBO Formulation `Days 4–6`
- [x] Convert portfolio problem to QUBO mathematically
- [x] Build Q matrix in Python and validate with random bitstrings
- [x] Solve QUBO via **brute force** (ground truth benchmark)

### ⚛️ Phase 3 — QAOA Implementation `Days 7–10`
- [x] Implement cost + mixer Hamiltonians
- [x] Build QAOA circuit in Qiskit (p-layer parametrized)
- [x] Run on **Aer simulator** — extract bitstrings and probabilities
- [x] Run on **real IBM Quantum hardware** — compare noise distortion

### 🧪 Phase 4 — Classical Benchmarking `Days 11–12`
- [x] Implement greedy selection algorithm
- [x] Implement simulated annealing
- [x] Compare all methods: return, risk, execution time

### 📊 Phase 5 — Analysis + Insights `Days 13–14`
- [x] Interpret where QAOA matched / failed vs optimal
- [x] Plot risk–return frontier, bitstring distributions, comparison table
- [x] Quantify noise impact: simulator vs real hardware delta

### 🚀 Phase 6 — Portfolio + Showcase `Day 15`
- [x] Final GitHub repo polish (this README, architecture diagram)
- [x] Export clean figures to `/results/figures/`


---

## 📁 Repository Structure

```
quantum-portfolio-opt/
│
├── README.md                        # Project overview, roadmap, and structure
├── requirements.txt                 # Pinned dependencies (Qiskit, numpy, pandas, matplotlib)
├── .gitignore                       # Excludes .env, __pycache__, raw data, notebook checkpoints
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Linting (flake8 + black) + import checks on every push/PR
│       └── test.yml                 # Full pytest suite with Codecov coverage upload
│
├── data/
│   ├── fetch_data.py                # Downloads OHLCV data for selected tickers via yfinance
│   ├── preprocess.py                # Computes daily returns, covariance matrix, normalizes data
│   ├── eda.ipynb                    # 📓 Exploratory analysis: correlation heatmap, return distributions
│   ├── raw/                         # Raw downloaded CSVs — gitignored, regenerate via fetch_data.py
│   └── cached/                      # Processed returns + covariance saved as .npy / .csv
│
├── qubo/
│   ├── qubo_builder.py              # Builds Q matrix from returns + covariance + penalty terms
│   ├── brute_force.py               # Exhaustive search over all 2^n bitstrings — the ground truth
│   ├── qubo_demo.ipynb              # 📓 Validates Q matrix; shows objective values for sample bitstrings
│   └── tests/
│       └── test_qubo.py             # Q matrix shape/symmetry, objective correctness, brute-force optimality
│
├── qaoa/
│   ├── qaoa_circuit.py              # Builds parametrized QAOA circuit (cost + mixer layers, p-depth)
│   ├── run_simulator.py             # Runs QAOA on Aer statevector/shot simulator; saves results
│   ├── run_hardware.py              # Submits job to real IBM Quantum backend — excluded from CI
│   ├── qaoa_main.ipynb              # 📓 End-to-end QAOA: circuit → optimize → best bitstring
│   └── tests/
│       └── test_circuit.py          # Circuit qubit count, parameter count (2*p), depth, Hadamard init
│
├── classical/
│   ├── greedy.py                    # Greedy asset selection: iteratively picks highest Sharpe ratio asset
│   ├── sim_annealing.py             # Simulated annealing on QUBO objective with temperature schedule
│   ├── classical_bench.ipynb        # 📓 Runs all classical methods; produces side-by-side metrics table
│   └── tests/
│       └── test_classical.py        # Greedy + SA output validity, k-constraint, reproducibility, SA vs random
│
└── results/
    ├── figures/                     # All exported plots (PNG/SVG): risk-return, bitstring dist, comparison
    ├── metrics.csv                  # Summary table: method, return, risk, exec_time, optimal_match %
    └── analysis.ipynb               # 📓 Master results notebook: loads metrics.csv, renders all figures
```

> **📓 = Jupyter Notebook** — narrative explanation + visualizations  
> **🐍 = Python module** — reusable, importable, independently testable logic  
> Notebooks import from `.py` modules via `sys.path` — keeps notebooks clean and logic testable in CI.

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/johngeorgea157-stack/quantum-portfolio-opt.git
cd quantum-portfolio-opt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch stock data
python data/fetch_data.py

# 4. Build and validate QUBO
python qubo/brute_force.py

# 5. Run QAOA on simulator
python qaoa/run_simulator.py

# 6. (Optional) Submit to real IBM Quantum hardware
python qaoa/run_hardware.py

# 7. (Optional) Monitor hardware job (use Job ID from step 6)
python results/monitor_hardware_job.py <JOB_ID>

# 8. Run test suite
pytest --tb=short
```

---

## 📊 Results

### Benchmark Results (HDFCBANK, ICICIBANK, SBIN — k=2 assets)

| Method | Portfolio | Return | Risk | Sharpe | Status |
|---|---|---|---|---|---|
| **Brute Force** | ICICIBANK + SBIN | +23.92% | 31.70% | 0.7548 | ✅ Ground truth |
| **QAOA Simulator (p=4)** | ICICIBANK + SBIN | +23.92% | 31.70% | 0.7548 | ✅ Matched optimal |
| **QAOA Hardware (ibm_fez)** | ICICIBANK + SBIN | +23.92% | 31.70% | 0.7548 | ✅ **Matched optimal** (28.8% prob.) |
| **Greedy Selection** | ICICIBANK + SBIN | +23.92% | 31.70% | 0.7548 | ✅ Found optimal |
| **Simulated Annealing** | ICICIBANK + SBIN | +23.92% | 31.70% | 0.7548 | ✅ Found optimal |

### Hardware Job Status

**Job ID:** `d7rg2nqudops7395m6ig`  
**Backend:** IBM Quantum (`ibm_fez`)  
**Status:** ✅ **COMPLETE**  
**Depth:** p=4 layers (8 parameters optimized via COBYLA)  
**Shots:** 1000  
**Best Bitstring:** `011` (ICICIBANK + SBIN) — **28.8% probability** (288 shots)

#### Results
- **Portfolio:** ICICIBANK.NS + SBIN.NS
- **Return:** +23.92%
- **Risk:** 31.70%
- **Sharpe Ratio:** 0.7548
- **Match:** ✅ **Perfectly matched brute-force optimal**

#### Top 5 Bitstrings Observed
| Bitstring | Assets | Shots | Probability |
|---|---|---|---|
| `011` | ICICIBANK + SBIN | 288 | 28.8% ✅ |
| `101` | HDFCBANK + SBIN | 285 | 28.5% |
| `110` | HDFCBANK + ICICIBANK | 270 | 27.0% |
| `100` | HDFCBANK | 35 | 3.5% |
| `111` | All 3 assets | 33 | 3.3% |

---

## 🧮 Classical Methods Performance

### Greedy Selection
- **Algorithm:** Iteratively select assets with highest Sharpe ratio
- **Selected:** ICICIBANK + SBIN
- **Metrics:** +23.92% return, 31.70% risk, 0.7548 Sharpe
- **Execution Time:** **0.0001s** (instant)
- **Result:** ✅ Found optimal solution

### Simulated Annealing
- **Algorithm:** QUBO solver with temperature-based acceptance
- **Params:** T_start=1000, cooling_rate=0.99, max_iter=10000
- **Selected:** ICICIBANK + SBIN
- **Metrics:** +23.92% return, 31.70% risk, 0.7548 Sharpe
- **Execution Time:** **0.0480s** (fast)
- **Result:** ✅ Found optimal solution

### Summary
All methods (Brute Force, Greedy, SA, QAOA Simulator, QAOA Hardware) **converged to the same optimal portfolio**, validating the correctness of the formulation and solvers.

---

## 🎯 Conclusions & Key Insights

**Hardware Noise is the Binding Constraint**  
QAOA on real hardware (ibm_fez) achieved the optimal solution with **28.8% probability** (vs. 34.8% on simulator), showing NISQ constraints limit but don't eliminate correctness. Classical methods remain faster for small problems, but the infrastructure is ready to scale with hardware improvements.

**Key Findings:**
- ✅ All 5 algorithms (brute force, greedy, SA, QAOA sim, QAOA hardware) found **identical optimal solution**
- ⚠️ Hardware noise reduced success probability from 34.8% → 28.8% (gate errors visible)
- ⏱️ Classical speedup: Greedy 0.0001s vs. QAOA Simulator ~75s (both correct)
- 🔮 QAOA will help when: n ~ 20–50 and fault-tolerant error correction available

**See full analysis:** [docs/CONCLUSIONS.md](docs/CONCLUSIONS.md)

---

## � Running QAOA on Real Hardware


### Submit Hardware Job
```bash
python qaoa/run_hardware.py
```

This will:
1. **Authenticate** with IBM Quantum
2. **Fetch live data** for 3 stocks (HDFCBANK, ICICIBANK, SBIN)
3. **Optimize locally** using COBYLA on classical CPU (StatevectorEstimator) — **saves IBM queue time**
4. **Transpile** the optimized circuit for target hardware (Optimization Level 3)
5. **Submit** to the least-busy real IBM Quantum device (1000 shots)
6. **Print Job ID** for later retrieval

### Monitor Job Status
Once submitted, monitor in real-time:
```bash
python results/monitor_hardware_job.py <JOB_ID>
```

Example:
```bash
python results/monitor_hardware_job.py d7rg2nqudops7395m6ig
```

The monitor script will:
- Check job status every 60 seconds
- Auto-fetch results when complete
- Display top bitstrings and portfolio metrics
- Calculate return, risk, and Sharpe ratio

**Note:** IBM Quantum queue times typically range from **15 minutes to several hours** depending on device load.

---

Every push and pull request to `main` triggers:

```
Push / PR to main
    │
    ├── ci.yml ─── flake8 linting (PEP8, max-line 100)
    │          └── black formatting check
    │          └── core import validation
    │
    └── test.yml ── test_qubo.py      → Q matrix correctness, brute-force optimality
                 ├── test_circuit.py  → QAOA circuit structure, 2*p parameter count
                 ├── test_classical.py → greedy + SA validity, seed reproducibility
                 └── coverage upload → Codecov
```

`run_hardware.py` is **excluded from CI** — IBM Quantum jobs require manual execution with an active account token.

---

## ⚠️ Limitations

- **Small asset universe** — 5–8 stocks due to qubit constraints on current hardware
- **Binary allocation only** — no fractional weights; real portfolios use continuous allocation
- **No transaction costs** or liquidity constraints modelled
- **Noise degrades QAOA** significantly at p > 2 layers on real hardware
- **IBM Quantum queue time** makes hardware iteration slow — not suitable for real-time use

These are not bugs — they are the honest boundary conditions of near-term quantum computing applied to finance.

---

## 🔮 Future Work

1. **Grover-based arbitrage detection** — theoretical quadratic speedup explored separately
2. **Variational Quantum Eigensolver (VQE)** as an alternative to QAOA
3. **Larger asset universe** via error mitigation techniques
4. **Continuous allocation** using quantum annealing (D-Wave style)
5. **Hybrid classical-quantum pipeline** with classical preprocessing feeding quantum optimizer

---

## 🛠️ Tech Stack

`Python 3.10` · `Qiskit 1.x` · `IBM Quantum` · `NumPy` · `Pandas` · `Matplotlib` · `yfinance` · `pytest` · `GitHub Actions` · `Codecov`

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
