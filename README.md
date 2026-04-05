# ⚛️ Hybrid Quantum-Classical Portfolio Optimization

[![CI](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/ci.yml/badge.svg)](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/ci.yml)
[![Tests](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/test.yml/badge.svg)](https://github.com/johngeorgea157-stack/quantum-portfolio-opt/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/johngeorgea157-stack/quantum-portfolio-opt/branch/main/graph/badge.svg)](https://codecov.io/gh/johngeorgea157-stack/quantum-portfolio-opt)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=ibm&logoColor=white)](https://qiskit.org/)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Real%20Hardware-000000?logo=ibm&logoColor=white)](https://quantum.ibm.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)](https://github.com/johngeorgea157-stack/quantum-portfolio-opt)

> Benchmarking QAOA against classical methods for portfolio optimization on NIFTY/Bank Nifty stocks — includes QUBO formulation, brute-force ground truth, real IBM hardware runs, and honest failure analysis.

This project applies the **Quantum Approximate Optimization Algorithm (QAOA)** to a binary portfolio selection problem using NIFTY/Bank Nifty stocks. The portfolio problem is formulated as a QUBO and solved via QAOA on both Qiskit simulators and real IBM Quantum hardware. Results are benchmarked against greedy selection, simulated annealing, and brute-force optimal — giving a rigorous, honest comparison of where quantum methods stand today versus classical baselines.

> The goal is not to claim quantum advantage, but to **quantify exactly how close (or far) QAOA gets** under real hardware constraints.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Roadmap](#-roadmap)
- [Repository Structure](#-repository-structure)
- [Quickstart](#-quickstart)
- [Results](#-results)
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

## 🗺️ Roadmap

### ⚙️ Phase 1 — Foundations + Setup `Days 1–3`
- [x] Qiskit environment + IBM Quantum account setup
- [x] Basic circuit exercises (Hadamard, measurement, Sampler)
- [ ] Pull 5–8 Bank Nifty stocks via `yfinance`
- [ ] Compute daily returns and covariance matrix
- [ ] Visualize correlation heatmap
- [ ] Mean-variance optimization + efficient frontier (classical baseline)

### 🧮 Phase 2 — QUBO Formulation `Days 4–6`
- [ ] Convert portfolio problem to QUBO mathematically
- [ ] Build Q matrix in Python and validate with random bitstrings
- [ ] Solve QUBO via **brute force** (ground truth benchmark)

### ⚛️ Phase 3 — QAOA Implementation `Days 7–10`
- [ ] Implement cost + mixer Hamiltonians
- [ ] Build QAOA circuit in Qiskit (p-layer parametrized)
- [ ] Run on **Aer simulator** — extract bitstrings and probabilities
- [ ] Run on **real IBM Quantum hardware** — compare noise distortion

### 🧪 Phase 4 — Classical Benchmarking `Days 11–12`
- [ ] Implement greedy selection algorithm
- [ ] Implement simulated annealing
- [ ] Compare all methods: return, risk, execution time

### 📊 Phase 5 — Analysis + Insights `Days 13–14`
- [ ] Interpret where QAOA matched / failed vs optimal
- [ ] Plot risk–return frontier, bitstring distributions, comparison table
- [ ] Quantify noise impact: simulator vs real hardware delta

### 🚀 Phase 6 — Portfolio + Showcase `Day 15`
- [ ] Final GitHub repo polish (this README, architecture diagram)
- [ ] Export clean figures to `/results/figures/`
- [ ] Write LinkedIn article: *"Why Quantum is NOT yet better than classical (but will be)"*

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

# 6. Run test suite
pytest --tb=short
```

> **IBM Quantum hardware runs** require an account token. Add it as a GitHub Secret (`IBM_TOKEN`) or set it locally via environment variable before running `qaoa/run_hardware.py`.

---

## 📊 Results

> ⏳ Full results will be populated after Phase 4. Placeholder table below.

| Method | Portfolio Return | Portfolio Risk | Exec Time | Match Optimal? |
|---|---|---|---|---|
| Brute Force | — | — | — | ✅ Ground truth |
| QAOA (Simulator) | — | — | — | — |
| QAOA (Real Hardware) | — | — | — | — |
| Greedy | — | — | — | — |
| Simulated Annealing | — | — | — | — |

---

## 🔄 CI/CD Pipeline

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
