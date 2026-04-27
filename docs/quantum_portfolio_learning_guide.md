# Quantum Portfolio Optimization — Complete Learning Guide

**Author**: John George Alexander  
**Project**: Hybrid Quantum-Classical Portfolio Optimization  
**Repository**: [github.com/johngeorgea157-stack/quantum-portfolio-opt](https://github.com/johngeorgea157-stack/quantum-portfolio-opt)

---

## Table of Contents

1. [Classical Portfolio Theory](#1-classical-portfolio-theory)
2. [QUBO — The Mathematical Bridge](#2-qubo--the-mathematical-bridge)
3. [Quantum Computing Models](#3-quantum-computing-models)
4. [Hamiltonians — The Language of Quantum Physics](#4-hamiltonians--the-language-of-quantum-physics)
5. [QAOA — The Algorithm](#5-qaoa--the-algorithm)
6. [How QUBO & QAOA Are Used in Our Code](#6-how-qubo--qaoa-are-used-in-our-code)
7. [Repository Pipeline Overview](#7-repository-pipeline-overview)
8. [Why Quantum is NOT Yet Better Than Classical (But Will Be)](#8-why-quantum-is-not-yet-better-than-classical-but-will-be)
9. [Limitations & Future Improvements](#9-limitations--future-improvements)
10. [Key Formulas Reference](#10-key-formulas-reference)

---

## 1. Classical Portfolio Theory

### The Markowitz Mean-Variance Model (1952)

Harry Markowitz showed that a rational investor should not just pick high-return stocks — they should consider the *covariance* between assets. Two assets that are individually volatile can together produce a stable portfolio if their movements cancel out.

**The Optimization Problem:**

Given `n` assets, find weights `w` that:

$$\min_w \quad \frac{1}{2} \mathbf{w}^T \Sigma \mathbf{w}$$

Subject to:
- $\mathbf{w}^T \boldsymbol{\mu} = \mu_{\text{target}}$ (target return)  
- $\mathbf{w}^T \mathbf{1} = 1$ (weights sum to 1)  
- $w_i \geq 0$ (no short selling)  

Where:
- $\boldsymbol{\mu}$ = vector of expected returns (annualized)
- $\Sigma$ = covariance matrix (annualized)
- $\mathbf{w}$ = portfolio weight vector

### The Efficient Frontier

By solving the above for every possible target return, we trace out the **Efficient Frontier** — the curve of optimal portfolios where you cannot improve return without increasing risk, or reduce risk without sacrificing return.

> **Key Insight for Quantum**: Classical Markowitz uses *continuous* weights ($w_i \in [0, 1]$). Quantum optimization uses *binary* selection ($x_i \in \{0, 1\}$). This creates a fundamental "discretization gap" — the quantum formulation can only reach discrete points, not the full continuous frontier.

---

## 2. QUBO — The Mathematical Bridge

### What is QUBO?

**QUBO** = Quadratic Unconstrained Binary Optimization.

It is a mathematical framework that rewrites any optimization problem so that:
1. All variables are **binary** ($x_i \in \{0, 1\}$)
2. Constraints are absorbed into the objective as **penalty terms**
3. The objective is at most **quadratic** ($x_i \cdot x_j$ terms)

### Standard Form

$$\min_{\mathbf{x}} \quad f(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x} = \sum_i \sum_j Q_{ij} \cdot x_i \cdot x_j$$

Where $Q$ is an $n \times n$ matrix that encodes the entire problem.

### Our Portfolio QUBO

We have $n$ stocks. $x_i = 1$ means "buy stock $i$", $x_i = 0$ means "skip it".

$$f(\mathbf{x}) = \underbrace{-\sum_i \mu_i x_i}_{\text{Return (negated)}} + \underbrace{q \sum_i \sum_j \sigma_{ij} x_i x_j}_{\text{Risk}} + \underbrace{\lambda(\sum_i x_i - k)^2}_{\text{Cardinality Penalty}}$$

### Building the Q Matrix

After expanding the penalty term $(\sum_i x_i - k)^2$ and using the identity $x_i^2 = x_i$ for binary variables:

**Diagonal entries:**
$$Q_{ii} = -\mu_i + q \cdot \sigma_{ii} + \lambda(1 - 2k)$$

**Off-diagonal entries ($i \neq j$):**
$$Q_{ij} = q \cdot \sigma_{ij} + \lambda$$

Where:
- $\mu_i$ = expected return of stock $i$
- $\sigma_{ij}$ = covariance between stocks $i$ and $j$
- $q$ = risk aversion weight (default 1.0)
- $\lambda$ = penalty strength (typically $\approx 10 \times \max(|\mu|)$)
- $k$ = number of stocks to select

### Reading QUBO Values

The QUBO value $f(\mathbf{x})$ is the "energy" of a portfolio:
- **More negative** = better (high return, low risk, valid cardinality)
- **Less negative or positive** = worse (low return, high risk, or violates $k$ constraint)

---

## 3. Quantum Computing Models

### 3.1 Circuit (Gate-Based) Model

**Hardware**: IBM Quantum, Google Sycamore, Quantinuum

Computation proceeds through **discrete logic gates** applied to qubits in sequence:
- **Hadamard (H)**: Creates superposition ($|0\rangle \to \frac{|0\rangle + |1\rangle}{\sqrt{2}}$)
- **CNOT**: Entangles two qubits
- **Rz, Rx**: Rotations (parameterized gates used in QAOA)

**Properties:**
- Mathematically **universal** — can simulate any quantum algorithm
- Supports Shor's (cryptography), Grover's (search), VQE (chemistry), QAOA (optimization)
- Current limitation: high gate noise, small qubit counts (100–1000 qubits)

### 3.2 Adiabatic Quantum Computation (AQC)

**Hardware**: D-Wave Systems

Instead of logic gates, AQC uses a continuous physical process:
1. **Start** in a simple, known quantum ground state (high-energy superposition)
2. **Slowly** evolve the physical Hamiltonian toward the problem Hamiltonian
3. **End** in the ground state of the problem — which IS the optimal solution

**The Adiabatic Theorem**: If the evolution is slow enough, the system stays in the ground state throughout, guaranteeing the optimal answer.

**Properties:**
- **Not universal** — specialized for optimization (QUBO/Ising) problems only
- Cannot run Shor's or Grover's algorithm
- Scales to thousands of qubits (D-Wave Advantage: 5000+ qubits)
- Natively interprets QUBO as a physical magnetic landscape

### 3.3 The Key Difference

| Feature | Circuit Model | Adiabatic (AQC) |
|:---|:---|:---|
| Execution | Discrete digital gates | Continuous analog physics |
| Universality | Universal (any algorithm) | Optimization only |
| QUBO solving | Via QAOA algorithm | Native (no algorithm needed) |
| Scale today | ~100–1000 qubits | ~5000+ qubits |
| Noise profile | Gate errors accumulate | Analog noise, different character |

---

## 4. Hamiltonians — The Language of Quantum Physics

### What is a Hamiltonian?

A **Hamiltonian** is a matrix (operator) that encodes the total energy of a quantum system. In quantum computing, we use Hamiltonians to define:
- What problem we are solving (Cost Hamiltonian)
- How to explore the solution space (Mixer Hamiltonian)

### Cost Hamiltonian ($H_C$)

The Cost Hamiltonian encodes our QUBO problem directly into quantum mechanics. Each eigenstate (measurement outcome) of $H_C$ corresponds to a bitstring, and its eigenvalue is the QUBO energy of that bitstring.

$$H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{offset}$$

Where:
- $Z_i$ is the Pauli-Z operator on qubit $i$
- $h_i$ and $J_{ij}$ are coefficients derived from the QUBO matrix $Q$
- The ground state (lowest eigenvalue) of $H_C$ = the optimal portfolio

**QUBO → Ising Mapping**: We convert each binary variable $x_i \in \{0,1\}$ to a spin variable $z_i \in \{-1, +1\}$ using $x_i = \frac{1 - z_i}{2}$. This transforms the QUBO matrix into Pauli-Z strings that a quantum computer understands.

### Mixer Hamiltonian ($H_M$)

The Mixer Hamiltonian drives transitions between different bitstrings. It ensures the algorithm explores the full solution space rather than getting stuck.

$$H_M = \sum_i X_i$$

Where $X_i$ is the Pauli-X operator (quantum NOT gate) on qubit $i$. This operator flips $|0\rangle \leftrightarrow |1\rangle$, mixing the probability amplitudes across all possible portfolios.

### The Relationship

- $H_C$ defines **what** we want to find (the lowest energy portfolio)
- $H_M$ defines **how** we search (by flipping qubits to explore new combinations)
- QAOA alternates between these two: "look at the cost" → "explore neighbors" → repeat

---

## 5. QAOA — The Algorithm

### Quantum Approximate Optimization Algorithm

QAOA was invented by Farhi, Goldstone, and Gutmann (2014) to solve combinatorial optimization problems on gate-based quantum computers.

### How It Works (Step by Step)

**Step 1**: Start all qubits in uniform superposition:
$$|\psi_0\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle$$

Every portfolio has equal probability of being measured.

**Step 2**: Apply $p$ alternating layers of:
$$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^{p} \left[ e^{-i\beta_l H_M} \cdot e^{-i\gamma_l H_C} \right] |\psi_0\rangle$$

- $e^{-i\gamma_l H_C}$: "Cost layer" — rotates phases proportional to QUBO energy. Good portfolios accumulate more constructive interference.
- $e^{-i\beta_l H_M}$: "Mixer layer" — allows amplitude to flow between bitstrings, enabling exploration.

**Step 3**: Measure the qubits. The bitstring with highest probability is the approximate optimal portfolio.

**Step 4**: Use a classical optimizer (COBYLA) to tune $\gamma$ and $\beta$ parameters to maximize the probability of measuring the optimal bitstring.

### The Trotterization Connection

### Why $R_z$ and $R_x$ Gates? (The "Digital Chopping")

If we want to simulate the Cost Hamiltonian $H_C$ and Mixer Hamiltonian $H_M$ on a circuit, we have to translate them into standard quantum logic gates.

1. **The Cost Layer ($R_z$ gates)**: 
   $H_C$ is built entirely out of Pauli-Z operators (because the QUBO uses binary variables, and Pauli-Z measures exactly 0 or 1 states). 
   Mathematically, evolving the system by $e^{-i\gamma H_C}$ means applying **$R_z$ rotations** (and $ZZ$ coupling rotations for the $Q_{ij}$ terms) to the qubits. The angle $\gamma$ dictates how far they rotate. These $R_z$ gates don't change the *probability* of a state; they only change its *phase* based on how good the portfolio is.

2. **The Mixer Layer ($R_x$ gates)**: 
   $H_M$ is built entirely out of Pauli-X operators. Evolving the system by $e^{-ieta H_M}$ means applying **$R_x$ rotations** to every qubit. The $R_x$ gate physically flips the state between $|0\rangle$ and $|1\rangle$. The angle $eta$ dictates how aggressively it flips. This is how QAOA explores new portfolios.

Because a circuit-based quantum computer cannot run $H_C$ and $H_M$ at the exact same time (like AQC does), it "chops" the process into alternating digital slices: a burst of $R_z$ gates, then a burst of $R_x$ gates. This is Trotterization.


As $p \to \infty$, QAOA mathematically converges to **exact Adiabatic Quantum Computation**. Each layer becomes an infinitesimal step in the continuous adiabatic evolution. This is called **Trotterization** — QAOA is a digitized (discretized) version of AQC.

### Parameters

- **$p$ (depth)**: Number of alternating Cost/Mixer layers. Higher $p$ = better approximation but deeper circuit = more noise.
- **$\gamma_l$**: Cost layer angles (one per layer)
- **$\beta_l$**: Mixer layer angles (one per layer)
- **Total parameters**: $2p$ (optimized by COBYLA)

---

## 6. How QUBO & QAOA Are Used in Our Code

### Pipeline Flow

```
yfinance data → μ, Σ → build_Q_matrix() → qubo_to_ising() → QAOAAnsatz → COBYLA → Sampler → Results
```

### `qubo/qubo_builder.py` — Build the Q Matrix

```python
def build_Q_matrix(returns, cov, penalty, k, risk_weight=1.0):
    Q[i][i] = -returns[i] + risk_weight * cov[i][i] + penalty * (1 - 2*k)  # diagonal
    Q[i][j] = risk_weight * cov[i][j] + penalty                            # off-diagonal
```

Takes the financial data (μ, Σ) and encodes it into a single matrix $Q$. This matrix IS the portfolio problem.

### `qaoa/qaoa_circuit.py` — QUBO → Quantum Circuit

```python
def qubo_to_ising(Q):
    # Converts Q matrix → SparsePauliOp (Pauli Z strings)
    # Uses: x_i = (1 - Z_i) / 2

def create_qaoa_circuit(Q, p=1):
    hamiltonian, offset = qubo_to_ising(Q)
    circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p)
```

Maps the mathematical Q matrix into quantum physics (Pauli operators), then builds the parameterized QAOA circuit using Qiskit's `QAOAAnsatz`.

### `qaoa/run_simulator.py` — The Optimization Loop

```python
estimator = StatevectorEstimator()

def evaluate_expectation(params):
    return estimator.run([(circuit, hamiltonian, params)]).result()[0].data.evs

res = minimize(evaluate_expectation, initial_point, method='COBYLA')
```

The classical-quantum hybrid loop: COBYLA proposes $\gamma, \beta$ → Qiskit evaluates the quantum expectation value → COBYLA refines → repeat until convergence.

### `qaoa/run_hardware.py` — Real IBM Quantum Execution

Runs the optimization loop locally on a simulator (fast), then submits only the final optimized circuit to IBM hardware (one queue wait instead of thousands).

---

## 7. Repository Pipeline Overview

```
quantum-portfolio-opt/
│
├── data/                          ← Phase 1: Financial Data
│   ├── fetch_data.py              # Downloads stock prices via yfinance
│   ├── preprocess.py              # Computes μ (returns) and Σ (covariance)
│   └── eda.ipynb                  # Exploratory data analysis notebook
│
├── qubo/                          ← Phase 2: Mathematical Formulation
│   ├── qubo_builder.py            # Builds the Q matrix from μ, Σ
│   ├── qubo_demo.ipynb            # Interactive QUBO exploration
│   └── brute_force.py             # Exhaustive search (ground truth)
│
├── qaoa/                          ← Phase 3: Quantum Algorithm
│   ├── qaoa_circuit.py            # QUBO→Ising + QAOAAnsatz builder
│   ├── run_simulator.py           # Full pipeline on local simulator
│   ├── run_hardware.py            # Submit to real IBM Quantum hardware
│   └── qaoa_main.ipynb            # End-to-end QAOA notebook
│
├── classical/                     ← Phase 4: Classical Benchmarks
│   ├── greedy.py                  # Greedy portfolio selection
│   ├── sim_annealing.py           # Simulated annealing on QUBO
│   └── classical_bench.ipynb      # Head-to-head comparison
│
└── results/                       ← Phase 5: Analysis
    ├── analysis.ipynb             # Visualizations + interpretation
    ├── metrics.csv                # Comparison table (auto-generated)
    └── figures/                   # Exported plots
```

### The Data Flow

1. **yfinance** → raw stock prices (1 year of daily close prices)
2. **preprocess.py** → daily returns → annualized μ (returns) and Σ (covariance)
3. **qubo_builder.py** → Q matrix encoding return + risk + cardinality penalty
4. **qaoa_circuit.py** → Q → Ising Hamiltonian → QAOA circuit (parameterized)
5. **COBYLA optimizer** → finds optimal γ, β angles
6. **Sampler** → measures circuit → probability distribution over bitstrings
7. **analysis.ipynb** → compares QAOA vs Brute Force vs Greedy vs SA

---

## 8. Why Quantum is NOT Yet Better Than Classical (But Will Be)

### The Honest Truth Today

At N = 6 stocks (our project scale):
- **Brute Force** checks all 64 combinations in < 0.001 seconds
- **Greedy** finds the optimum in < 0.0001 seconds
- **Simulated Annealing** finds the optimum in < 0.01 seconds
- **QAOA** takes 5–30 seconds (COBYLA iterations + circuit simulation)

**QAOA is 1000× slower and finds the same answer.** There is zero quantum advantage.

### Why It Will Change

The computational complexity tells the real story:

| Method | Time Complexity | N=6 | N=50 | N=1000 |
|:---|:---|:---|:---|:---|
| Brute Force | $O(2^N)$ | 64 | $10^{15}$ | $10^{301}$ |
| Greedy | $O(N^2)$ | 36 | 2,500 | 1,000,000 |
| SA | $O(\text{iterations})$ | Fast | Slow | Very Slow |
| QAOA | $O(\text{poly}(N) \cdot p)$ | Overkill | Feasible | Feasible* |

*Feasible with fault-tolerant quantum hardware (not yet available).

**The crossover point**: Around N = 30–50, brute force becomes impossible. Greedy and SA start failing to find the true optimum (getting trapped in local minima). QAOA's polynomial scaling means it remains tractable — but only if hardware noise is low enough.

### What Needs to Happen

1. **Error Correction**: Current NISQ devices have ~0.1% gate error rates. We need < 0.001% for deep QAOA circuits.
2. **More Qubits**: 1000-stock portfolios need 1000 logical qubits (potentially 1 million physical qubits with error correction).
3. **Faster Classical Optimizers**: The COBYLA loop is a bottleneck. Better variational methods (ADAPT-QAOA, warm-starting) can reduce iterations.

---

## 9. Limitations & Future Improvements

### Current Limitations

1. **Scale**: Only 3–6 stocks tested. Real portfolios have 50–3000 assets.
2. **Binary Selection**: Our QUBO only picks stocks (0 or 1). Real portfolios need continuous weights.
3. **Static Data**: Uses historical returns. Real optimization needs forward-looking estimates.
4. **Noise**: IBM hardware introduces 5–15% degradation in solution quality.
5. **COBYLA Convergence**: Classical optimizer sometimes gets stuck, especially with high $p$.
6. **No Transaction Costs**: Real portfolios must account for brokerage, slippage, taxes.

### Future Improvements

1. **ADAPT-QAOA**: Dynamically grows the circuit instead of using fixed $p$ layers.
2. **Warm-Starting**: Initialize QAOA parameters from a classical solution instead of random.
3. **CVaR Objective**: Replace expected value with Conditional Value-at-Risk for better tail-risk handling.
4. **Continuous Weights**: Use QAOA with multi-qubit encoding to represent fractional weights.
5. **Error Mitigation**: Apply Zero-Noise Extrapolation (ZNE) or Probabilistic Error Cancellation (PEC).
6. **Larger Universe**: Test with 20+ stocks to approach the classical-quantum crossover.

---

## 10. Key Formulas Reference

### Portfolio Theory
- **Expected Return**: $R_p = \mathbf{w}^T \boldsymbol{\mu}$
- **Portfolio Variance**: $\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$
- **Sharpe Ratio**: $S = \frac{R_p - R_f}{\sigma_p}$

### QUBO
- **Objective**: $f(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$
- **Q diagonal**: $Q_{ii} = -\mu_i + q\sigma_{ii} + \lambda(1-2k)$
- **Q off-diagonal**: $Q_{ij} = q\sigma_{ij} + \lambda$

### Ising / Hamiltonian
- **Binary → Spin**: $x_i = \frac{1 - Z_i}{2}$
- **Cost Hamiltonian**: $H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j$
- **Mixer Hamiltonian**: $H_M = \sum_i X_i$

### QAOA
- **State evolution**: $|\psi\rangle = \prod_{l=1}^{p} e^{-i\beta_l H_M} e^{-i\gamma_l H_C} |+\rangle^n$
- **Total parameters**: $2p$ ($p$ gammas + $p$ betas)
- **Convergence**: As $p \to \infty$, QAOA $\to$ exact AQC

### Simulated Annealing
- **Acceptance probability**: $P(\text{accept worse}) = e^{-\Delta E / T}$
- **Cooling schedule**: $T_{n+1} = \alpha \cdot T_n$ (typically $\alpha = 0.99$)

---

*This guide accompanies the quantum-portfolio-opt repository. All code, notebooks, and results are available on GitHub.*
