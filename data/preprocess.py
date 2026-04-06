"""
=============================================================================
Day 2 – Finance Basics (preprocess.py)
=============================================================================

WHAT THIS SCRIPT DOES (in plain English):
------------------------------------------
1. Loads the raw daily closing prices downloaded by fetch_data.py.
2. Computes DAILY LOG RETURNS  → "How much did each stock move each day?"
3. Computes the COVARIANCE MATRIX → "Do stocks move together or independently?"
4. Computes the CORRELATION MATRIX → "How strongly are the stocks related (−1 to +1)?"
5. Visualises the CORRELATION HEATMAP → a color-coded grid saved to results/.
6. Saves everything to data/cached/ for use in later quantum-optimization stages.

FINANCE CONCEPTS EXPLAINED:
---------------------------
• Log Return  : r_t = ln(P_t / P_{t-1})
                  Why log? They are time-additive and more statistically normal
                  than simple percentage returns.

• Mean Return (μ): Average daily log return × 252 (trading days)
                  → annualised expected return per asset.

• Covariance (Σ): Measures how two assets move together.
                  Positive → they rise/fall together (bad for diversification).
                  Near zero → they are independent (good for diversification).
                  Σ_ij = cov(r_i, r_j) × 252

• Correlation: Normalised covariance, always between −1 and +1.
               corr_ij = Σ_ij / (σ_i × σ_j)
               +1 → perfect positive link   −1 → perfect negative link

• Why does this feed Quantum Optimization (Day 3+)?
  The QUBO (Quadratic Unconstrained Binary Optimization) problem that QAOA
  solves requires μ (expected returns) and Σ (risk / covariance) as inputs.
  This script produces both.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for consistent visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: pretty-print a DataFrame so it fits the terminal
# ─────────────────────────────────────────────────────────────────────────────
def _print_df(label, df, round_to=4):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(df.round(round_to).to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 – Load raw closing prices
# ─────────────────────────────────────────────────────────────────────────────
def load_prices(input_dir):
    """
    Reads every CSV in input_dir, pulls the 'Adj Close' (or 'Close') column,
    aligns all tickers on the same date index, and returns a clean DataFrame.
    
    Why 'Adj Close'?  → It accounts for dividends and stock splits, giving a
    fair comparison of price movements over time.
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{input_dir}'. Run fetch_data.py first."
        )

    adj_close_data = {}
    for file in csv_files:
        ticker = os.path.basename(file).replace(".csv", "")
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True, header=[0, 1])
            # yfinance ≥ 0.2.x saves multi-level headers → flatten
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = pd.read_csv(file, index_col=0, parse_dates=True)

        if "Adj Close" in df.columns:
            adj_close_data[ticker] = df["Adj Close"]
        elif "Close" in df.columns:
            adj_close_data[ticker] = df["Close"]
        else:
            print(f"  ⚠  Could not find Close prices for {ticker}. Columns: {list(df.columns)}")

    if not adj_close_data:
        raise ValueError("No valid price data could be loaded from any CSV file.")

    # Align all tickers on the same trading-day index
    prices_df = pd.DataFrame(adj_close_data)
    prices_df.sort_index(inplace=True)

    # Forward-fill gaps (e.g. halted trading), then drop any remaining NaNs
    prices_df.ffill(inplace=True)
    prices_df.dropna(inplace=True)

    print(f"\n✔  Loaded prices for {prices_df.shape[1]} assets over {prices_df.shape[0]} trading days.")
    print(f"   Date range : {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
    print(f"   Assets     : {list(prices_df.columns)}")
    return prices_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Compute Daily Log Returns
# ─────────────────────────────────────────────────────────────────────────────
def compute_returns(prices_df):
    """
    Formula : r_t = ln( P_t / P_{t-1} )

    Why log returns?
    • Symmetry  : +100% gain and −50% loss are equal in magnitude.
    • Additive  : Weekly return = sum of daily log returns (easy algebra).
    • Normal-ish: Log returns are closer to a normal distribution → required
                  by many portfolio models (Markowitz, QAOA QUBO, etc.).

    The first row becomes NaN (no previous price) and is dropped.
    """
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    print(f"\n✔  Computed daily log returns — shape: {returns_df.shape}")
    _print_df("Daily Log Returns (last 5 rows)", returns_df.tail())
    return returns_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Compute Annualised Mean Returns (μ) and Covariance Matrix (Σ)
# ─────────────────────────────────────────────────────────────────────────────
def compute_statistics(returns_df, annual_factor=252):
    """
    μ  (mu)    = mean daily return  × 252
                 → annualised expected return per asset.
                 Used in QUBO as the 'reward' term.

    Σ (sigma)  = daily covariance matrix  × 252
                 → annualised risk.  Shape: (n_assets, n_assets).
                 Σ_ii is the variance of asset i  (=σ_i²).
                 Σ_ij is how much assets i and j move together.
                 Used in QUBO as the 'risk/penalty' term.

    annual_factor = 252  (standard: approx. number of trading days per year)
    """
    mu    = returns_df.mean()   * annual_factor   # Series  (n_assets,)
    sigma = returns_df.cov()    * annual_factor   # DataFrame (n_assets, n_assets)

    print(f"\n✔  Annualised Mean Returns (μ)  [annual_factor={annual_factor}]")
    _print_df("μ — Expected Annual Return per Asset", mu.to_frame(name="μ (annual)"))

    print(f"\n✔  Annualised Covariance Matrix (Σ)")
    _print_df("Σ — Covariance Matrix", sigma)

    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Compute Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
def compute_correlation(returns_df):
    """
    corr_ij = Σ_ij / (σ_i × σ_j)   ← normalised covariance

    Values always lie in [−1, +1]:
      +1  → asset i and j always move in the same direction
       0  → no linear relationship (good diversification candidate)
      −1  → asset i and j always move in opposite directions

    The diagonal is always 1.0 (every stock is perfectly correlated with itself).

    Portfolio insight:
      A diagonal of 1s and off-diagonals close to 0 means all stocks are
      independent → maximum diversification → minimum portfolio risk.
    """
    corr = returns_df.corr()
    print(f"\n✔  Correlation Matrix")
    _print_df("Correlation Matrix (−1 to +1)", corr)
    
    # Additional analysis
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_flat = corr.values[mask]
    
    print(f"\n✔  Correlation Statistics:")
    print(f"   Highest correlation:  {corr_flat.max():.4f}")
    print(f"   Lowest correlation:   {corr_flat.min():.4f}")
    print(f"   Mean correlation:     {corr_flat.mean():.4f}")
    
    if corr_flat.mean() > 0.6:
        print("\n💡 Portfolio Insight: HIGH average correlation → Limited diversification")
    elif corr_flat.mean() > 0.4:
        print("\n💡 Portfolio Insight: MODERATE average correlation → Some diversification")
    else:
        print("\n💡 Portfolio Insight: LOW average correlation → Good diversification")
    
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.5 – Visualize Price Evolution
# ─────────────────────────────────────────────────────────────────────────────
def plot_price_evolution(prices_df, results_dir):
    """
    Normalize prices and plot their evolution over time for easy comparison.
    """
    if prices_df.empty:
        print("⚠  Cannot plot prices: DataFrame is empty")
        return None
    
    os.makedirs(results_dir, exist_ok=True)
    
    prices_normalized = prices_df / prices_df.iloc[0] * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in prices_normalized.columns:
        ax.plot(prices_normalized.index, prices_normalized[col], label=col, linewidth=2, alpha=0.8)
    
    ax.set_title("Normalized Price Evolution (Base 100) – Bank Nifty Stocks (2 Years)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized Price Index", fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "price_evolution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n✔  Price evolution plot saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.6 – Visualize Return Distributions
# ─────────────────────────────────────────────────────────────────────────────
def plot_return_distributions(returns_df, results_dir):
    """
    Create histograms of daily log returns for each stock.
    """
    if returns_df.empty:
        print("⚠  Cannot plot returns: DataFrame is empty")
        return None
    
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, stock in enumerate(returns_df.columns):
        axes[idx].hist(returns_df[stock], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].axvline(returns_df[stock].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[idx].set_title(f"{stock} – Daily Log Return Distribution", fontweight='bold')
        axes[idx].set_xlabel("Daily Log Return")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle("Return Distribution Analysis – Bank Nifty Stocks",
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "return_distributions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✔  Return distribution plot saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Visualise Correlation Heatmap and Save
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(corr, results_dir):
    """
    A heatmap turns the correlation matrix into a colour-coded grid:
      • Dark red / warm  → high positive correlation (stocks move together)
      • White / neutral  → near-zero correlation    (independent)
      • Dark blue / cool → negative correlation     (stocks move oppositely)

    Why does this matter for portfolio optimisation?
      Stocks with HIGH correlation add redundant risk — choosing both gives no
      diversification benefit.  QAOA ideally selects a MIX of low-correlation
      assets to minimise portfolio variance for a given expected return.

    The plot is saved to results/correlation_heatmap.png so it can be embedded
    in reports or inspected without re-running the script.
    """
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # diverging palette: red = high correlation, blue = negative correlation
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,          # print the correlation value inside each cell
        fmt=".3f",           # 3 decimal places
        cmap=cmap,
        vmin=-1, vmax=1,     # fix scale to the full −1…+1 range
        center=0,            # white = 0 (no correlation)
        square=True,         # keep cells square for readability
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(
        "Day 2 – Correlation Heatmap\nBank Nifty Stocks (Log Returns, Annualised)",
        fontsize=14, fontweight="bold", pad=18
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "correlation_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✔  Correlation heatmap saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Save All Artefacts to data/cached/
# ─────────────────────────────────────────────────────────────────────────────
def save_artefacts(mu, sigma, corr, returns_df, output_dir):
    """
    Saves everything downstream scripts (QUBO builder, QAOA runner) will need:

    File                    Format   Used by
    ─────────────────────── ──────── ──────────────────────────────────────
    expected_returns.csv    CSV      human inspection / classical benchmark
    covariance_matrix.csv   CSV      human inspection / classical benchmark
    correlation_matrix.csv  CSV      Day 2 analysis / QUBO risk penalty
    mu.npy                  NumPy    qubo/build_qubo.py  (fast array load)
    sigma.npy               NumPy    qubo/build_qubo.py  (fast array load)
    returns.csv             CSV      any future statistical analysis
    """
    os.makedirs(output_dir, exist_ok=True)

    mu.to_csv(os.path.join(output_dir, "expected_returns.csv"))
    sigma.to_csv(os.path.join(output_dir, "covariance_matrix.csv"))
    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    returns_df.to_csv(os.path.join(output_dir, "returns.csv"))

    np.save(os.path.join(output_dir, "mu.npy"),    mu.values)
    np.save(os.path.join(output_dir, "sigma.npy"), sigma.values)

    print(f"\n✔  All artefacts saved to: {output_dir}/")
    print("   Files: expected_returns.csv, covariance_matrix.csv,")
    print("          correlation_matrix.csv, returns.csv, mu.npy, sigma.npy")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: preprocess_data()
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(input_dir, output_dir, results_dir=None):
    """
    Master function that runs Steps 0–5 in order.
    Called by __main__ below and importable by other modules (e.g. notebooks).

    Parameters
    ----------
    input_dir   : path to data/raw/      (CSV files from fetch_data.py)
    output_dir  : path to data/cached/   (processed artefacts)
    results_dir : path to results/        (heatmap PNG and other visualizations)
    """
    print("\n" + "═"*70)
    print("  DAY 2 – FINANCE BASICS  |  preprocess.py")
    print("═"*70)

    # Step 0 – Load prices
    prices_df = load_prices(input_dir)

    # Step 1 – Compute returns
    returns_df = compute_returns(prices_df)

    # Step 2 – Compute μ & Σ
    mu, sigma = compute_statistics(returns_df)

    # Step 3 – Compute correlation
    corr = compute_correlation(returns_df)

    # Step 3.5 – Visualizations (only if results_dir specified)
    if results_dir:
        plot_price_evolution(prices_df, results_dir)
        plot_return_distributions(returns_df, results_dir)
        plot_correlation_heatmap(corr, results_dir)

    # Step 4 – Save all artefacts
    save_artefacts(mu, sigma, corr, returns_df, output_dir)

    print("\n" + "═"*70)
    print("  ✔ Done!  Day 2 preprocessing complete.")
    print("  📊 Visualizations saved to results/")
    print("  💾 Data cached in data/cached/ for downstream use")
    print("  🚀 Next → Day 3: Build the QUBO matrix using μ and Σ.")
    print("═"*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir    = os.path.dirname(__file__)
    input_dir   = os.path.join(base_dir, "raw")
    output_dir  = os.path.join(base_dir, "cached")
    results_dir = os.path.join(base_dir, "..", "results")

    print(f"Input  (raw CSVs) : {input_dir}")
    print(f"Output (cached)   : {output_dir}")
    print(f"Results (heatmap) : {results_dir}")

    preprocess_data(input_dir, output_dir, results_dir)
