"""
Day 3 – Classical Portfolio Theory: Mean-Variance Optimization (Markowitz Model)

This module implements the classical Markowitz mean-variance portfolio optimization.
It provides functions to:
- Optimize portfolio weights for minimum variance given a target return
- Generate the efficient frontier (set of optimal portfolios)
- Compare continuous vs binary allocation approaches

The implementation uses scipy.optimize for quadratic programming.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os


def load_portfolio_data(cached_dir='/Users/johngeorgealexander/qc/quantum-portfolio-opt/data/cached'):
    """
    Load portfolio data from Day 2 cached results.

    Returns:
        mu (np.array): Expected returns vector
        sigma (np.array): Covariance matrix
        stock_names (list): List of stock names
    """
    # Load mean returns (μ) and covariance (Σ)
    mu = np.load(os.path.join(cached_dir, 'mu.npy'))
    sigma = np.load(os.path.join(cached_dir, 'sigma.npy'))

    # Load stock names from CSV
    expected_returns_df = pd.read_csv(os.path.join(cached_dir, 'expected_returns.csv'), index_col=0)
    stock_names = expected_returns_df.index.tolist()

    return mu, sigma, stock_names


def portfolio_variance(w, sigma):
    """Calculate portfolio variance: w^T Σ w"""
    return w.T @ sigma @ w


def portfolio_return(w, mu):
    """Calculate portfolio expected return: w^T μ"""
    return w.T @ mu


def optimize_portfolio(mu, sigma, target_return):
    """
    Solve the Markowitz mean-variance optimization problem.
    Minimize variance subject to target return and full investment.

    Args:
        mu (np.array): Expected returns vector
        sigma (np.array): Covariance matrix
        target_return (float): Target portfolio return

    Returns:
        tuple: (optimal_weights, achieved_return, achieved_variance) or (None, None, None) if failed
    """
    n_assets = len(mu)

    # Initial guess: equal weights
    w0 = np.ones(n_assets) / n_assets

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},  # weights sum to 1
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_return}  # target return
    ]

    # Bounds: no short selling (w_i >= 0)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Minimize portfolio variance
    result = minimize(
        portfolio_variance,
        w0,
        args=(sigma,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False}
    )

    if result.success:
        w_opt = result.x
        variance_opt = portfolio_variance(w_opt, sigma)
        return_opt = portfolio_return(w_opt, mu)
        return w_opt, return_opt, variance_opt
    else:
        print(f"Optimization failed for target return {target_return:.4f}")
        return None, None, None


def generate_efficient_frontier(mu, sigma, n_points=50):
    """
    Generate points on the efficient frontier by solving for minimum variance
    portfolios across a range of target returns.

    Args:
        mu (np.array): Expected returns vector
        sigma (np.array): Covariance matrix
        n_points (int): Number of points on the frontier

    Returns:
        list: List of dictionaries with 'return', 'variance', 'volatility', 'weights'
    """
    # Define range of target returns
    min_return = np.min(mu)
    max_return = np.max(mu)
    target_returns = np.linspace(min_return, max_return, n_points)

    # Store results
    portfolios = []

    for target_ret in target_returns:
        w_opt, ret_opt, var_opt = optimize_portfolio(mu, sigma, target_ret)
        if w_opt is not None:
            portfolios.append({
                'return': ret_opt,
                'variance': var_opt,
                'volatility': np.sqrt(var_opt),
                'weights': w_opt
            })

    return portfolios


def get_efficient_frontier_data(mu, sigma, n_points=30):
    """
    Get efficient frontier data for plotting or analysis.

    Returns:
        tuple: (returns_list, volatilities_list, portfolios_list)
    """
    portfolios = generate_efficient_frontier(mu, sigma, n_points)
    returns = [p['return'] for p in portfolios]
    volatilities = [p['volatility'] for p in portfolios]
    return returns, volatilities, portfolios


# Example usage
if __name__ == "__main__":
    # Load data
    mu, sigma, stock_names = load_portfolio_data()

    print("=" * 70)
    print("MEAN-VARIANCE OPTIMIZATION DEMO")
    print("=" * 70)
    print(f"Assets: {', '.join(stock_names)}")

    # Test optimization
    target_return = 0.05  # 5% annual return
    w_opt, ret_opt, var_opt = optimize_portfolio(mu, sigma, target_return)

    if w_opt is not None:
        print(f"\nTarget return: {target_return:.4f}")
        print(f"Optimized return: {ret_opt:.4f}")
        print(f"Optimized volatility: {np.sqrt(var_opt):.4f}")

        print("\nOptimal weights:")
        for name, weight in zip(stock_names, w_opt):
            print(f"  {name:12}: {weight:>8.4f}")

    # Generate efficient frontier
    portfolios = generate_efficient_frontier(mu, sigma, n_points=20)
    if portfolios:
        print(f"\nGenerated {len(portfolios)} portfolios on efficient frontier")
        print(f"Return range: {min(p['return'] for p in portfolios):.4f} to {max(p['return'] for p in portfolios):.4f}")
        print(f"Volatility range: {min(p['volatility'] for p in portfolios):.4f} to {max(p['volatility'] for p in portfolios):.4f}")

    print("=" * 70)