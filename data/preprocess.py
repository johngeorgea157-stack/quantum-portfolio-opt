import pandas as pd
import numpy as np
import os
import glob

def preprocess_data(input_dir, output_dir):
    """
    Reads raw CSV data, computes log returns, mean returns vector (mu) and covariance matrix (Sigma).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Run fetch_data.py first.")
        return

    # Dictionary to hold adjusted close prices for each ticker
    adj_close_data = {}
    
    for file in csv_files:
        ticker = os.path.basename(file).replace(".csv", "")
        # read_csv on yfinance output might have a multi-row header or single-row header
        # Let's specify header=0 to be safe, dropping extra lines if necessary
        # We will use header=0 first and check if 'Adj Close' exists
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            # Usually yfinance columns are capitalised
            if 'Adj Close' in df.columns:
                adj_close_data[ticker] = df['Adj Close']
            elif 'Close' in df.columns:
                adj_close_data[ticker] = df['Close']
            else:
                print(f"Warning: Columns of {ticker}: {df.columns}. Could not find Close prices.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not adj_close_data:
        print("No valid price data could be loaded.")
        return

    # Create a unified dataframe of all prices, aligning on index
    prices_df = pd.DataFrame(adj_close_data)
    
    # Forward fill missing data and drop the rest
    prices_df.ffill(inplace=True)
    prices_df.dropna(inplace=True)
    
    # Calculate daily log returns
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    
    # Calculate annualized expected returns (mu) and covariance matrix (Sigma)
    # Assuming 252 trading days in a year
    annual_factor = 252
    mu = returns_df.mean() * annual_factor
    sigma = returns_df.cov() * annual_factor
    
    # Save processed data
    mu.to_csv(os.path.join(output_dir, "expected_returns.csv"))
    sigma.to_csv(os.path.join(output_dir, "covariance_matrix.csv"))
    
    # Also save as numpy arrays for quicker loading in Qiskit optimizations
    np.save(os.path.join(output_dir, "mu.npy"), mu.values)
    np.save(os.path.join(output_dir, "sigma.npy"), sigma.values)
    
    print(f"Computed expected returns and covariance matrix for {len(mu)} assets.")
    print("Assets list (in order):", list(mu.index))
    print(f"Saved artifacts to {output_dir}/")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "raw")
    output_dir = os.path.join(base_dir, "cached")
    
    print(f"Processing data from: {input_dir}")
    preprocess_data(input_dir, output_dir)
    print("Preprocessing complete.")
