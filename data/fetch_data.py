import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_data(tickers, start_date, end_date, output_dir):
    """
    Downloads OHLCV data for given tickers from Yahoo Finance and saves them to CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            file_path = os.path.join(output_dir, f"{ticker.replace('.NS', '')}.csv")
            data.to_csv(file_path)
            print(f"Saved {ticker} to {file_path}")
        else:
            print(f"Warning: No data found for {ticker}")

if __name__ == "__main__":
    # Top 6 Indian Banking Stocks (Bank Nifty constituents)
    tickers = [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "AXISBANK.NS",
        "KOTAKBANK.NS",
        "INDUSINDBK.NS"
    ]
    
    # 2 years of historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    print(f"Saving data to: {output_dir}")
    
    fetch_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), output_dir)
    print("Data fetching complete.")
