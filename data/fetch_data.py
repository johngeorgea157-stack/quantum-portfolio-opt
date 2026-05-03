"""
=============================================================================
Day 2 – Fetch Stock Data (fetch_data.py)
=============================================================================

WHAT THIS SCRIPT DOES:
Downloads historical OHLCV (Open, High, Low, Close, Volume) data for Bank Nifty
stocks from Yahoo Finance and saves them as CSV files in the raw/ directory.

This is the first step in the Day 2 workflow and prepares data for preprocessing.
=============================================================================
"""

import yfinance as yf
import os
from datetime import datetime, timedelta


def fetch_data(tickers, start_date, end_date, output_dir):
    """
    Downloads OHLCV data for given tickers from Yahoo Finance and saves them to CSV.

    Parameters
    ----------
    tickers : list of str
        Stock tickers (e.g., ["HDFCBANK.NS", "ICICIBANK.NS"])
    start_date : str
        Start date in "YYYY-MM-DD" format
    end_date : str
        End date in "YYYY-MM-DD" format
    output_dir : str
        Directory to save CSV files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 70)
    print("STEP 1: FETCH HISTORICAL PRICE DATA")
    print("=" * 70)

    for ticker in tickers:
        print(f"\n📥 Downloading {ticker}...", end=" ")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                print("✗ No data returned")
                continue

            file_path = os.path.join(output_dir, f"{ticker.replace('.NS', '')}.csv")
            data.to_csv(file_path)
            print(f"✔ ({len(data)} trading days)")
            print(f"   → {file_path}")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

    print("\n" + "=" * 70)
    print("Data fetching complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────────
    # CONFIGURATION: Bank Nifty constituents (6 major Indian banking stocks)
    # ─────────────────────────────────────────────────────────────────────────────
    tickers = [
        "HDFCBANK.NS",  # HDFC Bank
        "ICICIBANK.NS",  # ICICI Bank
        "SBIN.NS",  # State Bank of India
        "AXISBANK.NS",  # Axis Bank
        "KOTAKBANK.NS",  # Kotak Mahindra Bank
        "INDUSINDBK.NS",  # IndusInd Bank
    ]

    # 2 years of historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365)

    output_dir = os.path.join(os.path.dirname(__file__), "raw")

    print("=" * 70)
    print("DAY 2 – DATA COLLECTION | fetch_data.py")
    print("=" * 70)
    print(f"✔ Selected Stocks: {', '.join(tickers)}")
    print(f"✔ Data Period: {start_date.date()} → {end_date.date()}")
    print(f"✔ Output Directory: {output_dir}")
    print("=" * 70)

    fetch_data(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        output_dir,
    )
