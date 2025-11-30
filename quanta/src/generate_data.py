#!/usr/bin/env python3

import os
import sys

# Ensure project root is on Python path so imports from src/ work
# If this script lives in "src/", project root is its parent directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_prep import prepare_master_dataframe


def main():
    """
    Command-line entry point for building and saving the master feature DataFrame.

    Prompts the user for a date range and output path, then constructs a
    multi-ETF, macro-augmented, feature-rich DataFrame and writes it to CSV.

    Returns:
        None
    """
    # Interactive prompts for dates and output path
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    out_path = input("Enter output file path [data/master_df.csv]: ").strip()
    if not out_path:
        out_path = "data/master_df.csv"

    # Mapping of FRED series codes to column names
    fred_map = {
        "VIXCLS": "VIX",      # CBOE Volatility Index
        "DGS10":  "yield10Y"  # 10-year Treasury yield
    }

    df = prepare_master_dataframe(
        etfs=[
            # consumer discretionary:
            "XLY", "IYC", "VCR", "XHB",

            # consumer staples:
            "XLP", "IYK", "VDC",

            # energy:
            "XLE", "IYE", "VDE", "XOP", "AMLP", "OIH",

            # financials:
            "XLF", "IYF", "VFH", "KBE", "KRE",

            # healthcare:
            "XLV", "IYH", "IBB", "XBI", "VHT",

            # industrials:
            "XLI", "IYJ", "VIS",

            # materials:
            "XLB", "IYM", "VAW", "GDX", "GDXJ",

            # information technology:
            "XLK", "IYW", "VGT", "FDN", "IGV",

            # communication services:
            "IYZ", "VOX", "XLC",

            # utilities:
            "XLU", "IDU", "VPU",

            # real estate:
            "RWR", "XLRE", "VNQ",
        ],
        start_date=start_date,
        end_date=end_date,
        fred_map=fred_map
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save to CSV
    df.to_csv(out_path)
    print(f"Wrote master DataFrame to {out_path}")


if __name__ == "__main__":
    main()
