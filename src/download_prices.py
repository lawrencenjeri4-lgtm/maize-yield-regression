"""
Download Kenya's WFP food price data (includes maize and fuel prices) from the
Humanitarian Data Exchange (HDX) and print a summary of what it contains.

Usage (from the repo root, needs internet):
    python src/download_prices.py

Output:
    data/raw/wfp_food_prices_ken.csv

If the download fails, get the file by hand:
  1. Open https://data.humdata.org/dataset/wfp-food-prices-for-kenya
  2. Download the "Kenya - Food Prices" CSV (not the QuickCharts one)
  3. Save it as data/raw/wfp_food_prices_ken.csv, then run this script again
     (it will skip the download and just print the summary).
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-kenya"
OUT = Path("data/raw/wfp_food_prices_ken.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (maize-yield-regression student project)"}


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download():
    meta = json.loads(get(API))
    resources = meta["result"]["resources"]
    csvs = [r for r in resources
            if str(r.get("format", "")).lower() == "csv"
            and "quickchart" not in (str(r.get("name", "")) + str(r.get("url", ""))).lower()]
    if not csvs:
        raise RuntimeError("No suitable CSV resource found on the HDX dataset page.")
    chosen = csvs[0]
    url = chosen.get("download_url") or chosen.get("url")
    print(f"Downloading: {chosen.get('name')}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(get(url))
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


def summarise():
    df = pd.read_csv(OUT, low_memory=False)
    df = df[~df["date"].astype(str).str.startswith("#")]  # drop the HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dates: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")

    com = df["commodity"].astype(str)
    print("\nMaize-related commodities (rows):")
    print(df[com.str.contains("maize", case=False)]["commodity"].value_counts().to_string())
    print("\nFuel-related commodities (rows):")
    fuel = df[com.str.contains("fuel|diesel|petrol|gasoline", case=False)]
    print(fuel["commodity"].value_counts().to_string() if len(fuel) else "  none found")

    if "unit" in df.columns:
        maize = df[com.str.contains("maize", case=False)]
        print("\nUnits used for maize rows:")
        print(maize["unit"].value_counts().to_string())


def main():
    if OUT.exists():
        print(f"{OUT} already exists, skipping download.")
    else:
        try:
            download()
        except Exception as e:
            raise SystemExit(f"Download failed: {e}\n\n{__doc__}")
    summarise()


if __name__ == "__main__":
    main()
