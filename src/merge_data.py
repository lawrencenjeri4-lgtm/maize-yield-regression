"""
Merge maize yield data with the seasonal rainfall table.

Usage (from the repo root):
    python src/merge_data.py                                   # data/raw/yield_data.csv
    python src/merge_data.py data/raw/yield_data_county.csv data/processed/maize_data_county.csv
    python src/merge_data.py data/raw/yield_data_national.csv data/processed/maize_data_national.csv

Inputs:
    data/raw/yield_data.csv               YOU fill this in from KNBS / Ministry reports
    data/processed/rainfall_seasonal.csv  made by src/download_rainfall.py

yield_data.csv columns:
    county, year, yield_kg_per_acre            (or yield_bags_per_ha instead)
    optional: fertilizer_kg_per_acre

Use county = "Kenya" for national figures. Rainfall for "Kenya" is then the
average of your sampled counties. That is a rough proxy, so say so in your README.

Output:
    data/processed/maize_data.csv   ready for src/maize_yield_regression.py
"""
import sys

import pandas as pd

# Optional arguments: yield file, then output file
YIELD_FILE = sys.argv[1] if len(sys.argv) > 1 else "data/raw/yield_data.csv"
RAIN_FILE = "data/processed/rainfall_seasonal.csv"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/processed/maize_data.csv"

BAG_KG = 90          # one bag of maize = 90 kg
ACRES_PER_HA = 2.471


def main():
    y = pd.read_csv(YIELD_FILE)
    r = pd.read_csv(RAIN_FILE)

    # Convert 90 kg bags per hectare into kg per acre if needed
    if "yield_kg_per_acre" not in y.columns:
        if "yield_bags_per_ha" not in y.columns:
            raise SystemExit("Need yield_kg_per_acre or yield_bags_per_ha in yield_data.csv")
        y["yield_kg_per_acre"] = y["yield_bags_per_ha"] * BAG_KG / ACRES_PER_HA

    # National proxy: average rainfall and temperature over the sampled counties
    national = r.groupby("year", as_index=False)[["rainfall_mm", "temp_c"]].mean()
    national["county"] = "Kenya"
    r = pd.concat([r, national], ignore_index=True)

    merged = y.merge(r, on=["county", "year"], how="inner")

    lost = len(y) - len(merged)
    if lost:
        print(f"Warning: {lost} yield rows had no matching rainfall (county name or year mismatch).")

    cols = ["county", "year", "yield_kg_per_acre", "rainfall_mm", "temp_c"]
    if "fertilizer_kg_per_acre" in merged.columns:
        cols.append("fertilizer_kg_per_acre")
    merged[cols].round(1).to_csv(OUT_FILE, index=False)
    print(f"Saved {len(merged)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
