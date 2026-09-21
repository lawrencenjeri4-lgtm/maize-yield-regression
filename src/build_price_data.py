"""
Build a monthly national dataset for the maize price model.

Usage (from the repo root):
    python src/build_price_data.py
    python src/build_price_data.py --commodity "Maize (white)"     # choose the maize commodity yourself

Inputs:
    data/raw/wfp_food_prices_ken.csv    from: python src/download_prices.py
    data/raw/nasa_power_monthly.csv     from: python src/download_rainfall.py

Output:
    data/processed/maize_price_monthly.csv   one row per month, with columns:
        date, year, month, maize_price_kes_per_kg, fuel_price_kes_per_litre,
        season_rain_mm, n_markets

How the columns are made:
- maize price: average retail price across markets that report often enough (KES per kg).
- fuel price: average diesel price across markets (or other fuel if no diesel), KES per litre.
- season_rain_mm: March-August rainfall of the most recent completed growing season
  (average of the sampled counties). A price in September to December of year Y uses the
  season of year Y, and a price in January to August of year Y+1 also uses the season of year Y.
"""
import sys

import numpy as np
import pandas as pd

PRICES = "data/raw/wfp_food_prices_ken.csv"
RAIN = "data/raw/nasa_power_monthly.csv"
OUT = "data/processed/maize_price_monthly.csv"

START, END = "2010-01", "2024-12"   # NASA POWER file covers 2010-2024
MIN_MONTHS = 60                     # markets reporting fewer months than this are dropped
SEASON_MONTHS = [3, 4, 5, 6, 7, 8]  # must match download_rainfall.py


def load_prices():
    df = pd.read_csv(PRICES, low_memory=False)
    df = df[~df["date"].astype(str).str.startswith("#")].copy()  # drop the HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df["ym"] = df["date"].dt.to_period("M")
    return df[(df["ym"] >= START) & (df["ym"] <= END)]


def pick_maize(df, forced=None):
    com = df["commodity"].astype(str)
    if forced:
        rows = df[com.str.lower().str.strip() == forced.lower().strip()]
        if rows.empty:
            raise SystemExit(f"No rows for commodity '{forced}'. Run download_prices.py to see the names.")
        return rows, forced
    cand = df[com.str.contains("maize", case=False) & ~com.str.contains("flour|meal|bran|green|seed", case=False)]
    if cand.empty:
        raise SystemExit("No maize commodity found. Run python src/download_prices.py to see the names.")
    counts = cand["commodity"].value_counts()
    name = counts.index[0]
    print(f"Maize commodity used: '{name}' ({counts.iloc[0]:,} rows). Others: {list(counts.index[1:])}")
    return cand[cand["commodity"] == name], name


def clean_maize(rows):
    if "currency" in rows.columns and (rows["currency"] == "KES").any():
        rows = rows[rows["currency"] == "KES"]
    if "pricetype" in rows.columns and (rows["pricetype"].astype(str).str.lower() == "retail").any():
        rows = rows[rows["pricetype"].astype(str).str.lower() == "retail"]
    unit = rows["unit"].astype(str).str.strip().value_counts().index[0]
    rows = rows[rows["unit"].astype(str).str.strip() == unit]
    print(f"Price unit: '{unit}'" + ("" if unit.upper() == "KG" else "  <-- not KG! Labels in the app assume KES per kg."))
    return rows


def monthly_price(rows):
    per_market = rows.groupby(["market", "ym"])["price"].mean().reset_index()
    months_per_market = per_market.groupby("market")["ym"].nunique().sort_values(ascending=False)
    keep = months_per_market[months_per_market >= MIN_MONTHS].index
    if len(keep) < 3:
        keep = months_per_market.index[:5]
        print(f"Few markets reach {MIN_MONTHS} months, using the {len(keep)} best-covered markets instead.")
    print(f"Markets used ({len(keep)}): {', '.join(map(str, keep))}")
    kept = per_market[per_market["market"].isin(keep)]
    out = kept.groupby("ym").agg(maize_price_kes_per_kg=("price", "mean"), n_markets=("market", "nunique"))
    return out


def monthly_fuel(df):
    com = df["commodity"].astype(str)
    fuel = df[com.str.contains("diesel", case=False)]
    label = "diesel"
    if fuel.empty:
        fuel = df[com.str.contains("fuel|petrol|gasoline", case=False)]
        label = "other fuel"
    if fuel.empty:
        print("No fuel prices found in the file. The fuel column will be empty.")
        return pd.Series(dtype=float, name="fuel_price_kes_per_litre")
    if "currency" in fuel.columns and (fuel["currency"] == "KES").any():
        fuel = fuel[fuel["currency"] == "KES"]
    print(f"Fuel series: {label} ({sorted(fuel['commodity'].unique())}), {fuel['ym'].nunique()} months")
    return fuel.groupby("ym")["price"].mean().rename("fuel_price_kes_per_litre")


def season_rainfall():
    r = pd.read_csv(RAIN)
    r = r[r["month"].isin(SEASON_MONTHS)]
    per_year = r.groupby(["county", "year"])["rainfall_mm"].sum().reset_index()
    counts = r.groupby(["county", "year"])["month"].nunique().reset_index(name="m")
    per_year = per_year.merge(counts, on=["county", "year"])
    per_year = per_year[per_year["m"] == len(SEASON_MONTHS)]
    return per_year.groupby("year")["rainfall_mm"].mean()  # average of the sampled counties


def main():
    forced = sys.argv[sys.argv.index("--commodity") + 1] if "--commodity" in sys.argv else None
    df = load_prices()
    maize_rows, _ = pick_maize(df, forced)
    price = monthly_price(clean_maize(maize_rows))
    fuel = monthly_fuel(df)
    rain_by_year = season_rainfall()

    out = price.join(fuel, how="left").reset_index()
    out["date"] = out["ym"].dt.to_timestamp()
    out["year"], out["month"] = out["date"].dt.year, out["date"].dt.month
    season_year = np.where(out["month"] >= 9, out["year"], out["year"] - 1)
    out["season_rain_mm"] = pd.Series(season_year).map(rain_by_year).to_numpy()
    out = out.dropna(subset=["maize_price_kes_per_kg", "season_rain_mm"])

    cols = ["date", "year", "month", "maize_price_kes_per_kg", "fuel_price_kes_per_litre",
            "season_rain_mm", "n_markets"]
    if "fuel_price_kes_per_litre" not in out.columns:
        out["fuel_price_kes_per_litre"] = np.nan
    out = out[cols].sort_values("date")
    num_cols = [c for c in cols if c != "date"]
    out[num_cols] = out[num_cols].round(2)
    out["date"] = out["date"].dt.strftime("%Y-%m")
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out)} monthly rows to {OUT} ({out['date'].iloc[0]} to {out['date'].iloc[-1]})")
    print(f"Months with a fuel price: {int(out['fuel_price_kes_per_litre'].notna().sum())}")


if __name__ == "__main__":
    main()
