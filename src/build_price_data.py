"""
Build a monthly national dataset for the maize price model.

Usage (from the repo root):
    python src/build_price_data.py                          # default: all maize variants combined
    python src/build_price_data.py --list-markets           # show which markets have maize prices, then stop
    python src/build_price_data.py --min-months 36          # accept markets with fewer months of data
    python src/build_price_data.py --markets "Eldoret,Kitale,Nakuru"   # choose the markets yourself
    python src/build_price_data.py --commodity "Maize (white)"         # use one maize commodity only

Inputs:
    data/raw/wfp_food_prices_ken.csv    from: python src/download_prices.py
    data/raw/nasa_power_monthly.csv     from: python src/download_rainfall.py

Output:
    data/processed/maize_price_monthly.csv   one row per month, with columns:
        date, year, month, maize_price_kes_per_kg, fuel_price_kes_per_litre,
        season_rain_mm, n_markets

How the columns are made:
- maize price: average retail price across the chosen markets (KES per kg). By default all maize
  labels ("Maize", "Maize (white)", "Maize (white, dry)") are combined, since they are the same grain
  recorded under different names.
- fuel price: average diesel price across markets (or other fuel if no diesel), KES per litre.
- season_rain_mm: March-August rainfall of the most recent completed growing season
  (average of the sampled counties). A price in September to December of year Y uses the
  season of year Y, and a price in January to August of year Y+1 also uses the season of year Y.
"""
import argparse

import numpy as np
import pandas as pd

PRICES = "data/raw/wfp_food_prices_ken.csv"
RAIN = "data/raw/nasa_power_monthly.csv"
OUT = "data/processed/maize_price_monthly.csv"

START, END = "2010-01", "2024-12"   # NASA POWER file covers 2010-2024
SEASON_MONTHS = [3, 4, 5, 6, 7, 8]  # must match download_rainfall.py
EXCLUDE = "flour|meal|bran|green|seed"


def load_prices():
    df = pd.read_csv(PRICES, low_memory=False)
    df = df[~df["date"].astype(str).str.startswith("#")].copy()  # drop the HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df["ym"] = df["date"].dt.to_period("M")
    return df[(df["ym"] >= START) & (df["ym"] <= END)]


def pick_maize(df, commodity=None):
    com = df["commodity"].astype(str)
    if commodity:
        rows = df[com.str.lower().str.strip() == commodity.lower().strip()]
        if rows.empty:
            raise SystemExit(f"No rows for commodity '{commodity}'. Run download_prices.py to see the names.")
        print(f"Maize commodity used: '{commodity}' ({len(rows):,} rows)")
        return rows
    rows = df[com.str.contains("maize", case=False) & ~com.str.contains(EXCLUDE, case=False)]
    if rows.empty:
        raise SystemExit("No maize commodity found. Run python src/download_prices.py to see the names.")
    counts = rows["commodity"].value_counts()
    print("Maize commodities combined: " + ", ".join(f"'{k}' ({v:,} rows)" for k, v in counts.items()))
    return rows


def clean_maize(rows):
    if "currency" in rows.columns and (rows["currency"] == "KES").any():
        rows = rows[rows["currency"] == "KES"]
    if "pricetype" in rows.columns and (rows["pricetype"].astype(str).str.lower() == "retail").any():
        rows = rows[rows["pricetype"].astype(str).str.lower() == "retail"]
    unit = rows["unit"].astype(str).str.strip().value_counts().index[0]
    rows = rows[rows["unit"].astype(str).str.strip() == unit]
    print(f"Price unit: '{unit}'" + ("" if unit.upper() == "KG" else "  <-- not KG! Labels in the app assume KES per kg."))
    return rows


def market_coverage(rows):
    per_market = rows.groupby(["market", "ym"])["price"].mean().reset_index()
    cov = per_market.groupby("market")["ym"].agg(months="nunique", first="min", last="max")
    return per_market, cov.sort_values("months", ascending=False)


def monthly_price(rows, min_months, chosen):
    per_market, cov = market_coverage(rows)
    if chosen:
        wanted = [m.strip().lower() for m in chosen.split(",")]
        keep = [m for m in cov.index if str(m).lower() in wanted]
        if not keep:
            raise SystemExit("None of those markets have maize prices. Run with --list-markets to see the names.")
    else:
        keep = list(cov[cov["months"] >= min_months].index)
        if len(keep) < 3:
            keep = list(cov.index[:5])
            print(f"Few markets reach {min_months} months, using the {len(keep)} best-covered markets instead.")
    print(f"Markets used ({len(keep)}): {', '.join(map(str, keep))}")
    kept = per_market[per_market["market"].isin(keep)]
    return kept.groupby("ym").agg(maize_price_kes_per_kg=("price", "mean"), n_markets=("market", "nunique"))


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
    print(f"Fuel series: {label} ({sorted(fuel['commodity'].unique())}), {fuel['ym'].nunique()} months, "
          f"{fuel['ym'].min()} to {fuel['ym'].max()}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--commodity", help="use only this maize commodity name")
    ap.add_argument("--min-months", type=int, default=60, help="minimum months of data per market (default 60)")
    ap.add_argument("--markets", help="comma-separated market names to use")
    ap.add_argument("--list-markets", action="store_true", help="list markets with maize prices and stop")
    args = ap.parse_args()

    df = load_prices()
    rows = clean_maize(pick_maize(df, args.commodity))

    if args.list_markets:
        _, cov = market_coverage(rows)
        cov = cov.reset_index()
        cov["first"], cov["last"] = cov["first"].astype(str), cov["last"].astype(str)
        print(f"\nMarkets with maize prices ({len(cov)}):")
        print(cov.to_string(index=False))
        return

    price = monthly_price(rows, args.min_months, args.markets)
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
