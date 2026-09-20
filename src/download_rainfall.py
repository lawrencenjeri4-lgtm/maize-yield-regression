"""
Download monthly rainfall and temperature from NASA POWER for chosen counties
and build a seasonal table you can merge with your maize yield data.

Usage (from the repo root):
    python src/download_rainfall.py

Needs internet. No login or API key. Uses only the standard library plus pandas.

Outputs:
    data/raw/nasa_power_monthly.csv       raw monthly values (county, year, month, rain, temp)
    data/processed/rainfall_seasonal.csv  county, year, rainfall_mm, temp_c

NOTES
- The coordinates below are APPROXIMATE points (a main town in each county).
  A single point does not represent a whole county - say so in your README,
  and change the points if you know better locations.
- NASA POWER gives rainfall in mm/day for monthly data. This script converts
  it to mm/month by multiplying by the days in that month.
- Change SEASON_MONTHS to match the maize growing season you want to test.
  The default (March to August) is a starting guess, not a fixed rule.
"""
import calendar
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

START_YEAR = 2010
END_YEAR = 2024
SEASON_MONTHS = [3, 4, 5, 6, 7, 8]  # March to August

# Approximate points: (latitude, longitude). Check and adjust these.
COUNTIES = {
    "Trans Nzoia": (1.02, 35.00),
    "Uasin Gishu": (0.52, 35.27),
    "Nakuru": (-0.30, 36.07),
    "Bungoma": (0.57, 34.56),
    "Narok": (-1.08, 35.87),
    "Nyeri": (-0.42, 36.95),
}

API = "https://power.larc.nasa.gov/api/temporal/monthly/point"
MISSING = -999


def fetch(lat, lon):
    params = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def parse(county, data):
    """Turn one API response into monthly rows. Skips the month-13 annual value."""
    p = data["properties"]["parameter"]
    rain, temp = p["PRECTOTCORR"], p["T2M"]
    rows = []
    for key, rain_mm_day in rain.items():
        year, month = int(key[:4]), int(key[4:])
        if month > 12:  # 13 = annual average
            continue
        t = temp.get(key)
        if rain_mm_day == MISSING:
            continue
        days = calendar.monthrange(year, month)[1]
        rows.append({
            "county": county,
            "year": year,
            "month": month,
            "rainfall_mm": rain_mm_day * days,
            "temp_c": None if t in (None, MISSING) else t,
        })
    return rows


def seasonal(monthly):
    """Total rainfall and mean temperature over SEASON_MONTHS for each county-year."""
    s = monthly[monthly["month"].isin(SEASON_MONTHS)]
    out = s.groupby(["county", "year"], as_index=False).agg(
        rainfall_mm=("rainfall_mm", "sum"),
        temp_c=("temp_c", "mean"),
        months_used=("month", "count"),
    )
    # Drop years where some season months were missing
    out = out[out["months_used"] == len(SEASON_MONTHS)].drop(columns="months_used")
    return out.round(1)


def main():
    all_rows = []
    for county, (lat, lon) in COUNTIES.items():
        print(f"Downloading {county} ...")
        try:
            all_rows += parse(county, fetch(lat, lon))
        except Exception as e:
            print(f"  Failed for {county}: {e}")

    if not all_rows:
        raise SystemExit("No data downloaded. Check your internet connection.")

    monthly = pd.DataFrame(all_rows)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    monthly.to_csv("data/raw/nasa_power_monthly.csv", index=False)

    seas = seasonal(monthly)
    seas.to_csv("data/processed/rainfall_seasonal.csv", index=False)
    print(f"\nSaved {len(monthly)} monthly rows and {len(seas)} seasonal rows.")
    print(seas.head())


if __name__ == "__main__":
    main()
