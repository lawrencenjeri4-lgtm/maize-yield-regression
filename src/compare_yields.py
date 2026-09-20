"""
Compare maize yield across counties and against the national figure.

Usage (from the repo root):
    python src/compare_yields.py

Inputs (both use the columns county, year, yield_bags_per_ha):
    data/raw/yield_data_county.csv
    data/raw/yield_data_national.csv

Outputs (in outputs/):
    county_yield_ranking.csv
    county_yield_ranking.png      average yield per county, with the national line
    county_yield_over_time.png    yield per year for each county and the national figure

Only years present in BOTH files are used for the national comparison.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COUNTY_FILE = "data/raw/yield_data_county.csv"
NATIONAL_FILE = "data/raw/yield_data_national.csv"
COL = "yield_bags_per_ha"


def main():
    county = pd.read_csv(COUNTY_FILE)
    national = pd.read_csv(NATIONAL_FILE)

    years = sorted(set(county["year"]) & set(national["year"]))
    if not years:
        raise SystemExit("No overlapping years between the county and national files.")
    print(f"Years compared: {years[0]} to {years[-1]} ({len(years)} years)")

    county = county[county["year"].isin(years)]
    nat = national[national["year"].isin(years)].set_index("year")[COL].sort_index()
    nat_mean = nat.mean()

    ranking = (
        county.groupby("county")[COL]
        .agg(mean_yield="mean", lowest="min", highest="max", std_dev="std")
        .sort_values("mean_yield", ascending=False)
    )
    ranking["vs_national_pct"] = (ranking["mean_yield"] / nat_mean - 1) * 100
    ranking = ranking.round(1)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    print(f"\nNational average yield: {nat_mean:.1f} bags/ha\n")
    print("Average maize yield by county (90 kg bags per hectare):")
    print(ranking.to_string())

    Path("outputs").mkdir(exist_ok=True)
    ranking.to_csv("outputs/county_yield_ranking.csv")

    # Chart 1: average yield per county
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ranking.index, ranking["mean_yield"], color="#2e7d32")
    ax.axhline(nat_mean, color="red", linestyle="--", label=f"National average ({nat_mean:.1f})")
    ax.set_ylabel("Average yield (90 kg bags per hectare)")
    ax.set_title(f"Maize yield by county, {years[0]} to {years[-1]}")
    ax.legend()
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig("outputs/county_yield_ranking.png", dpi=150)

    # Chart 2: yield over time
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name, g in county.groupby("county"):
        g = g.sort_values("year")
        ax2.plot(g["year"], g[COL], marker="o", label=name)
    ax2.plot(nat.index, nat.values, color="black", linewidth=3, linestyle="--", label="Kenya (national)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Yield (90 kg bags per hectare)")
    ax2.set_title("Maize yield over time")
    ax2.set_xticks(years)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("outputs/county_yield_over_time.png", dpi=150)

    print("\nSaved: outputs/county_yield_ranking.csv, county_yield_ranking.png, county_yield_over_time.png")


if __name__ == "__main__":
    main()
