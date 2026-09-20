"""
Maize yield vs rainfall (and fertilizer) - linear regression starter.

Usage:
    python maize_yield_regression.py --demo          # test the pipeline with FAKE data
    python maize_yield_regression.py maize_data.csv  # run on your real data

Expected CSV columns (one row per county per year):
    county, year, yield_kg_per_acre, rainfall_mm, fertilizer_kg_per_acre

fertilizer_kg_per_acre can be left out or partly empty; the script then
skips the fertilizer models or drops the rows that lack it.

Install: pip install pandas numpy statsmodels matplotlib
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def demo_data(n_counties=6, years=range(2015, 2025)):
    """FAKE data, only to check that the script runs. Never present it as real."""
    rng = np.random.default_rng(42)
    counties = ["Trans Nzoia", "Uasin Gishu", "Nakuru", "Bungoma", "Narok", "Nyeri"][:n_counties]
    rows = []
    for c in counties:
        base = rng.uniform(600, 900)
        for y in years:
            rain = rng.normal(900, 150)
            fert = rng.uniform(20, 60)
            yld = base + 0.35 * (rain - 900) + 4.0 * fert + rng.normal(0, 60)
            rows.append([c, y, yld, rain, fert])
    return pd.DataFrame(rows, columns=[
        "county", "year", "yield_kg_per_acre", "rainfall_mm", "fertilizer_kg_per_acre"])


def summarise(name, model):
    print(f"\n=== {name} ===")
    print(model.summary().tables[1])
    print(f"R^2: {model.rsquared:.3f} | Adj. R^2: {model.rsquared_adj:.3f} | n = {int(model.nobs)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--demo":
        df = demo_data()
        print("USING FAKE DEMO DATA - results mean nothing.")
    else:
        df = pd.read_csv(sys.argv[1])

    # Basic checks
    required = {"county", "year", "yield_kg_per_acre", "rainfall_mm"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {missing}")

    print(df.describe(include="all").T)
    print("\nRows per county:\n", df["county"].value_counts())

    # Model 1: rainfall only
    m1 = smf.ols("yield_kg_per_acre ~ rainfall_mm", data=df.dropna(subset=["rainfall_mm"])).fit()
    summarise("Model 1: yield ~ rainfall", m1)

    models = {"Rainfall only": m1}

    # Model 2: rainfall + fertilizer (if available)
    if "fertilizer_kg_per_acre" in df.columns and df["fertilizer_kg_per_acre"].notna().sum() > 10:
        d2 = df.dropna(subset=["rainfall_mm", "fertilizer_kg_per_acre"])
        m2 = smf.ols("yield_kg_per_acre ~ rainfall_mm + fertilizer_kg_per_acre", data=d2).fit()
        summarise("Model 2: yield ~ rainfall + fertilizer", m2)
        models["Rainfall + fertilizer"] = m2

        # Model 3: same plus county effects (controls for soil/altitude differences)
        m3 = smf.ols(
            "yield_kg_per_acre ~ rainfall_mm + fertilizer_kg_per_acre + C(county)", data=d2
        ).fit()
        print(f"\nModel 3 (with county effects) Adj. R^2: {m3.rsquared_adj:.3f}")
        models["+ county effects"] = m3

    # Compare models: higher adjusted R^2 and lower AIC are better
    print("\n--- Model comparison ---")
    for name, m in models.items():
        print(f"{name:25s} adj R^2 = {m.rsquared_adj:.3f} | AIC = {m.aic:.1f}")

    # Plot: yield vs rainfall with the fitted line from Model 1
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["rainfall_mm"], df["yield_kg_per_acre"], alpha=0.7)
    xs = np.linspace(df["rainfall_mm"].min(), df["rainfall_mm"].max(), 100)
    ax.plot(xs, m1.params["Intercept"] + m1.params["rainfall_mm"] * xs, color="red")
    ax.set_xlabel("Seasonal rainfall (mm)")
    ax.set_ylabel("Maize yield (kg per acre)")
    ax.set_title("Maize yield vs rainfall")
    fig.tight_layout()
    fig.savefig("yield_vs_rainfall.png", dpi=150)
    print("\nSaved plot: yield_vs_rainfall.png")

    # Residual check: pattern in residuals means a straight line is not a good fit
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(m1.fittedvalues, m1.resid, alpha=0.7)
    ax2.axhline(0, color="red")
    ax2.set_xlabel("Fitted values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals (Model 1)")
    fig2.tight_layout()
    fig2.savefig("residuals.png", dpi=150)


if __name__ == "__main__":
    main()
