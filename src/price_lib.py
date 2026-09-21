"""
Maize price models: monthly national maize price vs rainfall, fuel price and month of year.

Usage (from the repo root):
    python src/price_lib.py

Input:
    data/processed/maize_price_monthly.csv   from: python src/build_price_data.py

Outputs:
    outputs/price_results.csv        model comparison table
    outputs/price_over_time.png      actual price and the best model's fitted values

Models (price in KES per kg):
    1. price ~ rainfall in the last growing season
    2. price ~ rainfall + fuel price
    3. price ~ rainfall + fuel price + month-of-year effects

Prices in the same season move together, so standard errors and p-values are
clustered by growing season (about 14 independent groups). That is why the
p-values here are less impressive than an ordinary regression would give.
Prices are in current (nominal) shillings, so fuel partly picks up general inflation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRICE_FILE = "data/processed/maize_price_monthly.csv"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_price(path=PRICE_FILE):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = np.where(df["month"] >= 9, df["year"], df["year"] - 1)
    return df.sort_values("date").reset_index(drop=True)


def ols_cluster(X, y, groups):
    """OLS with cluster-robust standard errors (CR1) and t(G-1) p-values."""
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((k, k))
    ids = np.unique(groups)
    for g in ids:
        s = X[groups == g].T @ u[groups == g]
        meat += np.outer(s, s)
    G = len(ids)
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(V))
    p = 2 * stats.t.sf(np.abs(beta / se), G - 1)
    ssr = float(u @ u)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ssr / tss
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
    return {
        "beta": beta, "se": se, "p": p, "n": n, "k": k, "clusters": G,
        "r2": r2, "adj": adj, "aic": -2 * llf + 2 * k,
        "resid_sd": float(np.sqrt(ssr / (n - k))),
        "durbin_watson": float(np.sum(np.diff(u) ** 2) / ssr),
        "fitted": X @ beta,
    }


def _design(df, use_fuel, use_month):
    cols, names = [np.ones(len(df)), df["season_rain_mm"].to_numpy()], ["Intercept", "rain"]
    if use_fuel:
        cols.append(df["fuel_price_kes_per_litre"].to_numpy())
        names.append("fuel")
    if use_month:
        for m in range(2, 13):
            cols.append((df["month"] == m).astype(float).to_numpy())
            names.append(f"month_{m}")
    return np.column_stack(cols), names


def fit_price_models(df):
    """Fit the price models on one common sample. Returns a list of dicts."""
    has_fuel = df["fuel_price_kes_per_litre"].notna().sum() > 24
    d = df.dropna(subset=["maize_price_kes_per_kg", "season_rain_mm"] +
                  (["fuel_price_kes_per_litre"] if has_fuel else [])).reset_index(drop=True)
    specs = [("1. Rainfall only", False, False)]
    if has_fuel:
        specs += [("2. Rainfall + fuel price", True, False), ("3. Rainfall + fuel + month", True, True)]
    else:
        specs += [("2. Rainfall + month", False, True)]

    results = []
    for name, use_fuel, use_month in specs:
        X, names = _design(d, use_fuel, use_month)
        res = ols_cluster(X, d["maize_price_kes_per_kg"], d["season"])
        res.update({"name": name, "names": names, "data": d})
        res["rain_coef"], res["rain_p"] = res["beta"][1], res["p"][1]
        if use_fuel:
            res["fuel_coef"], res["fuel_p"] = res["beta"][names.index("fuel")], res["p"][names.index("fuel")]
        else:
            res["fuel_coef"] = res["fuel_p"] = np.nan
        results.append(res)
    return results


def predict_price(res, rain, fuel, month):
    """Predict the maize price (KES/kg) from a fitted model dict."""
    beta, names = res["beta"], res["names"]
    value = beta[0] + beta[1] * rain
    if "fuel" in names:
        value += beta[names.index("fuel")] * fuel
    if month >= 2 and f"month_{month}" in names:
        value += beta[names.index(f"month_{month}")]
    return value


def results_table(results):
    def fmt_p(p):
        return "" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
    rows = []
    for r in results:
        rows.append({
            "Model": r["name"],
            "n (months)": r["n"],
            "Rain effect (KES/kg per +100 mm)": r["rain_coef"] * 100,
            "p (rain)": fmt_p(r["rain_p"]),
            "Fuel effect (KES/kg per +10 KES/L)": r["fuel_coef"] * 10 if not np.isnan(r["fuel_coef"]) else np.nan,
            "p (fuel)": fmt_p(r["fuel_p"]),
            "Adjusted R²": r["adj"],
            "AIC": r["aic"],
            "Durbin-Watson": r["durbin_watson"],
        })
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_price()
    results = fit_price_models(df)
    table = results_table(results)
    print(f"Months: {len(df)} ({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}), "
          f"growing seasons: {df['season'].nunique()}\n")
    print(table.round(3).to_string(index=False))
    print("\nDurbin-Watson well below 2 means neighbouring months are strongly related "
          "(hence clustered standard errors).")

    Path("outputs").mkdir(exist_ok=True)
    table.to_csv("outputs/price_results.csv", index=False)

    best = results[-1]
    d = best["data"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["date"], d["maize_price_kes_per_kg"], label="Actual price", color="#2e7d32")
    ax.plot(d["date"], best["fitted"], label=f"Fitted ({best['name']})", color="red", alpha=0.7)
    ax.set_ylabel("Maize price (KES per kg)")
    ax.set_title("Monthly maize price: actual vs model")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/price_over_time.png", dpi=150)
    print("\nSaved outputs/price_results.csv and outputs/price_over_time.png")


if __name__ == "__main__":
    main()
