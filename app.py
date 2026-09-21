"""
Maize Yield and Rainfall in Kenya - interactive web app.

Start it with either:
    python app.py
    streamlit run app.py

Then open the address shown in the terminal (in Codespaces, click "Open in Browser").

The app reads the project files (run the merge steps first):
    data/raw/yield_data_county.csv
    data/raw/yield_data_national.csv
    data/processed/maize_data_county.csv
    data/processed/maize_data_national.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Lets "python app.py" start the website (Streamlit itself runs this file with "streamlit run").
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists
    except ImportError:
        sys.exit("Streamlit is not installed. Run: pip install streamlit")
    if not exists():
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")])
        sys.exit()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(ROOT / "src"))
from export_excel import fit_rainfall  # regression helper used by the Excel export
from price_lib import MONTH_NAMES, fit_price_models, load_price, predict_price, results_table

BAGS_PER_KG_ACRE = 2.471 / 90  # kg per acre -> 90 kg bags per hectare

st.set_page_config(page_title="Maize Yield & Rainfall in Kenya", page_icon="🌽", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_data():
    files = {
        "raw_county": ROOT / "data/raw/yield_data_county.csv",
        "raw_national": ROOT / "data/raw/yield_data_national.csv",
        "merged_county": ROOT / "data/processed/maize_data_county.csv",
        "merged_national": ROOT / "data/processed/maize_data_national.csv",
    }
    missing = [str(p.relative_to(ROOT)) for p in files.values() if not p.exists()]
    if missing:
        return None, missing
    return {k: pd.read_csv(p) for k, p in files.items()}, []


data, missing = load_data()
if data is None:
    st.error("Some project files are missing, so the app cannot start.")
    st.write("Missing files:")
    for m in missing:
        st.code(m)
    st.write("Run the download and merge steps first (see the README, 'How to run').")
    st.stop()

raw_c, raw_n = data["raw_county"], data["raw_national"]
m_c, m_n = data["merged_county"], data["merged_national"]


@st.cache_data
def run_models(merged_national, merged_county):
    return (fit_rainfall(merged_national),
            fit_rainfall(merged_county),
            fit_rainfall(merged_county, county_effects=True))


m1, m2, m3 = run_models(m_n, m_c)


@st.cache_data
def county_ranking(county_df, national_df):
    years = sorted(set(county_df["year"]) & set(national_df["year"]))
    c = county_df[county_df["year"].isin(years)]
    nat_mean = national_df[national_df["year"].isin(years)]["yield_bags_per_ha"].mean()
    r = (c.groupby("county")["yield_bags_per_ha"]
         .agg(avg="mean", low="min", high="max")
         .sort_values("avg", ascending=False)
         .reset_index())
    r.insert(0, "Rank", range(1, len(r) + 1))
    r["vs"] = (r["avg"] / nat_mean - 1) * 100
    r.columns = ["Rank", "County", "Average yield (bags/ha)", "Lowest year", "Highest year", "vs national (%)"]
    return r, nat_mean, years


ranking, nat_mean, years = county_ranking(raw_c, raw_n)


def fit_county_model(df):
    """Rainfall + county effects model, used for the predictor tab."""
    d = df.dropna(subset=["yield_kg_per_acre", "rainfall_mm"])
    counties = sorted(d["county"].unique())
    cols = [np.ones(len(d)), d["rainfall_mm"].to_numpy()]
    cols += [(d["county"] == c).astype(float).to_numpy() for c in counties[1:]]
    X, y = np.column_stack(cols), d["yield_kg_per_acre"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sd = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
    return beta, counties, sd


def predict(beta, counties, county, rain):
    value = beta[0] + beta[1] * rain
    if county in counties[1:]:
        value = value + beta[2 + counties[1:].index(county)]
    return value


beta, counties, resid_sd = fit_county_model(m_c)
sig = lambda p: "significant" if p < 0.05 else "not significant"
fmt_p = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"

# ------------------------------------------------------------------ header
st.title("🌽 Maize Yield and Rainfall in Kenya")
st.caption("Does rainfall explain maize yield? A linear regression study at national and county level.")

tab_over, tab_county, tab_models, tab_predict, tab_price, tab_data = st.tabs(
    ["Overview", "County comparison", "Regression models", "Predict yield", "Maize price", "Data and limits"])

# ------------------------------------------------------------------ overview
with tab_over:
    best = ranking.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("National average yield", f"{nat_mean:.1f} bags/ha", help=f"{years[0]}-{years[-1]}")
    c2.metric("Highest-yield county", best["County"], f"{best['Average yield (bags/ha)']:.1f} bags/ha")
    c3.metric("National model fit (adj. R²)", f"{m1['adj']:.2f}")
    c4.metric("Rainfall effect within a county", f"+{m3['coef'] * 100:.0f} kg/acre",
              help="Per extra 100 mm of seasonal rainfall (Model 3)")

    st.subheader("Key findings")
    st.markdown(
        f"- **Nationally, rainfall explains little.** Adjusted R² is {m1['adj']:.2f} and the rainfall effect is "
        f"{sig(m1['p'])} (p = {fmt_p(m1['p'])}), using {int(m1['n'])} years of data.\n"
        f"- **Across counties the link looks strong, but it is misleading.** The simple county model gives "
        f"+{m2['coef'] * 100:.0f} kg/acre per 100 mm (adjusted R² = {m2['adj']:.2f}), but it mixes rainfall with "
        f"permanent differences between counties such as soil, altitude and farming system.\n"
        f"- **Controlling for county gives a smaller effect.** Within the same county, 100 mm more rainfall goes with "
        f"about +{m3['coef'] * 100:.0f} kg/acre ({sig(m3['p'])}, p = {fmt_p(m3['p'])}). This model fits best "
        f"(adjusted R² = {m3['adj']:.2f}).\n"
        f"- **{best['County']} has the highest average yield** of the counties studied "
        f"({best['Average yield (bags/ha)']:.1f} bags/ha against {nat_mean:.1f} nationally)."
    )
    st.info("These results come from a small sample (a few counties over a few years). "
            "Read them as evidence of an association, not proof of cause.")

# ------------------------------------------------------------------ county comparison
with tab_county:
    st.subheader("Maize yield by county")
    st.caption(f"Average yield in 90 kg bags per hectare, {years[0]}-{years[-1]}. "
               f"National average for the same years: {nat_mean:.1f}.")

    shown = ranking.copy()
    shown["vs national (%)"] = shown["vs national (%)"].map(lambda v: f"{v:+.0f}%")
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "Average yield (bags/ha)": st.column_config.ProgressColumn(
                "Average yield (bags/ha)", format="%.1f", min_value=0,
                max_value=float(ranking["Average yield (bags/ha)"].max() * 1.1)),
            "Lowest year": st.column_config.NumberColumn(format="%.1f"),
            "Highest year": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ranking["County"], ranking["Average yield (bags/ha)"], color="#2e7d32")
        ax.axhline(nat_mean, color="red", linestyle="--", label=f"National average ({nat_mean:.1f})")
        ax.set_ylabel("Average yield (bags/ha)")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        over_time = (raw_c[raw_c["year"].isin(years)]
                     .pivot(index="year", columns="county", values="yield_bags_per_ha"))
        national_series = raw_n[raw_n["year"].isin(years)].set_index("year")["yield_bags_per_ha"]
        over_time["Kenya (national)"] = national_series
        st.line_chart(over_time)

    st.caption("These are counties chosen for the study, not all of Kenya, so beating the national average is "
               "expected. Counties with close averages should be treated as roughly tied.")

# ------------------------------------------------------------------ models
with tab_models:
    st.subheader("Regression models")
    st.caption("Outcome: maize yield (kg per acre). Predictor: seasonal rainfall, March to August (mm).")

    ny = int(m_n["year"].nunique())
    nc = int(m_c["county"].nunique())
    yr = f"{int(m_c['year'].min())}-{int(m_c['year'].max())}"
    rows = [
        ("1. Rainfall only", f"National, {ny} years", m1),
        ("2. Rainfall only", f"{nc} counties, {yr}", m2),
        ("3. Rainfall + county effects", f"{nc} counties, {yr}", m3),
    ]
    table = pd.DataFrame({
        "Model": [r[0] for r in rows],
        "Data": [r[1] for r in rows],
        "n": [int(r[2]["n"]) for r in rows],
        "Rainfall effect (kg/acre per mm)": [r[2]["coef"] for r in rows],
        "p-value": [fmt_p(r[2]["p"]) for r in rows],
        "Effect of +100 mm (kg/acre)": [r[2]["coef"] * 100 for r in rows],
        "Adjusted R²": [r[2]["adj"] for r in rows],
        "AIC": [r[2]["aic"] for r in rows],
        "Significant at 5%?": ["Yes" if r[2]["p"] < 0.05 else "No" for r in rows],
    })
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Rainfall effect (kg/acre per mm)": st.column_config.NumberColumn(format="%.4f"),
            "Effect of +100 mm (kg/acre)": st.column_config.NumberColumn(format="%.1f"),
            "Adjusted R²": st.column_config.NumberColumn(format="%.3f"),
            "AIC": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption("AIC (lower is better) can only be compared between Models 2 and 3, which use the same data.")

    st.markdown(
        "**How to read this.** Model 1 (national) has few data points and a weak fit. Model 2 mixes counties, "
        "so part of its rainfall effect is really differences between counties. Model 3 gives each county its own "
        "baseline, so the rainfall effect reflects year-to-year changes within a county."
    )

    st.subheader("Yield vs rainfall by county (Model 3)")
    d = m_c.dropna(subset=["yield_kg_per_acre", "rainfall_mm"])
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for i, c in enumerate(counties):
        g = d[d["county"] == c]
        ax2.scatter(g["rainfall_mm"], g["yield_kg_per_acre"], color=f"C{i}", label=c)
        xs = np.linspace(g["rainfall_mm"].min(), g["rainfall_mm"].max(), 20)
        ax2.plot(xs, predict(beta, counties, c, xs), color=f"C{i}", alpha=0.7)
    ax2.set_xlabel("Seasonal rainfall (mm)")
    ax2.set_ylabel("Yield (kg per acre)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.caption("Each county has its own line, but all lines share the same slope.")

# ------------------------------------------------------------------ predictor
with tab_predict:
    st.subheader("Predict maize yield")
    st.write("Pick a county and a seasonal rainfall total. The prediction uses Model 3 "
             "(rainfall + county effects). The national model is not used because it fits poorly.")

    county_choice = st.selectbox("County", counties)
    g = m_c[m_c["county"] == county_choice]
    lo, hi = int(g["rainfall_mm"].min()), int(g["rainfall_mm"].max()) + 1
    rain = st.slider("Seasonal rainfall, March to August (mm)", lo, hi, int(g["rainfall_mm"].mean()),
                     help="The slider only covers rainfall values seen in this county's data.")

    pred = float(predict(beta, counties, county_choice, rain))
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted yield", f"{pred:,.0f} kg/acre")
    p2.metric("Same in bags per hectare", f"{pred * BAGS_PER_KG_ACRE:.1f}")
    p3.metric("Typical model error", f"± {resid_sd:,.0f} kg/acre")
    st.caption("The typical error is the model's average miss on the data it was fitted to. "
               "It is a rough guide only, because the sample is small.")
    st.warning("This is a teaching model built on a few counties and years. "
               "Do not use it for real farming or financial decisions.")

# ------------------------------------------------------------------ price model
@st.cache_data
def run_price_models(df):
    results = fit_price_models(df)
    return results_table(results), results


with tab_price:
    st.subheader("Maize price model")
    price_path = ROOT / "data" / "processed" / "maize_price_monthly.csv"
    if not price_path.exists():
        st.info("The price data has not been built yet. Run these two steps, then refresh this page:")
        st.code("python src/download_prices.py\npython src/build_price_data.py")
    else:
        pdf = load_price(price_path)
        price_table, price_results = run_price_models(pdf)
        best_price = price_results[-1]
        st.caption("Monthly national maize price (KES per kg, average of markets that report often) against "
                   "rainfall in the last growing season, fuel price and month of the year.")

        latest = pdf.iloc[-1]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Latest price ({latest['date']:%b %Y})", f"{latest['maize_price_kes_per_kg']:.1f} KES/kg")
        k2.metric("Average price, all months", f"{pdf['maize_price_kes_per_kg'].mean():.1f} KES/kg")
        k3.metric("Best model fit (adj. R²)", f"{max(r['adj'] for r in price_results):.2f}")

        st.line_chart(pdf.set_index("date")[["maize_price_kes_per_kg"]])

        st.dataframe(
            price_table,
            hide_index=True,
            column_config={
                "Rain effect (KES/kg per +100 mm)": st.column_config.NumberColumn(format="%.2f"),
                "Fuel effect (KES/kg per +10 KES/L)": st.column_config.NumberColumn(format="%.2f"),
                "Adjusted R²": st.column_config.NumberColumn(format="%.3f"),
                "AIC": st.column_config.NumberColumn(format="%.1f"),
                "Durbin-Watson": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption("p-values are clustered by growing season, so they allow for months in the same season "
                   "moving together. A Durbin-Watson value far below 2 means neighbouring months are strongly related. "
                   "AIC can only be compared between models with the same n.")

        rain_coef, rain_p = best_price["rain_coef"], best_price["rain_p"]
        st.markdown(
            f"**Rainfall.** In the fullest model, 100 mm more rain in the last growing season goes with a change of "
            f"**{rain_coef * 100:+.1f} KES/kg** in the maize price, which is "
            f"{'significant' if rain_p < 0.05 else 'not significant'} at 5% (p = "
            f"{'<0.001' if rain_p < 0.001 else f'{rain_p:.3f}'}). Only about {pdf['season'].nunique()} growing seasons "
            f"are in the data, so treat this as a rough guide."
        )

        st.subheader("Predict the maize price")
        has_fuel = "fuel" in best_price["names"]
        q1, q2, q3 = st.columns(3)
        month_name = q1.selectbox("Month", MONTH_NAMES)
        rain_lo, rain_hi = float(pdf["season_rain_mm"].min()), float(pdf["season_rain_mm"].max())
        rain_in = q2.slider("Rainfall in last growing season, Mar-Aug (mm)", rain_lo, rain_hi,
                            float(pdf["season_rain_mm"].mean()))
        fuel_in = 0.0
        if has_fuel:
            f_lo, f_hi = float(pdf["fuel_price_kes_per_litre"].min()), float(pdf["fuel_price_kes_per_litre"].max())
            fuel_in = q3.slider("Fuel price (KES per litre)", f_lo, f_hi, float(pdf["fuel_price_kes_per_litre"].mean()))
        month_num = MONTH_NAMES.index(month_name) + 1
        price_pred = float(predict_price(best_price, rain_in, fuel_in, month_num))
        r1, r2 = st.columns(2)
        r1.metric("Predicted maize price", f"{price_pred:.1f} KES/kg")
        r2.metric("Typical model error", f"\u00b1 {best_price['resid_sd']:.1f} KES/kg")
        st.warning("Teaching model only. Prices are in current shillings (not adjusted for inflation), fuel partly "
                   "reflects inflation, and rainfall comes from a few satellite points. Do not use it for trading decisions.")

# ------------------------------------------------------------------ data
with tab_data:
    st.subheader("Data sources")
    st.markdown(
        "- **National yield:** Ministry of Agriculture, Economic Review of Agriculture (2010-2014) and KNBS "
        "national area and production (2019-2024). The two sources may use different methods, and 2015-2018 is missing.\n"
        "- **County yield:** KNBS National Agriculture Production Report (area and production by county), "
        "converted to bags per hectare. The latest year may be provisional.\n"
        "- **Rainfall and temperature:** NASA POWER monthly point data, one point per county (approximate main town). "
        "Rainfall is the March to August total."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Small sample, so estimates are uncertain.\n"
        "- County rows are repeated observations of the same counties, so p-values are likely too optimistic.\n"
        "- One rainfall point per county does not represent the whole county.\n"
        "- Fertilizer, seed type, pests, soil and farming practice are not in the models.\n"
        "- Association, not proof of cause."
    )

    with st.expander("County yield data"):
        st.dataframe(raw_c, hide_index=True)
    with st.expander("National yield data"):
        st.dataframe(raw_n, hide_index=True)
    with st.expander("Model-ready county data (yield and rainfall)"):
        st.dataframe(m_c, hide_index=True)

    xlsx = ROOT / "outputs" / "maize_results.xlsx"
    if xlsx.exists():
        st.download_button("Download the Excel results", data=xlsx.read_bytes(), file_name="maize_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
