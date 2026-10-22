"""
LaunchCast - Step 3: Hypothesis tests on wind speed
1) Welch t-test: NE Monsoon vs rest of year
2) Same test on DAILY means (hourly data is autocorrelated, which inflates
   significance when n is ~100k, so the daily test is the more defensible one)
3) Effect size (Cohen's d)
4) One-way ANOVA + Kruskal-Wallis across all four seasons
"""

import numpy as np
import pandas as pd
from scipy import stats

INPUT_FILE = "launchcast_weather_data_classified.csv"
ALPHA = 0.05


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def effect_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def welch(a: pd.Series, b: pd.Series, label: str):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)
    print(f"\n--- Welch t-test ({label}) ---")
    print(f"NE Monsoon: n={len(a)}, mean={a.mean():.2f} km/h | Rest: n={len(b)}, mean={b.mean():.2f} km/h")
    print(f"t = {t:.3f}, p {fmt_p(p)}, Cohen's d = {d:.3f} ({effect_label(d)} effect)")
    print("=> REJECT H0: significant difference." if p < ALPHA
          else "=> FAIL TO REJECT H0: no significant difference.")


def main():
    df = pd.read_csv(INPUT_FILE)

    # H0: mean wind speed is the same in NE Monsoon and the rest of the year.
    ne = df.loc[df["season"] == "NE_Monsoon", "wind_speed_10m"]
    rest = df.loc[df["season"] != "NE_Monsoon", "wind_speed_10m"]
    welch(ne, rest, "hourly data")

    daily = df.groupby("date").agg(wind=("wind_speed_10m", "mean"), season=("season", "first"))
    welch(daily.loc[daily["season"] == "NE_Monsoon", "wind"],
          daily.loc[daily["season"] != "NE_Monsoon", "wind"], "daily means")

    print("\n--- Wind speed by season (hourly) ---")
    print(df.groupby("season")["wind_speed_10m"].agg(["mean", "std", "count"]).round(2))

    groups = [g["wind"].to_numpy() for _, g in daily.groupby("season")]
    f, p_anova = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    print("\n--- Across all 4 seasons (daily means) ---")
    print(f"One-way ANOVA: F = {f:.2f}, p {fmt_p(p_anova)}")
    print(f"Kruskal-Wallis (non-parametric check): H = {h:.2f}, p {fmt_p(p_kw)}")
    print("\nInterpretation tip: significance shows the difference is unlikely to be chance; "
          "Cohen's d shows whether it is large enough to matter operationally.")


if __name__ == "__main__":
    main()
