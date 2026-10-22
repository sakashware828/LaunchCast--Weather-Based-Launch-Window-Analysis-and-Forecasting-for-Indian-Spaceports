"""
LaunchCast - Step 1: Build daily series + stationarity test (ADF)
Reads the CLASSIFIED dataset (output of launch_rules_corrected.py).
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller

INPUT_FILE = "launchcast_weather_data_classified.csv"
OUTPUT_FILE = "daily_go_score.csv"


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Average the hourly go_score into ONE value per day."""
    daily = df.groupby("date")["go_score"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").set_index("date")


def adf_report(series: pd.Series) -> dict:
    """H0: series has a unit root (non-stationary). p < 0.05 -> stationary."""
    stat, p, _, _, crit, _ = adfuller(series)
    print("--- ADF Test Results ---")
    print(f"ADF Statistic: {stat:.4f}")
    print(f"p-value: {p:.6f}")
    print(f"Critical values: {crit}")
    if p < 0.05:
        print("=> p < 0.05: reject H0, no unit root (stationary), so d=0 is fine.")
    else:
        print("=> p >= 0.05: non-stationary, differencing (d=1) is needed.")
    print("NOTE: ADF tests for a unit root/trend only. A series can pass ADF and "
          "still have a strong ANNUAL seasonal cycle, which is why step 2 adds "
          "yearly Fourier seasonal terms.")
    return {"adf_stat": stat, "p_value": p, "critical_values": crit}


def main():
    df = pd.read_csv(INPUT_FILE)
    daily = build_daily_series(df)
    print(f"Daily series built: {len(daily)} days "
          f"({daily.index.min().date()} to {daily.index.max().date()})")
    print(daily.head(), "\n", daily.tail(), "\n")
    adf_report(daily["go_score"])
    daily.to_csv(OUTPUT_FILE)
    print(f"\nSaved daily series to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
