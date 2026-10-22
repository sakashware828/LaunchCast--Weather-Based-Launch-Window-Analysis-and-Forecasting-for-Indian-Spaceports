"""
LaunchCast - Data Cleaning & Feature Engineering
Loads the raw weather CSV, validates it, and adds time-based features
needed by the rest of the project (launch rules, forecasting, visualizations).

Time handling: Open-Meteo data are requested in Asia/Kolkata, so timestamps
are explicitly stored as timezone-aware IST values.
"""

import numpy as np
import pandas as pd

INPUT_FILE = "launchcast_weather_data.csv"
OUTPUT_FILE = "launchcast_weather_data_cleaned.csv"
TIMEZONE = "Asia/Kolkata"

NUMERIC_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "wind_gusts_10m", "cloud_cover",
]


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw CSV, fix timezone, drop impossible/missing/duplicate rows."""
    df = pd.read_csv(filepath)

    if "time" not in df.columns:
        raise ValueError("Missing required 'time' column.")
    print(f"Raw rows: {len(df)}")
    print(f"Missing values per column (before cleaning):\n{df.isnull().sum()}\n")

    # Open-Meteo was requested in Asia/Kolkata. Make the timezone explicit.
    parsed_time = pd.to_datetime(df["time"], errors="raise")
    if parsed_time.dt.tz is None:
        df["time"] = parsed_time.dt.tz_localize(TIMEZONE)
    else:
        df["time"] = parsed_time.dt.tz_convert(TIMEZONE)

    # Physically impossible values (NaN comparisons are False, so NaNs are
    # handled separately by dropna below).
    invalid_mask = (
        (df["relative_humidity_2m"] < 0) | (df["relative_humidity_2m"] > 100)
        | (df["wind_speed_10m"] < 0)
        | (df["wind_gusts_10m"] < 0)
        | (df["cloud_cover"] < 0) | (df["cloud_cover"] > 100)
        | (df["precipitation"] < 0)
    )
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        print(f"Found {n_invalid} rows with impossible values - dropping them.")
        df = df.loc[~invalid_mask].copy()

    before = len(df)
    df = df.dropna().copy()
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing values.")

    # Duplicates: include site so the same timestamp at two sites is kept.
    keys = ["site", "time"] if "site" in df.columns else ["time"]
    duplicate_mask = df.duplicated(subset=keys, keep="first")
    n_duplicates = int(duplicate_mask.sum())
    if n_duplicates > 0:
        print(f"Found {n_duplicates} duplicate observations - keeping the first copy.")
        df = df.loc[~duplicate_mask].copy()

    return df.sort_values(keys).reset_index(drop=True)


def report_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report (but do NOT remove) statistical outliers using the IQR rule.
    Extreme wind/rain are real weather events that matter for launch
    decisions, so they are kept. Only physically impossible values are dropped.
    """
    rows = []
    for col in ["wind_speed_10m", "wind_gusts_10m", "precipitation", "temperature_2m"]:
        values = df[col]
        if col == "precipitation":
            # Rain is zero in most hours, so the IQR collapses to 0. Judge
            # intensity only among hours when it actually rained.
            values = values[values > 0]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((values < lo) | (values > hi)).sum())
        rows.append({"column": col, "lower_fence": round(lo, 2),
                     "upper_fence": round(hi, 2), "n_outliers": n_out,
                     "pct": round(100 * n_out / len(values), 2)})
    out = pd.DataFrame(rows)
    print("IQR outlier report (kept in data - real weather extremes):")
    print(out.to_string(index=False))
    return out


def get_season(month: int) -> str:
    """Seasons for the Sriharikota (Andhra Pradesh) east-coast climate."""
    if month in (10, 11, 12):
        return "NE_Monsoon"
    if month in (1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    return "SW_Monsoon"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["date"] = df["time"].dt.date
    df["season"] = df["month"].map(get_season)
    return df


def main():
    print("Loading and cleaning data...")
    df = load_and_clean(INPUT_FILE)
    print(f"\nRows after cleaning: {len(df)}")

    report_outliers(df)

    print("\nAdding time-based features...")
    df = add_time_features(df)
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Season counts:\n{df['season'].value_counts()}")

    keys = ["site", "time"] if "site" in df.columns else ["time"]
    print(f"\nDuplicate observations remaining: {df.duplicated(subset=keys).sum()}")
    print(f"Timezone: {df['time'].dt.tz}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
