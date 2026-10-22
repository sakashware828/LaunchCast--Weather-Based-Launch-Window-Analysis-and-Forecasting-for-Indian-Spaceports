"""
LaunchCast - Launch Weather Rules
Classifies each hourly weather record as GO / CAUTION / NO-GO for a launch,
based on configurable thresholds for wind, gusts, rain, and cloud cover.

IMPORTANT: These thresholds are illustrative, not official ISRO launch
commit criteria (detailed operational limits are not fully public).
Any measurable precipitation (> 0 mm in the hour) is treated as NO-GO,
which is deliberately conservative.
"""

import numpy as np
import pandas as pd

INPUT_FILE = "launchcast_weather_data_cleaned.csv"
OUTPUT_FILE = "launchcast_weather_data_classified.csv"

# ---- Configurable thresholds ----
# NO-GO  if a value is ABOVE its "nogo_above" limit.
# CAUTION if it is above "caution_above" (but not NO-GO).
# Rain has no CAUTION band: any precipitation above 0 mm is NO-GO.
THRESHOLDS = {
    "wind_speed_nogo_above": 27.0,      # km/h
    "wind_speed_caution_above": 20.0,   # km/h
    "wind_gust_nogo_above": 46.0,       # km/h
    "wind_gust_caution_above": 35.0,    # km/h
    "precip_nogo_above": 0.0,           # mm per hour
    "cloud_cover_nogo_above": 70.0,     # %
    "cloud_cover_caution_above": 50.0,  # %
}

GO_SCORES = {"GO": 1.0, "CAUTION": 0.5, "NO-GO": 0.0}


def classify_hour(row, t=THRESHOLDS) -> str:
    """Classify ONE row (easy to explain in the viva)."""
    if (row["wind_speed_10m"] > t["wind_speed_nogo_above"]
            or row["wind_gusts_10m"] > t["wind_gust_nogo_above"]
            or row["precipitation"] > t["precip_nogo_above"]
            or row["cloud_cover"] > t["cloud_cover_nogo_above"]):
        return "NO-GO"
    if (row["wind_speed_10m"] > t["wind_speed_caution_above"]
            or row["wind_gusts_10m"] > t["wind_gust_caution_above"]
            or row["cloud_cover"] > t["cloud_cover_caution_above"]):
        return "CAUTION"
    return "GO"


def classify_dataframe(df: pd.DataFrame, t=THRESHOLDS) -> pd.DataFrame:
    """Vectorized version of classify_hour (same logic, much faster)."""
    df = df.copy()
    nogo = (
        (df["wind_speed_10m"] > t["wind_speed_nogo_above"])
        | (df["wind_gusts_10m"] > t["wind_gust_nogo_above"])
        | (df["precipitation"] > t["precip_nogo_above"])
        | (df["cloud_cover"] > t["cloud_cover_nogo_above"])
    )
    caution = (
        (df["wind_speed_10m"] > t["wind_speed_caution_above"])
        | (df["wind_gusts_10m"] > t["wind_gust_caution_above"])
        | (df["cloud_cover"] > t["cloud_cover_caution_above"])
    )
    df["launch_status"] = np.select([nogo, caution], ["NO-GO", "CAUTION"], default="GO")
    df["go_score"] = df["launch_status"].map(GO_SCORES)
    return df


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = classify_dataframe(df)

    print("Launch status counts:")
    print(df["launch_status"].value_counts())
    print("\nShare of hours by status (%):")
    print((df["launch_status"].value_counts(normalize=True) * 100).round(1))

    print("\nMean go_score by month:")
    print(df.groupby("month")["go_score"].mean().sort_values(ascending=False).round(3))
    print("\nMean go_score by season:")
    print(df.groupby("season")["go_score"].mean().sort_values(ascending=False).round(3))
    print("\nMean go_score by hour of day:")
    print(df.groupby("hour")["go_score"].mean().sort_values(ascending=False).round(3))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved classified dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
