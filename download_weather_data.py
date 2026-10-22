"""
LaunchCast - Weather Data Downloader
Pulls hourly historical weather for Sriharikota (and optional comparison sites)
from Open-Meteo, and VERIFIES the data before saving, so you never end up
with an empty/zero-row CSV again.
"""

import requests
import pandas as pd

# ---- CONFIG: edit these ----
SITES = {
    "Sriharikota": (13.72, 80.23),
    # add comparison sites if you want, e.g.:
    # "Kulasekarapattinam": (8.35, 78.05),
}

START_DATE = "2015-01-01"
END_DATE   = "2026-09-25"   # don't use a future end date - Open-Meteo archive only has PAST data

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
]

OUTPUT_FILE = "launchcast_weather_data.csv"
# -----------------------------

all_dfs = []

for site_name, (lat, lon) in SITES.items():
    print(f"Fetching data for {site_name}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }

    resp = requests.get(url, params=params, timeout=60)

    # CHECK 1: did the request even succeed?
    if resp.status_code != 200:
        print(f"  ERROR: status code {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
        continue

    data = resp.json()

    # CHECK 2: does the response actually contain hourly data?
    if "hourly" not in data or "time" not in data["hourly"]:
        print(f"  ERROR: no 'hourly' data in response for {site_name}")
        print(f"  Full response: {data}")
        continue

    hourly = data["hourly"]
    n_rows = len(hourly["time"])

    # CHECK 3: is it actually non-empty?
    if n_rows == 0:
        print(f"  ERROR: 0 rows returned for {site_name}. Check your dates/coords.")
        continue

    df = pd.DataFrame(hourly)
    df["site"] = site_name
    all_dfs.append(df)

    print(f"  OK: {n_rows} rows fetched for {site_name}")

# CHECK 4: did we get ANYTHING across all sites?
if not all_dfs:
    raise RuntimeError(
        "No data was fetched for ANY site. Do not save/submit an empty file. "
        "Check your internet connection, coordinates, and date range above."
    )

final_df = pd.concat(all_dfs, ignore_index=True)

# CHECK 5: final sanity check before writing to disk
print("\n--- FINAL VERIFICATION ---")
print(f"Total rows: {len(final_df)}")
print(f"Columns: {list(final_df.columns)}")
print(f"Date range: {final_df['time'].min()} to {final_df['time'].max()}")
print(f"Any nulls per column:\n{final_df.isnull().sum()}")
print(final_df.head())

if len(final_df) == 0:
    raise RuntimeError("Final dataframe is empty. DO NOT save this file.")

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(final_df)} rows to {OUTPUT_FILE}")
print("Before uploading anywhere: open the CSV yourself and confirm it has real rows.")
