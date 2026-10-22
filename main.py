"""
LaunchCast - Main Pipeline
Runs the full project end-to-end:
cleaning -> classification -> stationarity -> forecasting -> hypothesis tests -> charts.

Requires launchcast_weather_data.csv (from download_weather_data.py).
Run:  python main.py
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Data Cleaning", "data_cleaning_corrected.py"),
    ("Launch Rules Classification", "launch_rules_corrected.py"),
    ("Stationarity Test (ADF)", "step1_stationarity.py"),
    ("Seasonal Forecasting (SARIMAX)", "step2_forecasting.py"),
    ("Hypothesis Tests (t-test / ANOVA)", "step3_ttest.py"),
    ("Visualizations", "visualizations/visualizations.py"),
]


def run_step(name: str, script: str):
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    start = time.time()
    # cwd=BASE_DIR so every script finds its CSVs regardless of where you run main.py
    result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n!!! {script} FAILED (exit code {result.returncode}). Stopping pipeline.")
        sys.exit(1)
    print(f"\n[{name} completed in {time.time() - start:.1f}s]")


def main():
    if not (BASE_DIR / "launchcast_weather_data.csv").exists():
        sys.exit("launchcast_weather_data.csv not found. Run download_weather_data.py first.")

    print("LaunchCast Full Pipeline (forecast fitting may take a minute or two)")
    t0 = time.time()
    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {time.time() - t0:.1f}s")
    print("=" * 70)
    print("""
Output files:
  launchcast_weather_data_cleaned.csv     cleaned data
  launchcast_weather_data_classified.csv  GO / CAUTION / NO-GO labels
  daily_go_score.csv                      daily series
  forecast_metrics.csv                    RMSE/MAE: SARIMAX vs baselines
  forecast_next_12_weeks.csv              12-week forecast with 95% interval
  forecast_validation_plot.png            forecast accuracy chart
  forecast_future_plot.png                future forecast chart
  visualizations/charts/*.png             7 analysis charts
""")


if __name__ == "__main__":
    main()
