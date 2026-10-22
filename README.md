# LaunchCast - Weather-Based Launch Window Analysis and Forecasting for Indian Spaceports

Analyses 11+ years of hourly weather at Sriharikota (SDSC SHAR) to find when
conditions are favourable for rocket launches, tests whether seasons differ
significantly, and forecasts launch suitability.

## Problem statement
Launch decisions depend heavily on weather (wind, gusts, rain, cloud cover).
This project (1) classifies every hour as GO / CAUTION / NO-GO, (2) finds the
best and worst months, seasons and hours, (3) tests seasonal differences
statistically, and (4) forecasts a weekly launch-suitability score.

## Data
- Source: [Open-Meteo Historical Weather API](https://open-meteo.com/) (hourly, timezone Asia/Kolkata)
- Site: Sriharikota (13.72 N, 80.23 E)
- Variables: temperature, humidity, precipitation, wind speed (10 m), wind gusts, cloud cover
- Period: 2015-01-01 onwards (see `END_DATE` in `download_weather_data.py`)

## Project structure
| File | Purpose |
|---|---|
| `download_weather_data.py` | Downloads and verifies raw hourly data |
| `data_cleaning_corrected.py` | Validation, missing/duplicate handling, outlier report, time features |
| `launch_rules_corrected.py` | GO / CAUTION / NO-GO rules and numeric go_score |
| `step1_stationarity.py` | Builds daily go_score series, ADF stationarity test |
| `step2_forecasting.py` | Weekly SARIMAX forecast vs baselines (RMSE / MAE) |
| `step3_ttest.py` | Welch t-test, effect size, ANOVA / Kruskal-Wallis |
| `visualizations/visualizations.py` | 7 charts in `visualizations/charts/` |
| `main.py` | Runs the whole pipeline in order |

## How to run
```bash
pip install -r requirements.txt
python download_weather_data.py   # only if you need fresh data
python main.py                    # runs all steps
```

## Method summary
- **Launch rules:** NO-GO if wind > 27 km/h, gusts > 46 km/h, any precipitation, or cloud cover > 70%.
  CAUTION if wind > 20, gusts > 35 or cloud > 50%. Otherwise GO.
  go_score: GO = 1, CAUTION = 0.5, NO-GO = 0.
  *These thresholds are illustrative, not official ISRO commit criteria.*
- **Forecasting:** daily go_score averaged to weekly; SARIMAX with ARMA(1,1) errors and
  annual Fourier terms; evaluated on the last 52 weeks against a monthly-average baseline
  and a persistence baseline.
- **Statistics:** Welch t-test (NE Monsoon vs rest of year) on hourly and daily data,
  Cohen's d, one-way ANOVA and Kruskal-Wallis across seasons.

## Results
Fill in after running `python main.py` (see `forecast_metrics.csv` and the charts).

## Limitations
- Reanalysis-based data (Open-Meteo), not on-site observations.
- Thresholds are illustrative; real launch commit criteria also use upper-air winds and lightning data.
- Single site; adding more spaceports would allow comparison.
