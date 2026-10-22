"""
LaunchCast - Step 2: Seasonal forecasting of launch suitability
-----------------------------------------------------------------
Why this differs from a plain daily SARIMA:
  * Weather suitability is driven by the ANNUAL cycle (monsoon vs winter),
    not a 7-day cycle. We model WEEKLY average go_score and capture the
    annual cycle with Fourier (sin/cos) terms as exogenous regressors,
    plus ARMA errors for short-term persistence (SARIMAX).
  * We compare against two simple baselines. A model is only useful if it
    beats them:
        1. Monthly climatology (average go_score of that calendar month in
           the training years)
        2. Persistence (repeat the last observed week)
  * Test period = last 52 weeks (a full year), not 30 days.
  * Forecasts and intervals are clipped to [0, 1] (go_score's valid range).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

INPUT_FILE = "daily_go_score.csv"
TEST_WEEKS = 52        # hold out the last year
FORECAST_WEEKS = 12    # weeks to forecast beyond the data
FOURIER_K = 3          # number of sin/cos pairs for the annual cycle
ORDER = (1, 0, 1)      # ARMA(1,1) errors (ADF shows d=0 is sufficient)


# ------------------------------------------------------------------ data
def load_weekly(path: str) -> pd.Series:
    """Daily go_score -> weekly mean (weeks ending Sunday), complete weeks only."""
    daily = pd.read_csv(path, parse_dates=["date"], index_col="date").asfreq("D")
    n_gaps = int(daily["go_score"].isna().sum())
    if n_gaps:
        print(f"Interpolating {n_gaps} missing days.")
    daily["go_score"] = daily["go_score"].interpolate()

    agg = daily["go_score"].resample("W").agg(["mean", "count"])
    weekly = agg.loc[agg["count"] == 7, "mean"].rename("go_score")  # drop partial weeks
    return weekly.asfreq("W")


def fourier_terms(index: pd.DatetimeIndex, k: int = FOURIER_K, period: float = 365.25) -> pd.DataFrame:
    """Annual seasonality as sin/cos pairs, based on day of year."""
    t = np.asarray(index.dayofyear, dtype=float)
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"cos{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------- models
def fit_model(y: pd.Series):
    exog = fourier_terms(y.index)
    model = SARIMAX(y, exog=exog, order=ORDER, trend="c",
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def forecast_model(fitted, index: pd.DatetimeIndex):
    res = fitted.get_forecast(steps=len(index), exog=fourier_terms(index))
    mean = pd.Series(np.asarray(res.predicted_mean), index=index).clip(0, 1)
    ci = np.asarray(res.conf_int())
    lower = pd.Series(ci[:, 0], index=index).clip(0, 1)
    upper = pd.Series(ci[:, 1], index=index).clip(0, 1)
    return mean, lower, upper


def monthly_baseline(train: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    month_mean = train.groupby(train.index.month).mean()
    return pd.Series(month_mean.reindex(index.month).to_numpy(), index=index)


def persistence_baseline(train: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(train.iloc[-1], index=index)


def rmse_mae(actual, pred):
    err = np.asarray(actual) - np.asarray(pred)
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


# ----------------------------------------------------------------- plots
def plot_validation(train, test, preds: dict, path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(train.index[-52:], train.iloc[-52:], label="Train (last 52 wks)", color="tab:blue")
    plt.plot(test.index, test, label="Actual (test)", color="green")
    styles = {"SARIMAX + annual Fourier": ("red", "--"), "Monthly average baseline": ("gray", ":")}
    for name, p in preds.items():
        c, ls = styles.get(name, ("black", "-."))
        plt.plot(test.index, p, label=name, color=c, linestyle=ls)
    plt.title("Weekly Go-Score: Forecast Validation on Held-out Final 52 Weeks")
    plt.xlabel("Week ending")
    plt.ylabel("Mean go_score (0 = NO-GO, 1 = GO)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_future(weekly, mean, lower, upper, path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(weekly.index[-52:], weekly.iloc[-52:], label="Recent actual")
    plt.plot(mean.index, mean, label="Forecast", color="orange")
    plt.fill_between(mean.index, lower, upper, color="orange", alpha=0.2, label="95% interval")
    plt.title(f"Weekly Go-Score Forecast - Next {len(mean)} Weeks")
    plt.xlabel("Week ending")
    plt.ylabel("Mean go_score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------------ main
def main():
    weekly = load_weekly(INPUT_FILE)
    print(f"Weekly series: {len(weekly)} weeks "
          f"({weekly.index.min().date()} to {weekly.index.max().date()})")

    train, test = weekly.iloc[:-TEST_WEEKS], weekly.iloc[-TEST_WEEKS:]
    print(f"Train: {len(train)} weeks | Test: {len(test)} weeks")

    print("\nFitting SARIMAX (ARMA errors + annual Fourier terms)...")
    fitted = fit_model(train)
    print(fitted.summary())

    sarimax_pred, _, _ = forecast_model(fitted, test.index)
    base_month = monthly_baseline(train, test.index)
    base_persist = persistence_baseline(train, test.index)

    rows = []
    for name, p in [("SARIMAX + annual Fourier", sarimax_pred),
                    ("Monthly average baseline", base_month),
                    ("Persistence baseline", base_persist)]:
        rmse, mae = rmse_mae(test, p)
        rows.append({"model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
    metrics = pd.DataFrame(rows)

    print(f"\n--- Validation on final {TEST_WEEKS} weeks ---")
    print(metrics.to_string(index=False))
    best = metrics.loc[metrics["RMSE"].idxmin(), "model"]
    print(f"\nLowest RMSE: {best}")
    if best != "SARIMAX + annual Fourier":
        print("NOTE: SARIMAX did not beat a simple baseline. Report this honestly: "
              "it means the annual cycle explains most of the predictable signal.")
    metrics.to_csv("forecast_metrics.csv", index=False)

    plot_validation(train, test,
                    {"SARIMAX + annual Fourier": sarimax_pred,
                     "Monthly average baseline": base_month},
                    "forecast_validation_plot.png")

    # ---- Refit on ALL data and forecast beyond the end ----
    full_fit = fit_model(weekly)
    future_index = pd.date_range(weekly.index[-1], periods=FORECAST_WEEKS + 1, freq="W")[1:]
    mean, lower, upper = forecast_model(full_fit, future_index)

    table = pd.DataFrame({"forecast_go_score": mean, "lower_95": lower, "upper_95": upper})
    table.index.name = "week_ending"
    print(f"\n--- Forecast for next {FORECAST_WEEKS} weeks ---")
    print(table.round(3))
    table.to_csv(f"forecast_next_{FORECAST_WEEKS}_weeks.csv")

    plot_future(weekly, mean, lower, upper, "forecast_future_plot.png")
    print("\nSaved: forecast_metrics.csv, forecast_next_12_weeks.csv, "
          "forecast_validation_plot.png, forecast_future_plot.png")


if __name__ == "__main__":
    main()
