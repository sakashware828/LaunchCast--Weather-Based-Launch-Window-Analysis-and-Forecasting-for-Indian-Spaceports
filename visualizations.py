"""LaunchCast charts. Reads ../launchcast_weather_data_classified.csv and
writes 7 PNGs to visualizations/charts/. Run directly or via main.py."""
import calendar
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no pop-up windows: safe to run from main.py
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parent / "launchcast_weather_data_classified.csv"
OUTPUT_DIR = HERE / "charts"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["is_go"] = df["launch_status"].eq("GO")
    df["is_no_go"] = df["launch_status"].eq("NO-GO")
    return df


def save_chart(fig, name, takeaway):
    """Add a readable takeaway BELOW the plot, then save."""
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.08, 0.075, "Takeaway: " + textwrap.fill(takeaway, 115),
             ha="left", va="center", fontsize=10, color="#273348")
    fig.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_go_heatmap(df, y0, y1):
    prob = df.pivot_table(index="month", columns="hour", values="is_go", aggfunc="mean") * 100
    prob = prob.reindex(index=range(1, 13), columns=range(24))
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(prob, cmap="YlGn", vmin=0, vmax=100, linewidths=.25,
                cbar_kws={"label": "GO probability (%)"}, ax=ax)
    ax.set_yticks(np.arange(12) + 0.5, labels=list(calendar.month_abbr)[1:], rotation=0)
    ax.set(xlabel="Hour of day (IST)", ylabel="Month",
           title=f"GO probability by month and hour ({y0}-{y1})")
    m, h = np.unravel_index(np.nanargmax(prob.to_numpy()), prob.shape)
    save_chart(fig, "01_go_heatmap.png",
               f"Best observed month-hour slot: {calendar.month_name[m + 1]} at {h:02d}:00, "
               f"with {prob.iloc[m, h]:.1f}% GO observations. Percentages are historical, "
               f"not forecast probabilities.")


def chart_monthly_no_go(df):
    s = df.groupby("month")["is_no_go"].mean().mul(100).reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=[calendar.month_abbr[m] for m in s.index], y=s.values, color="#cd5b54", ax=ax)
    ax.set(xlabel="Month", ylabel="NO-GO (%)", title="Monthly NO-GO percentage (higher = riskier)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(s):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    save_chart(fig, "02_monthly_no_go.png",
               f"{calendar.month_name[s.idxmax()]} has the highest historical NO-GO rate "
               f"({s.max():.1f}%), making it the riskiest month under the project rules.")


def chart_seasonal_go(df):
    order = ["Winter", "Summer", "SW_Monsoon", "NE_Monsoon"]
    s = df.groupby("season")["is_go"].mean().mul(100).reindex(order)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=s.index, y=s.values, hue=s.index, palette="Set2", legend=False, ax=ax)
    ax.set(xlabel="Season", ylabel="GO (%)", title="GO percentage by season")
    ax.set_ylim(0, 100)
    for i, v in enumerate(s):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    save_chart(fig, "03_seasonal_go.png",
               f"{s.idxmax()} offers the highest historical GO rate ({s.max():.1f}%), "
               f"while {s.idxmin()} has the lowest ({s.min():.1f}%).")


def chart_yearly_trends(df, y0, y1):
    yearly = df.groupby("year").agg(wind=("wind_speed_10m", "mean"),
                                    rainfall=("precipitation", "sum"))
    # Years with fewer than 12 distinct months are incomplete: mark with *
    partial = df.groupby("year")["month"].nunique() < 12
    labels = [f"{y}*" if partial[y] else str(y) for y in yearly.index]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    l1 = ax1.plot(yearly.index, yearly.wind, "o-", color="#277da1", label="Avg wind speed")
    ax1.set(xlabel="Year (* = incomplete year)", ylabel="Average wind speed (km/h)",
            title=f"Yearly wind speed and rainfall ({y0}-{y1})")
    ax2 = ax1.twinx()
    l2 = ax2.plot(yearly.index, yearly.rainfall, "s-", color="#e09f3e", label="Total rainfall")
    ax2.set_ylabel("Total rainfall (mm)")
    ax1.set_xticks(yearly.index)
    ax1.set_xticklabels(labels)
    ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper left")
    note = " Incomplete years (*) understate total rainfall." if partial.any() else ""
    save_chart(fig, "04_yearly_wind_rainfall.png",
               f"Yearly mean wind speed ranges from {yearly.wind.min():.1f} to "
               f"{yearly.wind.max():.1f} km/h. Rainfall is the sum of hourly precipitation "
               f"per year.{note}")


def chart_histograms(df, y0, y1):
    variables = [
        ("wind_speed_10m", "Wind speed", "km/h", "05_wind_speed_hist.png"),
        ("wind_gusts_10m", "Wind gusts", "km/h", "06_wind_gusts_hist.png"),
        ("cloud_cover", "Cloud cover", "%", "07_cloud_cover_hist.png"),
    ]
    for col, title, unit, filename in variables:
        v = df[col].dropna()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.histplot(v, bins=35, color="#5b8db8", edgecolor="white", ax=ax)
        ax.axvline(v.median(), color="#bc423b", linestyle="--", linewidth=2,
                   label=f"Median: {v.median():.1f} {unit}")
        ax.legend()
        ax.set(xlabel=f"{title} ({unit})", ylabel="Hourly observations",
               title=f"Distribution of {title.lower()} ({y0}-{y1})")
        save_chart(fig, filename,
                   f"The median {title.lower()} is {v.median():.1f} {unit}; "
                   f"90% of observed values are at or below {v.quantile(.9):.1f} {unit}.")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_data()
    y0, y1 = df["year"].min(), df["year"].max()
    chart_go_heatmap(df, y0, y1)
    chart_monthly_no_go(df)
    chart_seasonal_go(df)
    chart_yearly_trends(df, y0, y1)
    chart_histograms(df, y0, y1)
    print(f"All 7 charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
