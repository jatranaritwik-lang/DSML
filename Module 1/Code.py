"""
Mini Project — Exploratory Data Analysis on India's COVID-19 Data
-----------------------------------------------------------------
Reproduces every step and all 12 charts from the report:
Load -> Inspect -> Clean -> Explore (groupby / pivot_table) -> Visualise -> Summarise

Run:   python covid_india_eda.py
Needs: pip install pandas numpy matplotlib seaborn
Output: charts saved to ./charts/, cleaned CSV saved to ./cleaned_covid_india.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

SHOW_PLOTS = False          # set True to also pop each chart up in a window
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)


def save(fig, name):
    """Save a figure into charts/ and optionally display it."""
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{name}")


# =============================================================================
# STEP 1 — LOAD
# =============================================================================
print("\n=== STEP 1: Load ===")
CASES_URL = ("https://raw.githubusercontent.com/imdevskp/"
             "covid-19-india-data/master/complete.csv")
VAX_URL = ("https://raw.githubusercontent.com/owid/covid-19-data/master/"
           "public/data/vaccinations/country_data/India.csv")

cases = pd.read_csv(CASES_URL)
vax = pd.read_csv(VAX_URL)
print("Cases shape      :", cases.shape)
print("Vaccination shape:", vax.shape)
print(cases.head())


# =============================================================================
# STEP 2 — INSPECT BEFORE TOUCHING ANYTHING
# =============================================================================
print("\n=== STEP 2: Inspect ===")
print(cases.dtypes, "\n")
print("Missing values per column:\n", cases.isna().sum(), "\n")
print("Unique state/UT names:", cases["Name of State / UT"].nunique())
print(sorted(cases["Name of State / UT"].unique()))


# =============================================================================
# STEP 3 — CLEAN
# =============================================================================
print("\n=== STEP 3: Clean ===")
cases.columns = [
    "date", "state", "lat", "long", "confirmed",
    "deaths", "cured", "new_cases", "new_deaths", "new_recovered",
]

# Fix data types (Death column arrives as text)
cases["date"] = pd.to_datetime(cases["date"])
cases["deaths"] = pd.to_numeric(cases["deaths"], errors="coerce").fillna(0).astype(int)
cases["confirmed"] = cases["confirmed"].astype(int)
cases["cured"] = cases["cured"].astype(int)

# Merge duplicate spellings of the same state/UT
name_fix = {
    "Telangana***": "Telangana",
    "Telengana": "Telangana",
    "Union Territory of Jammu and Kashmir": "Jammu and Kashmir",
    "Union Territory of Ladakh": "Ladakh",
    "Union Territory of Chandigarh": "Chandigarh",
}
cases["state"] = cases["state"].replace(name_fix)

# Drop duplicate (date, state) rows the merge can create
cases = (cases.drop_duplicates(subset=["date", "state"])
              .sort_values(["state", "date"])
              .reset_index(drop=True))

# Derived column
cases["active"] = cases["confirmed"] - cases["deaths"] - cases["cured"]

print("Clean state count:", cases["state"].nunique())
print("Days covered     :", cases["date"].nunique(),
      f"({cases['date'].min().date()} to {cases['date'].max().date()})")
print("Remaining NaNs   :", cases.isna().sum().sum())


# =============================================================================
# STEP 4 — EXPLORE (groupby / pivot_table)
# =============================================================================
print("\n=== STEP 4: Explore ===")

# National daily totals
national = cases.groupby("date", as_index=False)[["confirmed", "deaths", "cured"]].sum()
national["recovery_rate"] = (national["cured"] / national["confirmed"] * 100).round(2)
national["death_rate"] = (national["deaths"] / national["confirmed"] * 100).round(2)
national["new_confirmed"] = national["confirmed"].diff().fillna(national["confirmed"])
national[["recovery_rate", "death_rate"]] = national[["recovery_rate", "death_rate"]].fillna(0)
national["month"] = national["date"].dt.strftime("%Y-%m")

latest_date = cases["date"].max()
row = national.iloc[-1]
print(f"As of {latest_date.date()}:")
print(f"  Total confirmed : {int(row['confirmed']):,}")
print(f"  Total recovered : {int(row['cured']):,}")
print(f"  Total deaths    : {int(row['deaths']):,}")
print(f"  Recovery rate   : {row['recovery_rate']}%")
print(f"  Death rate      : {row['death_rate']}%")
print(f"  Peak daily cases: {int(national['new_confirmed'].max()):,} "
      f"on {national.loc[national['new_confirmed'].idxmax(), 'date'].date()}")

# Latest snapshot per state
latest = cases[cases["date"] == latest_date].copy()
latest["recovery_rate"] = (latest["cured"] / latest["confirmed"] * 100).round(2)
latest["death_rate"] = (latest["deaths"] / latest["confirmed"] * 100).round(2)
latest = latest.sort_values("confirmed", ascending=False).reset_index(drop=True)

top10 = latest.head(10)
print("\nTop 10 states:\n",
      top10[["state", "confirmed", "deaths", "cured", "recovery_rate", "death_rate"]])

# Month-wise new cases for the top 5 states
cases["month"] = cases["date"].dt.strftime("%Y-%m")
top5_states = top10["state"].head(5).tolist()
month_pivot = cases[cases["state"].isin(top5_states)].pivot_table(
    values="new_cases", index="state", columns="month", aggfunc="sum", fill_value=0
)
print("\nNew cases by month (top 5 states):\n", month_pivot)


# =============================================================================
# STEP 5 — VISUALISE (12 charts, numbered as in the report)
# =============================================================================
print("\n=== STEP 5: Visualise ===")

# Figure 1 — National cumulative trend
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(national["date"], national["confirmed"], label="Confirmed", color="#2563eb", lw=2)
ax.plot(national["date"], national["cured"], label="Recovered", color="#16a34a", lw=2)
ax.plot(national["date"], national["deaths"], label="Deaths", color="#dc2626", lw=2)
ax.set_title("India: Cumulative COVID-19 Cases Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("People")
ax.legend(); fig.autofmt_xdate()
save(fig, "fig01_national_cumulative.png")

# Figure 2 — Daily new confirmed cases
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(national["date"], national["new_confirmed"], color="#2563eb", width=1.0)
ax.set_title("India: Daily New Confirmed Cases")
ax.set_xlabel("Date"); ax.set_ylabel("New cases")
fig.autofmt_xdate()
save(fig, "fig02_daily_new_cases.png")

# Figure 3 — Top 10 states by confirmed cases
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10, y="state", x="confirmed", hue="state",
            palette="Blues_r", legend=False, ax=ax)
ax.set_title(f"Top 10 States by Total Confirmed Cases (as of {latest_date.date()})")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("")
save(fig, "fig03_top10_states.png")

# Figure 4 — Share of national cases (pie)
top6 = latest.nlargest(6, "confirmed")[["state", "confirmed"]]
others = latest["confirmed"].sum() - top6["confirmed"].sum()
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.pie(top6["confirmed"].tolist() + [others],
       labels=top6["state"].tolist() + ["Rest of India"],
       autopct="%1.1f%%", startangle=90,
       colors=sns.color_palette("Blues_r", 7),
       wedgeprops={"edgecolor": "white"})
ax.set_title("Share of Total Confirmed Cases by State")
save(fig, "fig04_case_share_pie.png")

# Figure 5 — Growth curves, top 5 states
fig, ax = plt.subplots(figsize=(8, 4.5))
for s in top5_states:
    sub = cases[cases["state"] == s]
    ax.plot(sub["date"], sub["confirmed"], label=s, lw=2)
ax.set_title("Confirmed Cases Over Time — Top 5 States")
ax.set_xlabel("Date"); ax.set_ylabel("Confirmed cases")
ax.legend(fontsize=8); fig.autofmt_xdate()
save(fig, "fig05_top5_growth.png")

# Figure 6 — Recovery rate, top 10 states
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10.sort_values("recovery_rate"), y="state", x="recovery_rate",
            hue="state", palette="Greens", legend=False, ax=ax)
ax.set_title("Recovery Rate (%) — Top 10 Worst-Hit States")
ax.set_xlabel("Recovery rate (%)"); ax.set_ylabel("")
save(fig, "fig06_recovery_rate_top10.png")

# Figure 7 — National recovery rate over time
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(national["date"], national["recovery_rate"], color="#16a34a", lw=2)
ax.set_title("India: National Recovery Rate Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("Recovery rate (%)")
fig.autofmt_xdate()
save(fig, "fig07_national_recovery_rate.png")

# Figure 8 — Recovery vs death rate (bubble = case count)
sizeable = latest[latest["confirmed"] >= 1000]
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=sizeable, x="recovery_rate", y="death_rate", size="confirmed",
                sizes=(40, 600), hue="confirmed", palette="Blues", legend=False, ax=ax)
for _, r in sizeable.nlargest(6, "confirmed").iterrows():
    ax.annotate(r["state"], (r["recovery_rate"], r["death_rate"]), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_title("Recovery Rate vs Death Rate by State (bubble = case count)")
ax.set_xlabel("Recovery rate (%)"); ax.set_ylabel("Death rate (%)")
save(fig, "fig08_recovery_vs_death.png")

# Figure 9 — Correlation heatmap
corr_cols = ["confirmed", "deaths", "cured", "new_confirmed", "recovery_rate", "death_rate"]
corr = national[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Between National COVID Metrics")
save(fig, "fig09_correlation_heatmap.png")

# Figure 10 — Monthly spread of daily new cases
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=national, x="month", y="new_confirmed", hue="month",
            palette="Blues", legend=False, ax=ax)
ax.set_title("Spread of Daily New Cases by Month")
ax.set_xlabel("Month"); ax.set_ylabel("Daily new cases")
save(fig, "fig10_monthly_boxplot.png")

# Figure 11 — 10 least-affected states/UTs
bottom10 = latest[latest["confirmed"] > 0].nsmallest(10, "confirmed")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=bottom10, y="state", x="confirmed", hue="state",
            palette="Oranges", legend=False, ax=ax)
ax.set_title(f"10 Least-Affected States/UTs by Confirmed Cases (as of {latest_date.date()})")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("")
save(fig, "fig11_least_affected.png")

# Figure 12 — Vaccination progress
vax["date"] = pd.to_datetime(vax["date"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(vax["date"], vax["total_vaccinations"] / 1e9,
        label="Total doses (Bn)", color="#7c3aed", lw=2)
ax.plot(vax["date"], vax["people_fully_vaccinated"] / 1e9,
        label="Fully vaccinated (Bn)", color="#0d9488", lw=2)
ax.set_title("India: COVID-19 Vaccination Progress (2021-2024)")
ax.set_xlabel("Date"); ax.set_ylabel("People / doses (in billions)")
ax.legend(); fig.autofmt_xdate()
save(fig, "fig12_vaccination_progress.png")


# =============================================================================
# STEP 6 — SUMMARISE (numbers behind the written findings)
# =============================================================================
print("\n=== STEP 6: Summary ===")
top_state = latest.iloc[0]
share = top_state["confirmed"] / latest["confirmed"].sum() * 100
top6_share = top6["confirmed"].sum() / latest["confirmed"].sum() * 100
v_last = vax.dropna(subset=["total_vaccinations"]).iloc[-1]
fv_last = vax["people_fully_vaccinated"].dropna().iloc[-1]

print(f"States/UTs covered        : {cases['state'].nunique()}")
print(f"Days covered              : {cases['date'].nunique()}")
print(f"Hardest-hit state         : {top_state['state']} "
      f"({int(top_state['confirmed']):,} cases, {share:.1f}% of India)")
print(f"Top 6 states' share       : {top6_share:.1f}%")
print(f"Best recovery (top 10)    : {top10.loc[top10['recovery_rate'].idxmax(), 'state']} "
      f"({top10['recovery_rate'].max()}%)")
print(f"Worst recovery (top 10)   : {top10.loc[top10['recovery_rate'].idxmin(), 'state']} "
      f"({top10['recovery_rate'].min()}%)")
print(f"Highest death rate (top10): {top10.loc[top10['death_rate'].idxmax(), 'state']} "
      f"({top10['death_rate'].max()}%)")
print(f"Least-affected            : {', '.join(bottom10['state'].head(4))}")
print(f"Vaccine doses (latest)    : {v_last['total_vaccinations'] / 1e9:.2f} Bn "
      f"as of {v_last['date'].date()}")
print(f"Fully vaccinated (latest) : {fv_last / 1e6:,.0f} million")

cases.to_csv("cleaned_covid_india.csv", index=False)
print(f"\nSaved cleaned_covid_india.csv — {cases.shape[0]} rows. Done.")