"""
A/B Test Analysis for an E-Commerce Website
Comparing Two Pricing Strategies Using Hypothesis Testing
---------------------------------------------------------
Reproduces the report: cleaning -> descriptive stats -> Welch t-test ->
chi-square test -> one-way ANOVA (by country) -> figures -> recommendation.

Files expected in the same folder:
    ab_data.csv      (required)  user_id, timestamp, group, landing_page, converted
    countries.csv    (optional)  user_id, country   -> needed for the ANOVA / Figure 3

Run:   python ab_test_analysis.py
Needs: pip install pandas numpy scipy matplotlib seaborn
Output: figures in ./figures/, cleaned data in ./ab_data_clean.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

AB_FILE = "ab_data.csv"
COUNTRY_FILE = "countries.csv"
ALPHA = 0.05
SHOW_PLOTS = False            # True = also open each figure in a window
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
COLORS = {"control": "#4C72B0", "treatment": "#DD8452"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{name}")


def verdict(p):
    return "REJECT H0 (significant)" if p < ALPHA else "FAIL TO REJECT H0 (not significant)"


# =============================================================================
# 1. LOAD
# =============================================================================
print("=" * 70, "\n1. LOAD DATA\n" + "=" * 70)
df = pd.read_csv(AB_FILE)
print("Raw shape:", df.shape)
print(df.head(), "\n")
print(df.dtypes, "\n")
print("Missing values:\n", df.isna().sum())


# =============================================================================
# 2. CLEAN
# =============================================================================
print("\n" + "=" * 70, "\n2. CLEAN DATA\n" + "=" * 70)

# 2a. Group/page mismatches: control must see old_page, treatment must see new_page
mismatch = ((df["group"] == "control") & (df["landing_page"] != "old_page")) | \
           ((df["group"] == "treatment") & (df["landing_page"] != "new_page"))
print(f"Group/page mismatched rows : {mismatch.sum():,}")
print(f"Duplicate user_id rows (raw): {df['user_id'].duplicated().sum():,}")

clean = df[~mismatch].copy()

# 2b. Duplicate users: keep the first recorded visit
clean["timestamp"] = pd.to_datetime(clean["timestamp"])
clean = clean.sort_values("timestamp")
dups_left = clean["user_id"].duplicated().sum()
print(f"Duplicate users left after mismatch removal: {dups_left:,}")
clean = clean.drop_duplicates(subset="user_id", keep="first").reset_index(drop=True)

# 2c. Merge country data if available
has_country = os.path.exists(COUNTRY_FILE)
if has_country:
    countries = pd.read_csv(COUNTRY_FILE)
    clean = clean.merge(countries, on="user_id", how="inner")
    print(f"Merged with {COUNTRY_FILE}: {len(countries):,} rows")
else:
    print(f"NOTE: {COUNTRY_FILE} not found -> country ANOVA and Figure 3 will be skipped.")

print(f"\nRows after cleaning: {len(clean):,}")

summary = clean.groupby("group")["converted"].agg(users="count", conversions="sum")
summary["conv_rate_%"] = (summary["conversions"] / summary["users"] * 100).round(3)
print("\nTable 1 — Sample sizes after cleaning")
print(summary)
print(f"Total users: {summary['users'].sum():,}   Total conversions: {summary['conversions'].sum():,}")

# Figure 1 — sample size per group
fig, ax = plt.subplots(figsize=(6, 4.5))
order = clean["group"].value_counts().index.tolist()
counts = clean["group"].value_counts()
bars = ax.bar(order, counts[order], color=[COLORS[g] for g in order], edgecolor="black")
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}",
            ha="center", va="bottom", fontweight="bold")
ax.set_title("Sample Size by Group (after cleaning)")
ax.set_ylabel("Number of Users")
save(fig, "fig1_sample_size.png")


# =============================================================================
# 3. DESCRIPTIVE OVERVIEW
# =============================================================================
print("\n" + "=" * 70, "\n3. DESCRIPTIVE OVERVIEW\n" + "=" * 70)
control = clean.loc[clean["group"] == "control", "converted"]
treatment = clean.loc[clean["group"] == "treatment", "converted"]

p_c, p_t = control.mean(), treatment.mean()
n_c, n_t = len(control), len(treatment)
print(f"Control conversion   : {p_c * 100:.3f}%  (n={n_c:,})")
print(f"Treatment conversion : {p_t * 100:.3f}%  (n={n_t:,})")
print(f"Raw difference       : {(p_t - p_c) * 100:+.3f} percentage points")

# Figure 2 — conversion rate with 95% CI
ci_c = 1.96 * np.sqrt(p_c * (1 - p_c) / n_c) * 100
ci_t = 1.96 * np.sqrt(p_t * (1 - p_t) / n_t) * 100
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["control", "treatment"], [p_c * 100, p_t * 100], yerr=[ci_c, ci_t],
              capsize=8, color=[COLORS["control"], COLORS["treatment"]], edgecolor="black")
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, f"{b.get_height():.2f}%",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 14)
ax.set_title("Conversion Rate by Pricing Strategy (95% CI)")
ax.set_ylabel("Conversion Rate (%)")
save(fig, "fig2_conversion_rate_ci.png")


# =============================================================================
# 4. TEST 1 — WELCH'S INDEPENDENT T-TEST
# =============================================================================
print("\n" + "=" * 70, "\n4. TEST 1: WELCH'S t-TEST\n" + "=" * 70)
print("H0: mu_treatment = mu_control    H1: mu_treatment != mu_control")

t_stat, t_p = stats.ttest_ind(treatment, control, equal_var=False)

diff = p_t - p_c
var_t, var_c = treatment.var(ddof=1), control.var(ddof=1)
se = np.sqrt(var_t / n_t + var_c / n_c)
# Welch–Satterthwaite degrees of freedom
df_w = (var_t / n_t + var_c / n_c) ** 2 / (
    (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1))
t_crit = stats.t.ppf(1 - ALPHA / 2, df_w)
ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

pooled_sd = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
cohens_d = diff / pooled_sd

print("\nTable 2 — Welch's t-test")
print(f"  Control mean conversion      : {p_c * 100:.3f}%")
print(f"  Treatment mean conversion    : {p_t * 100:.3f}%")
print(f"  Difference (treat - control) : {diff * 100:+.3f} pp")
print(f"  95% CI of difference         : [{ci_low * 100:.3f}%, {ci_high * 100:.3f}%]")
print(f"  t-statistic                  : {t_stat:.3f}")
print(f"  p-value                      : {t_p:.3f}")
print(f"  Cohen's d                    : {cohens_d:.4f}")
print(f"  -> {verdict(t_p)}")


# =============================================================================
# 5. TEST 2 — CHI-SQUARE TEST OF INDEPENDENCE
# =============================================================================
print("\n" + "=" * 70, "\n5. TEST 2: CHI-SQUARE TEST\n" + "=" * 70)
print("H0: strategy and conversion are independent    H1: they are related")

contingency = pd.crosstab(clean["group"], clean["converted"])
contingency.columns = ["Not Converted", "Converted"]
print("\nTable 3 — Contingency table")
print(contingency)

chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)   # Yates-corrected (2x2)
print("\nExpected counts if independent:")
print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).round(1))
print("\nTable 4 — Chi-square test")
print(f"  Chi-square statistic : {chi2:.3f}")
print(f"  Degrees of freedom   : {dof}")
print(f"  p-value              : {chi_p:.3f}")
print(f"  -> {verdict(chi_p)}")


# =============================================================================
# 6. TEST 3 — ONE-WAY ANOVA (country, and country x strategy)
# =============================================================================
print("\n" + "=" * 70, "\n6. TEST 3: ONE-WAY ANOVA\n" + "=" * 70)
print("H0: mu_1 = mu_2 = ... = mu_k    H1: at least one group mean differs")

f1 = p1 = f2 = p2 = None
if has_country:
    by_country = clean.groupby("country")["converted"].agg(users="count", rate="mean")
    by_country["rate"] = (by_country["rate"] * 100).round(2)
    print("\nTable 5 — Conversion rate by country")
    print(by_country)

    # ANOVA across countries
    country_groups = [g["converted"].values for _, g in clean.groupby("country")]
    f1, p1 = stats.f_oneway(*country_groups)

    # ANOVA across the 6 country x strategy combinations
    clean["segment"] = clean["group"] + " - " + clean["country"]
    seg_groups = [g["converted"].values for _, g in clean.groupby("segment")]
    f2, p2 = stats.f_oneway(*seg_groups)

    print("\nTable 6 — One-way ANOVA")
    print(f"  Across {len(country_groups)} countries          : F = {f1:.3f}, p = {p1:.3f} -> {verdict(p1)}")
    print(f"  Across {len(seg_groups)} country x strategy  : F = {f2:.3f}, p = {p2:.3f} -> {verdict(p2)}")

    # Figure 3 — conversion by country and strategy
    seg = clean.pivot_table(values="converted", index="country", columns="group",
                            aggfunc="mean") * 100
    print("\nConversion (%) by country x strategy:\n", seg.round(2))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seg.plot(kind="bar", ax=ax, color=[COLORS["control"], COLORS["treatment"]],
             edgecolor="black", rot=0)
    ax.set_title("Conversion Rate by Country and Strategy")
    ax.set_xlabel("Country"); ax.set_ylabel("Conversion Rate (%)")
    ax.legend(title="Strategy")
    save(fig, "fig3_country_strategy.png")
else:
    print(f"Skipped: add {COUNTRY_FILE} (user_id, country) to this folder to run the ANOVA.")


# =============================================================================
# 7. CONCLUSION
# =============================================================================
print("\n" + "=" * 70, "\n7. CONCLUSION\n" + "=" * 70)
results = [("Welch t-test", t_p), ("Chi-square", chi_p)]
if has_country:
    results += [("ANOVA (countries)", p1), ("ANOVA (country x strategy)", p2)]
for name, p in results:
    print(f"  {name:28s} p = {p:.3f}  -> {verdict(p)}")

if all(p >= ALPHA for _, p in results):
    print("\nNo test finds a significant difference. The new pricing page does NOT")
    print("improve conversion. Recommendation: keep the existing (old) pricing page.")
else:
    print("\nAt least one test is significant — inspect the direction of the effect")
    print("before deciding whether to roll out the new page.")

clean.to_csv("ab_data_clean.csv", index=False)
print(f"\nSaved ab_data_clean.csv ({len(clean):,} rows). Done.")