"""
analysis.py
──────────────────────────────────────────────────────────────────────────────
Project:  Moneyball in the Premier League
Module:   BEE2041 – Data Science in Economics
Author:   [Dan Pockett / 740044571]

What this script does:
    1. Loads raw match-by-match EPL data (2000/01 – 2023/24)
    2. Engineers match-level points from results
    3. Reshapes and aggregates to a Team-Season panel dataset
    4. Runs an OLS multiple regression to test which underlying
       metric best predicts season points
    5. Saves 6 publication-quality figures to /figures/

Dependencies: pandas, numpy, matplotlib, seaborn, statsmodels
    Install: pip install pandas numpy matplotlib seaborn statsmodels
──────────────────────────────────────────────────────────────────────────────
"""

# ── Standard library ───────────────────────────────────────────────────────
from pathlib import Path

# ── Third-party ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf   # Unit 5 OLS regression tool

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Paths — relative to project root so the script works after cloning
ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "epl_final.csv"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)   # create figures/ if it doesn't exist

# Colour palette — Premier League brand colours
ACCENT = "#E8284B"   # PL red
BLUE   = "#37003C"   # PL dark purple
GOLD   = "#F0A500"   # highlight gold

# Global matplotlib style — keeps all plots consistent and clean
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F8F8",
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & VALIDATE RAW DATA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

df = pd.read_csv(DATA_RAW)
print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# The dataset includes a partial 2024/25 season — drop it so every team
# has a comparable number of games per season (38).
df = df[df["Season"] != "2024/25"].copy()
print(f"  After removing partial 2024/25 season: {len(df):,} rows")
print(f"  Seasons covered: {df['Season'].nunique()} ({df['Season'].min()} – {df['Season'].max()})")

# Quick sanity check — should be zero missing values
assert df.isnull().sum().sum() == 0, "Unexpected nulls found — check raw data!"
print("  ✓ No missing values detected")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: ENGINEER MATCH-LEVEL POINTS
# ══════════════════════════════════════════════════════════════════════════

print("\nSTEP 2: Calculating match points")

# Football points system: Win = 3, Draw = 1, Loss = 0
# The raw column 'FullTimeResult' contains: 'H' (home win), 'A' (away win), 'D' (draw)
HOME_PTS = {"H": 3, "D": 1, "A": 0}
AWAY_PTS = {"H": 0, "D": 1, "A": 3}

df["HomePoints"] = df["FullTimeResult"].map(HOME_PTS)
df["AwayPoints"] = df["FullTimeResult"].map(AWAY_PTS)

print(f"  Points distribution (home): {df['HomePoints'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: RESHAPE TO LONG FORMAT
# ══════════════════════════════════════════════════════════════════════════

print("\nSTEP 3: Reshaping to long format (one row per team-per-match)")

# Each match has TWO teams.  We split home and away into separate DataFrames
# then stack them.  After stacking, every row = one team's performance in one match.

home = df[[
    "Season", "HomeTeam", "HomePoints",
    "HomeShotsOnTarget", "HomeCorners", "HomeFouls",
    "HomeYellowCards", "HomeRedCards"
]].copy()
home.columns = [
    "Season", "Team", "Points",
    "ShotsOnTarget", "Corners", "Fouls", "YellowCards", "RedCards"
]

away = df[[
    "Season", "AwayTeam", "AwayPoints",
    "AwayShotsOnTarget", "AwayCorners", "AwayFouls",
    "AwayYellowCards", "AwayRedCards"
]].copy()
away.columns = [
    "Season", "Team", "Points",
    "ShotsOnTarget", "Corners", "Fouls", "YellowCards", "RedCards"
]

# Stack home and away records vertically
long_df = pd.concat([home, away], ignore_index=True)
print(f"  Long-format rows: {len(long_df):,} (should be 2 × match rows)")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: AGGREGATE TO TEAM-SEASON LEVEL
# ══════════════════════════════════════════════════════════════════════════

print("\nSTEP 4: Aggregating to Team-Season level")

# Group by Season + Team and sum each metric across all 38 matches
team_season = (
    long_df
    .groupby(["Season", "Team"], as_index=False)
    .agg(
        TotalPoints       = ("Points",        "sum"),
        TotalShotsOnTarget= ("ShotsOnTarget", "sum"),
        TotalCorners      = ("Corners",        "sum"),
        TotalFouls        = ("Fouls",          "sum"),
        TotalYellowCards  = ("YellowCards",    "sum"),
        TotalRedCards     = ("RedCards",       "sum"),
        GamesPlayed       = ("Points",         "count"),  # count as proxy for games
    )
)

# Robustness filter: keep only team-seasons with ≥ 30 games
# (handles any relegated/promoted teams with partial data)
team_season = team_season[team_season["GamesPlayed"] >= 30].copy()

print(f"  Team-season observations: {len(team_season)}")
print("\n  Descriptive statistics:")
print(team_season[
    ["TotalPoints", "TotalShotsOnTarget", "TotalCorners", "TotalFouls"]
].describe().round(1).to_string())


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: OLS REGRESSION
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5: OLS Regression")
print("=" * 60)

# Standardise all three predictors so their coefficients are directly
# comparable — i.e. we can say "a 1 SD increase in X leads to β more points".
# Without standardising, the raw coefficients can't be compared because
# Shots, Corners, and Fouls have very different scales.

for col in ["TotalShotsOnTarget", "TotalCorners", "TotalFouls"]:
    std_name = col.replace("Total", "Std")
    team_season[std_name] = (
        (team_season[col] - team_season[col].mean()) / team_season[col].std()
    )

# Fit OLS model using statsmodels formula API
# Formula notation:  TotalPoints ~ predictor1 + predictor2 + predictor3
model = smf.ols(
    formula="TotalPoints ~ StdShotsOnTarget + StdCorners + StdFouls",
    data=team_season
).fit()

# Print full regression summary to console
print(model.summary())

# Extract key values for use in the coefficient plot
coef_df = pd.DataFrame({
    "Variable":  ["Shots on Target", "Corners", "Fouls"],
    "Coef":      model.params[1:].values,
    "SE":        model.bse[1:].values,
    "p_value":   model.pvalues[1:].values,
})
coef_df["CI_lo"] = coef_df["Coef"] - 1.96 * coef_df["SE"]
coef_df["CI_hi"] = coef_df["Coef"] + 1.96 * coef_df["SE"]

r_squared = model.rsquared


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: GENERATE FIGURES
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 6: Generating figures")
print("=" * 60)

# Helper to save and close each figure
def save_fig(fig, filename):
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ── Figure 1: Correlation Heatmap ─────────────────────────────────────────
# Shows at a glance which metrics are correlated with points and with each other
corr_cols   = ["TotalPoints","TotalShotsOnTarget","TotalCorners","TotalFouls","TotalYellowCards"]
corr_labels = ["Points","Shots on Target","Corners","Fouls","Yellow Cards"]
corr_mat    = team_season[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 6))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap=cmap, center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, linecolor="#DDDDDD",
            xticklabels=corr_labels, yticklabels=corr_labels,
            annot_kws={"size": 10, "weight": "bold"})
ax.set_title("Correlation Matrix of Key Team-Season Metrics\n(2000/01 – 2023/24)",
             fontsize=13, fontweight="bold", pad=15)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
save_fig(fig, "plot1_correlation_heatmap.png")


# ── Figure 2: Shots on Target vs Points ──────────────────────────────────
# Classic scatter plot; gold dots highlight the champion team each season
champs = team_season.loc[team_season.groupby("Season")["TotalPoints"].idxmax()]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(team_season["TotalShotsOnTarget"], team_season["TotalPoints"],
           alpha=0.4, s=30, color=BLUE, edgecolors="none", label="All Team-Seasons")
m, b_val = np.polyfit(team_season["TotalShotsOnTarget"], team_season["TotalPoints"], 1)
x_line = np.linspace(team_season["TotalShotsOnTarget"].min(),
                     team_season["TotalShotsOnTarget"].max(), 200)
ax.plot(x_line, m*x_line + b_val, color=ACCENT, linewidth=2.5,
        label=f"OLS fit  (slope = {m:.2f})")
ax.scatter(champs["TotalShotsOnTarget"], champs["TotalPoints"],
           color=GOLD, s=80, zorder=5, edgecolors=BLUE,
           linewidths=0.8, label="Season Champions")
ax.set_xlabel("Total Shots on Target per Season")
ax.set_ylabel("Total League Points")
ax.set_title("Do More Shots on Target Mean More Points?\nEvery EPL Team-Season (2000/01 – 2023/24)",
             fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
save_fig(fig, "plot2_shots_vs_points.png")


# ── Figure 3: Corners vs Points ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(team_season["TotalCorners"], team_season["TotalPoints"],
           alpha=0.4, s=30, color="#2196F3", edgecolors="none", label="All Team-Seasons")
m_c, b_c = np.polyfit(team_season["TotalCorners"], team_season["TotalPoints"], 1)
x_c = np.linspace(team_season["TotalCorners"].min(), team_season["TotalCorners"].max(), 200)
ax.plot(x_c, m_c*x_c + b_c, color=ACCENT, linewidth=2.5,
        label=f"OLS fit  (slope = {m_c:.2f})")
ax.scatter(champs["TotalCorners"], champs["TotalPoints"],
           color=GOLD, s=80, zorder=5, edgecolors=BLUE, linewidths=0.8,
           label="Season Champions")
ax.set_xlabel("Total Corners per Season")
ax.set_ylabel("Total League Points")
ax.set_title("Do More Corners Lead to More Points?\nEvery EPL Team-Season (2000/01 – 2023/24)",
             fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
save_fig(fig, "plot3_corners_vs_points.png")


# ── Figure 4: Fouls vs Points ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(team_season["TotalFouls"], team_season["TotalPoints"],
           alpha=0.4, s=30, color="#4CAF50", edgecolors="none", label="All Team-Seasons")
m_f, b_f = np.polyfit(team_season["TotalFouls"], team_season["TotalPoints"], 1)
x_f = np.linspace(team_season["TotalFouls"].min(), team_season["TotalFouls"].max(), 200)
ax.plot(x_f, m_f*x_f + b_f, color=ACCENT, linewidth=2.5,
        label=f"OLS fit  (slope = {m_f:.2f})")
ax.scatter(champs["TotalFouls"], champs["TotalPoints"],
           color=GOLD, s=80, zorder=5, edgecolors=BLUE, linewidths=0.8,
           label="Season Champions")
ax.axhline(team_season["TotalPoints"].median(), color="grey",
           linestyle="--", alpha=0.7, linewidth=1.2, label="Median Points (all teams)")
ax.set_xlabel("Total Fouls per Season")
ax.set_ylabel("Total League Points")
ax.set_title("Does 'Playing Dirty' Actually Work?\nFouls vs League Points (2000/01 – 2023/24)",
             fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
save_fig(fig, "plot4_fouls_vs_points.png")


# ── Figure 5: OLS Coefficient Plot ───────────────────────────────────────
# The key output — shows each predictor's effect size with 95% CIs
fig, ax = plt.subplots(figsize=(7, 4))
for i, row in coef_df.reset_index(drop=True).iterrows():
    color = ACCENT if row["Coef"] > 0 else BLUE
    ax.errorbar(
        row["Coef"], i,
        xerr=[[row["Coef"] - row["CI_lo"]], [row["CI_hi"] - row["Coef"]]],
        fmt='o', color=color, ecolor=color,
        capsize=6, markersize=10, elinewidth=2, capthick=2
    )

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df["Variable"], fontsize=12)
ax.set_xlabel("Standardised Regression Coefficient (β)\n[95% Confidence Intervals]", fontsize=11)
ax.set_title(
    f"OLS Regression: What Actually Predicts League Points?\n"
    f"(Standardised coefficients; R² = {r_squared:.3f})",
    fontweight="bold"
)
pos_patch = mpatches.Patch(color=ACCENT, label="Positive effect on points")
neg_patch = mpatches.Patch(color=BLUE,   label="Negative effect on points")
ax.legend(handles=[pos_patch, neg_patch], fontsize=10, loc="lower right")
plt.tight_layout()
save_fig(fig, "plot5_coefficient_plot.png")


# ── Figure 6: Average Points by Shots-on-Target Quartile ─────────────────
# A clean, intuitive summary of the shots → points relationship
team_season["SOT_Quartile"] = pd.qcut(
    team_season["TotalShotsOnTarget"], q=4,
    labels=["Q1\n(Fewest Shots)", "Q2", "Q3", "Q4\n(Most Shots)"]
)
q_stats = (team_season
           .groupby("SOT_Quartile", observed=True)["TotalPoints"]
           .agg(["mean", "sem"])
           .reset_index())

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    q_stats["SOT_Quartile"], q_stats["mean"],
    color=[BLUE, "#5c2a7e", "#a0307e", ACCENT],
    edgecolor="white", linewidth=0.5, zorder=3
)
ax.errorbar(q_stats["SOT_Quartile"], q_stats["mean"],
            yerr=1.96 * q_stats["sem"],
            fmt='none', color='black', capsize=5, elinewidth=1.5,
            capthick=1.5, zorder=4)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel("Shots on Target Quartile")
ax.set_ylabel("Average Season Points")
ax.set_title("More Shots, More Points:\nAverage Points by Shots-on-Target Quartile",
             fontweight="bold")
ax.set_ylim(0, q_stats["mean"].max() * 1.15)
plt.tight_layout()
save_fig(fig, "plot6_points_by_sot_quartile.png")


# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
print("\n✅  All steps complete.")
print(f"    Figures saved to: {FIG_DIR}")
print(f"\n    R² = {r_squared:.4f}  |  n = {len(team_season)}")
print(f"\n    Coefficient summary:")
print(coef_df[["Variable","Coef","SE","p_value"]].to_string(index=False, float_format="{:.3f}".format))
