# Moneyball in the Premier League
### *Does 'Playing Dirty' or 'Shooting More' Actually Win Titles?*

A data-driven blog post investigating 24 seasons of Premier League football.

**Module:** BEE2041 – Data Science in Economics  
**Author:** [Dan Pockett / 740044571]  
**Blog:** [[https://hackmd.io/@95pnGzbHSsKmlkuGBwjlfg/rkrmQzah-x]]

---

## 📖 Project Overview

This project analyses every Premier League match from 2000/01 to 2023/24 (9,030 matches) to answer a specific question: which underlying team statistic — Shots on Target, Corners, or Fouls — is the strongest statistical predictor of league points over a full season?

The analysis uses **multiple OLS regression** (as covered in Unit 5 of the course) on a clean team-season panel dataset engineered from raw match-level data.

**Key finding:** Corners are the single strongest predictor of season points in the multivariate model (β = 8.65), followed by Shots on Target (β = 4.12). Fouling more is negatively associated with points (β = −3.00). The model explains ~53% of variation in season points (R² = 0.527).

---

## 📁 Project Structure

```
epl-moneyball/
├── data/
│   └── raw/
│       └── epl_final.csv          # Raw match-by-match EPL data (2000/01–2024/25)
├── scripts/
│   └── analysis.py                # Main Python script — all cleaning, modelling, plotting
├── figures/
│   ├── plot1_correlation_heatmap.png
│   ├── plot2_shots_vs_points.png
│   ├── plot3_corners_vs_points.png
│   ├── plot4_fouls_vs_points.png
│   ├── plot5_coefficient_plot.png
│   └── plot6_points_by_sot_quartile.png
├── blog.md                        # The full blog post (Markdown)
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## 🔧 Replication Instructions

Follow these steps exactly to reproduce all results and figures from scratch on your machine.

### Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **Git** (check with `git --version`)
- A terminal (Terminal.app on Mac, or any shell)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Pick-Pockett/epl-moneyball.git
cd epl-moneyball
```

---

### Step 2 — Create and Activate a Virtual Environment

Using a virtual environment keeps dependencies isolated and avoids conflicts with your system Python.

```bash
# Create the virtual environment (one-time setup)
python3 -m venv venv

# Activate it — you must do this every time you open a new terminal
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows (if applicable)
```

You'll know it's active when you see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `statsmodels`.

If you'd prefer to install manually:

```bash
pip install pandas numpy matplotlib seaborn statsmodels
```

---

### Step 4 — Run the Analysis Script

From the project root directory (where this README lives):

```bash
python3 scripts/analysis.py
```

The script will:
1. Load `data/raw/epl_final.csv`
2. Clean and reshape the data
3. Aggregate to team-season level (480 observations)
4. Fit the OLS regression and print the full model summary to the terminal
5. Save all 6 figures to the `figures/` folder

**Expected runtime:** < 30 seconds.

---

### Step 5 — Read the Blog Post

Open `blog.md` in any Markdown viewer, or paste it into [HackMD](https://hackmd.io) to view the formatted version with images embedded.

To view the live hosted version: **[https://hackmd.io/@95pnGzbHSsKmlkuGBwjlfg/rkrmQzah-x]**

---

## 📦 Requirements

See `requirements.txt`. The exact versions used during development:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
statsmodels>=0.14
```

---

## 🗂 Data

The raw dataset (`epl_final.csv`) is included in this repository under `data/raw/`. It contains match-by-match statistics for every Premier League match from 2000/01 to 2024/25, with the following columns:

| Column | Description |
|---|---|
| `Season` | EPL season (e.g. `2003/04`) |
| `MatchDate` | Date of match |
| `HomeTeam` / `AwayTeam` | Team names |
| `FullTimeHomeGoals` / `FullTimeAwayGoals` | Full-time score |
| `FullTimeResult` | `H` = Home win, `A` = Away win, `D` = Draw |
| `HomeShotsOnTarget` / `AwayShotsOnTarget` | Shots on target |
| `HomeCorners` / `AwayCorners` | Corners won |
| `HomeFouls` / `AwayFouls` | Fouls committed |
| `HomeYellowCards` / `AwayYellowCards` | Yellow cards |
| `HomeRedCards` / `AwayRedCards` | Red cards |

The 2024/25 season is excluded from analysis as it is only partial.

---

## 📝 Git History

This repository was version-controlled from the start. Key commits include:
- Initial project structure and data loading
- Data cleaning and reshaping to team-season level
- OLS regression model implementation
- Figure generation and styling
- Blog post writing and final edits

---

## ⚖️ Academic Conduct

This project is submitted in accordance with the BEE2041 academic honesty policy. All code was written independently. The data analysis, interpretation, and written blog post are original work.