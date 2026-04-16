# Moneyball in the Premier League: Does 'Playing Dirty' or 'Shooting More' Actually Win Titles?

*A data-driven investigation into 24 seasons of Premier League football*

---

Every Saturday, across the length and breadth of England, managers bark instructions from their technical areas. "Win the second balls." "Press high." "Get more shots on target." Football is awash with received wisdom — tactical mantras repeated so often they've become orthodoxy. But how much of it is actually backed up by the data?

In 2003, the book *Moneyball* exploded that question open in baseball: using data, the Oakland Athletics discovered that the sport was mispricing certain player skills, and they exploited that mispricing to compete with richer clubs on a fraction of the budget. The Premier League has its own version of this story — clubs like Leicester City's miraculous 2015/16 title win prompted the same question. Can data tell us which on-pitch behaviours are genuinely worth chasing?

This post tries to answer a specific version of that question: across **24 complete Premier League seasons (2000/01 to 2023/24)**, which underlying team statistic — **Shots on Target**, **Corners earned**, or **Fouls committed** — is the strongest statistical predictor of finishing the season with more points?

---

## The Data

The dataset used here contains every single Premier League match from the 2000/01 season through to 2023/24 — **9,030 matches** in total, with no missing values. For each game, we have both teams' shots on target, corners won, fouls committed, yellow and red cards, and of course the result.

The first job was to engineer a **Team-Season** dataset. Each team plays 38 league games per season, and we need to look at the *full picture* — not just one game in isolation. To do this, I calculated each team's points from every match (3 for a win, 1 for a draw, 0 for a loss), then summed everything up over the full season. The result is a panel of **480 team-season observations** — think of it as 480 rows in a spreadsheet, where each row is something like "Arsenal, 2003/04: 90 points, 234 shots on target, 217 corners, 392 fouls."

---

## Live Data Integration: Web Scraping

While the core of this investigation relies on 24 years of historical match data, a key requirement of a professional economic project is the ability to handle live data collection.

Using the BeautifulSoup4 and pandas libraries, the script programmatically extracts the live Premier League table from **Sky Sports**. This serves two purposes:

1. **Validation:** It allows for a quick comparison between historical averages and "live" current-season trends.

2. **Technical Mastery:** It demonstrates a complete data pipeline, from automated collection (scraping) to cleaning and, finally, econometric modelling.

By bridging the gap between historical archives and live web data, the project moves beyond a static analysis into a dynamic, "Moneyball" style tool.

---

## A First Look: What Even Correlates with Points?

Before modelling anything, it's worth just looking at how these variables relate to each other. The heatmap below shows the pairwise **Pearson correlation** between our key metrics across all 480 team-season observations.

![plot1_correlation_heatmap](https://hackmd.io/_uploads/S1qwEGp3Zl.png)


A few things immediately jump out. Both Shots on Target and Corners show a **positive** correlation with season points — teams that do more of these things tend to finish higher. Fouls, on the other hand, show a **negative** correlation — fouling more is associated with fewer points. And notice that Shots on Target and Corners are themselves positively correlated: dangerous teams tend to win corners precisely because they spend more time attacking.

This already gives us a working hypothesis. But correlations alone can be misleading — if Shots on Target and Corners are highly correlated with *each other*, we can't easily say which one is doing the "real" work. That's where regression comes in.

---

## Setting Up the Analysis

To properly test which metric is the strongest predictor, I built a **multiple OLS regression model**. The idea is straightforward: we let the model estimate the relationship between *each* predictor and season points, while *holding the other predictors constant*.

Before running the model, I standardised all three predictors. Standardising means converting each variable into units of *standard deviations away from the mean*, rather than raw totals. This is important: without it, we'd be comparing the effect of "one extra shot on target" against "one extra foul" — but those are on completely different scales. After standardising, all coefficients tell us: "what is the expected change in season points from a **one standard deviation increase** in this variable?"

The model is:

```
Season Points = β₀ + β₁(Shots on Target) + β₂(Corners) + β₃(Fouls) + ε
```

where all three predictors are standardised, and ε is the error term (everything the model doesn't explain).

The scatter plots below show each predictor's individual relationship with points, with a fitted regression line:

![plot2_shots_vs_points](https://hackmd.io/_uploads/BkdbHfa2Zl.png)


![plot3_corners_vs_points](https://hackmd.io/_uploads/HJIQSf6hWg.png)


![plot4_fouls_vs_points](https://hackmd.io/_uploads/SJhEHMahbe.png)


The gold dots represent the **season champion** in each year — the team that finished top of the table. Notice how champions cluster towards the top-right of the shots and corners plots (lots of shots, lots of corners, lots of points), but they scatter more randomly in the fouls plot. That's already a clue about what the model is about to tell us.

---

## The Results: A Surprising Winner

Here is the key output from the regression. I've reported the standardised coefficients (β), standard errors (SE), and a note on statistical significance:

| Predictor | Coefficient (β) | Std. Error | Significant? |
|---|---|---|---|
| **Corners** | **+8.65** | 0.73 | ✅ Yes (p < 0.001) |
| Shots on Target | +4.12 | 0.72 | ✅ Yes (p < 0.001) |
| Fouls | −3.00 | 0.55 | ✅ Yes (p < 0.001) |

**Overall model fit: R² = 0.527**

![plot5_coefficient_plot](https://hackmd.io/_uploads/SJIwSf6hWe.png)


Let's break this down.

**The biggest surprise: Corners is the strongest predictor.** A one standard deviation increase in total corners earned (roughly 35 more per season, or about one extra per game) is associated with **8.65 more league points**, holding everything else constant. That is more than double the effect of shots on target. This is genuinely counterintuitive — goals don't come from corners that often. So what's going on?

The most likely explanation is that corners are a *proxy* for territorial dominance and attacking pressure. Teams that earn lots of corners are teams that spend a lot of time in the opponent's half, forcing keepers to punch the ball out for another corner. In other words, corners aren't causing teams to win — they're a *symptom* of a team that's already playing well. The causal arrow isn't "win more corners → get more points." It's closer to "dominate possession and territory → win more corners *and* get more points."

**Shots on target still matters — a lot.** A one standard deviation increase in shots on target (about 58 per season, or 1.5 per game) is associated with **4.12 more points**. This makes intuitive sense: if you're shooting more on target, you're creating more genuine scoring chances. The relationship is positive, significant, and strong.

The clearest illustration of the shots-on-target advantage comes from quartile analysis:

![plot6_points_by_sot_quartile](https://hackmd.io/_uploads/r1ptBMph-x.png)


Teams in the top quartile for shots on target average **64.1 points per season** — comfortably in the top-four conversation. Teams in the bottom quartile average just **40.2** — a relegation fight. That's a 24-point gap between being shot-shy and shooting with intent.

**And 'playing dirty' genuinely hurts.** Fouling more is associated with **fewer** points (β = −3.00). This is intuitive once you think about it: fouls give the opposition set-piece opportunities, break up your own team's attacking rhythm, and increase the risk of red cards. Contrary to any folk wisdom about "getting stuck in," the data suggests aggressive fouling is a losing strategy at the aggregate level.

---

## The Moneyball Predictor: Be the Manager
Now that we have proven the statistical weight of corners, shots on target, and fouls, let's put the model to the test. 

I have built an interactive OLS Predictor based on the standardised coefficients of our regression. Use the sliders below to act as a Premier League manager. Adjust your team's underlying metrics (measured in standard deviations from the league average) to see exactly how it impacts your predicted points tally for the season. 

*Try maxing out the 'Fouls' slider to see how quickly a dirty tactical setup can drag a team into a relegation battle!*

<iframe src="https://Pick-Pockett.github.io/epl-moneyball/calculator.html" width="100%" height="600" style="border:none; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 10px;"></iframe>

---

## How Much Does This Actually Explain?

The model's R² of **0.527** means that these three variables together explain around **53% of the variation** in season points across all 480 team-seasons. That's genuinely impressive for such a simple, three-variable model — it tells us these metrics capture something real and important about how teams perform.

But 47% remains unexplained. This is a good moment to be honest about what the model *cannot* tell us — what economists call **omitted variable bias**. There are obvious factors we haven't included: the quality of opposition, the clubs' wage bills, whether a team changed manager mid-season, or key injuries. Player quality — measured, say, by transfer spend — almost certainly matters a great deal and is correlated with our predictors. Richer teams tend to shoot more accurately, win more corners, and commit fewer fouls because they have better players, not simply because they have a tactical philosophy around shots on target.

This means we should be careful about the causal interpretation. The model tells us that shots on target, corners, and fouls are **strongly associated** with league points. It doesn't definitively prove that a mid-table club could immediately rise by simply instructing their players to shoot more. The underlying driver of everything we see might partly be *squad quality* — which causes both better stats *and* more points simultaneously.

That said, these patterns are robust across 24 seasons and nearly 500 team-season observations. They are not noise.

---

## Conclusions

So, what does the data say?

1. **Shooting on target is a strong predictor of league success.** Teams in the top quartile for shots on target earn 24 more points per season than those in the bottom quartile. This is one of the clearest signals in the data.

2. **Corners are the unexpected champion predictor.** In the multivariate model, corners have the largest coefficient — suggesting they capture territorial dominance and attacking presence that goes beyond shots alone.

3. **Fouling hurts.** More fouls per season is associated with a statistically significant *reduction* in points. The "hard man" strategy doesn't pay off at the aggregate level.

4. **Our three predictors explain 53% of seasonal variation** in points — a striking result for three simple counting statistics.

The Moneyball lesson for the Premier League, at least from this data, is not that one magic metric wins titles. It's that sustained attacking pressure — measured together by shots on target and corners won — is the clearest statistical footprint of a successful team. Whether that's cause or consequence is the genuinely hard question. But the next time a pundit dismisses corners as meaningless, you can politely point them at the regression output.

---

*Data source: match-by-match EPL statistics, 2000/01 to 2023/24. All analysis performed in Python (pandas, statsmodels) with interactive elements built in JavaScript and HTML. Code and replication instructions available in the [GitHub repository](https://github.com/Pick-Pockett/epl-moneyball).*