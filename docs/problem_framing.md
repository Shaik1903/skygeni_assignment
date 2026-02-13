# Part 1: Problem Framing

## 1. Identifying the Real Business Problem

The CRO's complaint—*"Win rate has dropped... pipeline looks healthy... I don't know what to focus on"*—is a textbook **Decision Intelligence Gap**. The sales organization has plenty of data but lacks the analytical framework to translate it into action.

On the surface, it looks like a reporting problem. Dig deeper, and it reveals three systemic failures:

### Failure 1: Volume ≠ Quality
A "healthy pipeline" measured by total deal count or total dollar value is misleading if the composition has shifted. If the team is now chasing larger, more complex deals (or deals in unfamiliar industries), the pipeline *looks* full but the *conversion probability* per deal has silently dropped. This is the "empty calorie" pipeline problem.

### Failure 2: Lagging Indicators Dominate Decision-Making
The CRO is relying on win rate (a lagging indicator) to diagnose problems. By the time win rate drops, the root cause (e.g., a competitor launched in Q1, reps started targeting Enterprise accounts in Q2) has already played out. The CRO needs **leading indicators** like deal velocity, pipeline aging, and segment momentum—metrics that signal trouble *before* deals close.

### Failure 3: Aggregate Averages Mask Segment-Level Crises
A company-wide win rate of 40% might hide the fact that Region A is at 55% (thriving) while Region B collapsed to 20% (in crisis). Without segment-level decomposition, the CRO is making decisions based on "average truths" that are no one's reality.

**Our framing shifts the problem from "Why is win rate down?" to "Where is quality eroding, what behaviors are driving it, and what should each team do differently?"**

---

## 2. Key Questions an AI System Must Answer

We designed the system to answer four escalating tiers of questions:

| Tier | Question | Type | Our Implementation |
|------|----------|------|--------------------|
| **1. Diagnostic** | *"Why is our win rate fluctuating?"* | Root cause | EDA engine + SHAP analysis |
| **2. Predictive** | *"Where are we heading next quarter?"* | Forecasting | Prophet + XGBoost ensemble |
| **3. Prescriptive** | *"What specific deals need attention?"* | Action-oriented | PQS stall detection |
| **4. Strategic** | *"Where is our sweet spot for growth?"* | Portfolio strategy | WRE + SMI metrics |

The system is designed so that a CRO can look at a single dashboard and understand:
- **What happened** (trends and anomalies),
- **Why** (driver analysis via SHAP),
- **What's coming** (revenue forecast), and
- **What to do about it** (prioritized recommendations).

---

## 3. Metrics That Matter: Beyond Win Rate

Standard metrics (win rate, pipeline value, average deal size) are necessary but insufficient. They tell you *what* happened but not *why* or *what to do*. We designed **four custom metrics** specifically for this problem:

### Custom Metric 1: Pipeline Qualification Score (PQS) — *"How disciplined is our pipeline?"*

| Component | Formula | What It Measures |
|-----------|---------|-----------------|
| **Deal Qualification Efficiency (DQE)** | `1 - (Median Cycle Lost / Median Cycle Won)` | Whether we "fail fast" on bad deals |
| **Deal Stall Risk Score (DSRS)** | `Deal Cycle Days / P75(Won Cycle Days)` | Whether individual deals have exceeded the statistical winning window |
| **Composite PQS** | `50% × DQE + 30% × Stall Rate + 20% × Win Rate Gap` | Overall pipeline health (0–100) |

**Why it matters:** If lost deals take just as long as won deals (DQE ≈ 0), the sales team can't distinguish winners from losers early. That's a massive capacity waste—every day a rep spends on a dead deal is a day not spent on a winnable one.

**Action it drives:** Kill deals scoring >1.5× the winning window. Coach reps in segments with low PQS.

### Custom Metric 2: Win Rate Elasticity (WRE) — *"Does chasing bigger deals actually hurt us?"*

**Formula:** Elasticity coefficient between deal size buckets and their corresponding win rates.

**Why it matters:** If a 20% increase in deal size leads to a 50% drop in win probability, the team is destroying value by pursuing "elephant deals." WRE quantifies this trade-off and identifies the **sweet spot**—the deal size range with the optimal balance of win probability and revenue potential.

**Action it drives:** Set deal size guardrails. Route outsized deals to specialist teams.

### Custom Metric 3: Segment Momentum Index (SMI) — *"Which segments are growing vs. dying?"*

**Formula:** Composite of `Win Rate Trend + Volume Trend + Deal Size Trend` over recent vs. historical quarters.

**Why it matters:** A segment can appear "fine" in aggregate while its trend is deteriorating quarter-over-quarter. SMI catches this momentum shift early—before a full-blown crisis shows up in lagging KPIs. It prevents the "melting ice cube" problem.

**Action it drives:** Double down on high-momentum segments. Investigate or exit declining ones.

### Custom Metric 4: Deal Velocity Index (DVI) — *"How much revenue per day of sales effort?"*

**Formula:** `(Deal Amount / Sales Cycle Days)` normalized against the median.

**Why it matters:** A $100K deal that takes 120 days to close is less valuable per unit of effort than a $50K deal that closes in 30 days. DVI reveals which deal types and reps generate the most revenue per day of sales capacity.

---

## 4. Key Assumptions

1. **Data Integrity:** We assume `outcome` (Won/Lost) is accurately recorded and reflects signed contracts, not verbal commitments. If reps "sandbag" deal dates, velocity metrics will be distorted.

2. **Stable Process:** We assume the sales stages have not been redefined during the analysis period. If "Discovery" meant different things last year vs. this year, stage-based analysis becomes unreliable.

3. **Representative History:** We assume that the historical data (12–24 months) reflects conditions relevant to the future. If the market has fundamentally changed (e.g., a new competitor entered), historical patterns may not hold.

4. **Sales Cycle as Proxy:** We use `sales_cycle_days` (created → closed) as a proxy for sales effort. In reality, effort is not uniformly distributed across the cycle—a deal might be dormant for 30 days then active for 5. We lack activity-level data to measure true effort.

5. **No External Data:** We analyze only internal CRM data. We do not incorporate competitor moves, macroeconomic signals, or customer firmographic changes. These are significant real-world drivers that our system currently ignores.
