# Part 2: Data Exploration & Insights

## 1. Exploratory Data Analysis (EDA)

The primary deliverable for data exploration is the **[`01_EDA_and_Data_Insights.ipynb`](../notebooks/01_EDA_and_Data_Insights.ipynb)** notebook. This self-contained notebook provides a full narrative walkthrough of the dataset, featuring interactive visualizations and deep-dive analysis.

While the notebook is the primary exploration tool, the underlying logic is powered by a modular EDA engine (`src/eda.py`) that performs **7 automated analysis modules**. This engine ensures consistent results across the notebook, the interactive dashboard, and the AI agent.

### Analysis Module Overview

| Module | Purpose | Notebook Section |
|--------|---------|------------------|
| **Win Rate Trends** | Quarterly & monthly decomposition; detects trend direction | *2.3 Win Rate Over Time* |
| **Segment Performance** | Cross-tabulation across region, industry, product type | *2.4 Segment Drill-down* |
| **Deal Characteristics** | Won vs. Lost comparison on deal size and sales cycle | *3.1 Deal Size Analysis* |
| **Lead Source Analysis** | Conversion rates, volume, and revenue by source | *3.3 Lead Source ROI* |
| **Sales Rep Deep Dive** | Individual rep profiling and consistency scoring | *3.4 Sales Rep Efficiency* |
| **Time Patterns** | Seasonal and cyclical effects on performance | *3.5 Temporal Patterns* |
| **Auto Insights** | AI-ready structured business insights | *4.0 Multi-Metric Insights* |

The analysis returns structured findings consumed by the `Overview` page of the dashboard.

---

## 2. Three Meaningful Business Insights

### Insight 1: Win Rate Is Highly Elastic to Deal Size

**Finding:** Win rate drops from ~58% for the smallest deal bucket to ~31% for the largest bucket—a decline of 27 percentage points. The relationship is not linear; there's a "cliff" around the $50K mark where conversion probability collapses.

**Why it matters:** The sales team is systematically overestimating its ability to close enterprise-sized deals. Each $50K+ deal that enters the pipeline requires the *same* rep capacity as a $20K deal but converts at nearly half the rate. The net effect is lower win rate *and* higher opportunity cost.

**Action it drives:**
- Set deal size guardrails: Deals above the sweet spot range should require VP approval or be routed to a specialist team.
- Compensation: Adjust commissions to reward sweet-spot deals, not just any deal above a threshold.
- This insight is surfaced through the **Win Rate Elasticity (WRE)** metric on both the Overview page and the Win Rate Drivers page.

---

### Insight 2: Pipeline Is Full of "Zombie" Deals

**Finding:** Approximately 30% of closed deals exceeded the 75th percentile of won deals' sales cycle. These deals had near-zero conversion probability — they were statistically "dead" but consuming pipeline space and rep attention.

**Why it matters:** Every day a rep spends nurturing a zombie deal is a day not spent on a new, winnable opportunity. This "wasted capacity" can be quantified: if each stalling deal consumes 5+ hours of rep time over its lifecycle, the aggregate cost is thousands of rep-days per quarter.

**Action it drives:**
- Implement a weekly pipeline hygiene review: any deal past the "winning window" (P75 of won cycle days) should be escalated, killed, or assigned a recovery plan.
- Add automated alerts when deals cross the stall threshold.
- This insight is surfaced through the **Pipeline Qualification Score (PQS)** and its stall detection component.

---

### Insight 3: Segment Momentum Is Diverging

**Finding:** While overall performance appears stable, individual segments are moving in opposite directions. Some regions/industries show improving win rates, increasing volume, and growing deal sizes (positive SMI), while others show simultaneous decline across all three dimensions (negative SMI).

**Why it matters:** When a growing segment masks a declining one, the CRO misses the window to intervene. By the time the declining segment's drag overcomes the growth, the problem is much harder (and more expensive) to fix. This is the "melting ice cube" problem — it looks the same size until it suddenly isn't.

**Action it drives:**
- Reallocate territory investment: shift SDR capacity from declining to stable/growing segments.
- Investigate root cause in declining segments: is it a competitive issue, a rep turnover issue, or a product-market fit issue?
- This insight is surfaced through the **Segment Momentum Index (SMI)**, computed across region, industry, product type, and lead source.

---

## 3. Custom Metrics (4 Invented Metrics)

We designed 4 custom metrics beyond standard sales KPIs. These are not just reformulations of existing metrics — each one was built to answer a specific business question that standard reporting cannot.

### Metric 1: Pipeline Qualification Score (PQS)

**Question it answers:** *"Are we wasting sales capacity on deals that will never close?"*

**Components:**

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| **Deal Qualification Efficiency (DQE)** | `1 - (Median Lost Cycle / Median Won Cycle)` | DQE = 0 → lost deals drag on as long as wins (terrible). DQE > 0.3 → org catches losers early (healthy). |
| **Stall Risk Score** | `Deal Cycle Days / P75(Won Cycle Days)` | < 1.0 = within winning window. 1.0–1.5 = warning. > 1.5 = likely dead. |
| **Composite PQS (0–100)** | `50% × DQE + 30% × (1 - Stall Rate) + 20% × Win Rate Gap` | > 70 = healthy. 40–70 = needs improvement. < 40 = critical. |

**Business value:** PQS provides both the *magnitude* of the problem (org-level score) and the *location* of the problem (deal-level stall flags, segment-level breakdown by rep and region).

**Code:** `src/metrics.py` → `calculate_pipeline_qualification_score()`

---

### Metric 2: Win Rate Elasticity (WRE)

**Question it answers:** *"Does chasing bigger deals actually hurt our win rate?"*

**Components:**

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| **Elasticity Coefficient** | `(% Δ Win Rate) / (% Δ Deal Size)` | < -1.0 = Highly Elastic (High risk). -0.5 to -1.0 = Moderate. > -0.5 = Inelastic (Low risk). |
| **Revenue Sweet Spot** | `Win Rate × Avg Deal Size` | Identifies the size bucket that yields the highest expected revenue per unit of effort. |
| **Win Rate Drop** | `WR(Smallest) - WR(Largest)` | Quantifies the total "conversion penalty" incurred at the top end of the market. |

**Business value:** Tells the CRO whether "going upmarket" is a viable strategy or a value-destroying trap. The sweet spot range becomes a guardrail for deal pursuit decisions.

**Code:** `src/metrics.py` → `calculate_win_rate_elasticity()`

---

### Metric 3: Segment Momentum Index (SMI)

**Question it answers:** *"Which market segments are gaining or losing steam?"*

**Components:**

| Component | Weight | Calculation |
|-----------|--------|-------------|
| **Win Rate Momentum** | 50% | `(Recent Win Rate / Historical Win Rate) - 1` |
| **Volume Momentum** | 30% | `(Recent Deal Count / Historical Deal Count) - 1` |
| **Deal Size Momentum** | 20% | `(Recent Avg Deal Size / Historical Avg Deal Size) - 1` |
| **Composite SMI** | 100% | Weighted sum of signals (Categorized: Growth → Sharp Decline) |

**Business value:** Catches segment-level decay before it shows up in aggregate KPIs. Computable across any dimension — region, industry, product type, lead source.

**Code:** `src/metrics.py` → `calculate_segment_momentum_index()`

---

### Metric 4: Deal Velocity Index (DVI)

**Question it answers:** *"How much revenue are we generating per day of sales effort?"*

**Components:**

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| **Revenue per Day (RPD)** | `Deal Amount / Sales Cycle Days` | Raw efficiency of revenue generation vs. time elapsed. |
| **Normalization Factor** | `RPD / Median(All RPD)` | Benchmarks every deal against the organization's "average" efficiency. |
| **Index Value (DVI)** | `Normalized RPD` | > 1.0 = Above-average efficiency. < 1.0 = Slower than organization average. |

**Business value:** A $100K deal closing in 120 days (DVI = 0.8) is less efficient than a $50K deal closing in 30 days (DVI = 1.6). DVI reveals which deal profiles, reps, and segments generate the most revenue per unit of sales capacity.

**Code:** `src/metrics.py` → `calculate_deal_velocity_index()`

---

## 4. Insight Delivery: Two Layers

### Layer 1: Rule-Based Insights (Always Available)
`metrics.identify_key_insights()` and `eda.generate_eda_insights()` automatically produce 6–10 structured insights with:
- Category (Pipeline Health, Deal Sizing, Segment Shift, etc.)
- Severity (high / medium / low)
- Business action (concrete, specific recommendation)
- Estimated impact (quantified when possible)

### Layer 2: LLM-Powered Insights (Enhanced)
`llm_insights.py` sends the metric summaries to **Gemini 2.5 Flash** to produce deeper, CRO-level narratives. The LLM sees all metrics in context and can identify cross-metric correlations that rule-based logic misses (e.g., "stalling deals are concentrated in your declining regions — this suggests a systemic issue, not individual rep problems").

If the API key is unavailable or the call fails, the system silently falls back to Layer 1 with zero downtime.
