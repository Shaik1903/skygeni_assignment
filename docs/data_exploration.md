# Part 2: Data Exploration & Insights

## 1. Exploratory Data Analysis

The EDA engine (`src/eda.py`) performs **7 automated analysis modules** on the sales dataset. Each module produces structured outputs that feed both the dashboard visualizations and the insight generation pipeline.

### Analysis Module Overview

| Module | Function | What It Reveals |
|--------|----------|-----------------|
| **Win Rate Trends** | `analyze_win_rate_trends()` | Quarterly & monthly decomposition; detects trend direction (rising/stable/declining); identifies peak and trough quarters with exact rates |
| **Segment Performance** | `analyze_segment_performance()` | Win rate cross-tabulation across region, industry, product type; relative performance scoring (above/below average) |
| **Deal Characteristics** | `analyze_deal_characteristics()` | Won vs. Lost comparison on deal size and sales cycle; identifies statistical differences in deal profiles |
| **Lead Source Analysis** | `analyze_lead_source_performance()` | Conversion rates, volume, average deal size, and total revenue by lead source |
| **Sales Rep Deep Dive** | `analyze_sales_rep_deep()` | Individual rep profiling: win rate, volume, revenue, consistency; identifies top performers and coaching candidates |
| **Time Patterns** | `analyze_time_patterns()` | Day-of-week, month-of-year, and seasonal effects on win rates and deal size |
| **Auto Insights** | `generate_eda_insights()` | Rule-based engine that converts raw analysis into structured business insights with severity, action, and rationale |

All modules are orchestrated by `run_full_eda()`, which returns a single dictionary consumed by the dashboard.

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

**Method:**
1. Bucket all deals into quintiles by deal size
2. Calculate win rate per bucket
3. Compute elasticity: `% change in win rate / % change in deal size`
4. Identify the "sweet spot" — the bucket with the highest revenue potential (win rate × deal size)

**Business value:** Tells the CRO whether "going upmarket" is a viable strategy or a value-destroying trap. The sweet spot range becomes a guardrail for deal pursuit decisions.

**Code:** `src/metrics.py` → `calculate_win_rate_elasticity()`

---

### Metric 3: Segment Momentum Index (SMI)

**Question it answers:** *"Which market segments are gaining or losing steam?"*

**Method:**
1. Split data into recent quarters vs. historical quarters
2. For each segment, calculate trends in: win rate, deal volume, and deal size
3. Combine into a composite momentum score
4. Categorize: Strong Growth, Growing, Stable, Declining, Sharp Decline

**Business value:** Catches segment-level decay before it shows up in aggregate KPIs. Computable across any dimension — region, industry, product type, lead source.

**Code:** `src/metrics.py` → `calculate_segment_momentum_index()`

---

### Metric 4: Deal Velocity Index (DVI)

**Question it answers:** *"How much revenue are we generating per day of sales effort?"*

**Formula:** `(Deal Amount / Sales Cycle Days)` normalized so that 1.0 = median velocity.

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

### Layer 2: LLM-Powered Insights (Optional, Enhanced)
`llm_insights.py` sends the metric summaries to **Gemini 2.5 Flash** to produce deeper, CRO-level narratives. The LLM sees all metrics in context and can identify cross-metric correlations that rule-based logic misses (e.g., "stalling deals are concentrated in your declining regions — this suggests a systemic issue, not individual rep problems").

If the API key is unavailable or the call fails, the system silently falls back to Layer 1 with zero downtime.
