# Part 3: Decision Engine

The primary deliverable for the decision intelligence system is the **[`02_Decision_Engine.ipynb`](../notebooks/02_Decision_Engine.ipynb)** notebook. This notebook contains the complete end-to-end implementation, training, validation, and explainability results for every model.

While the notebook serves as the research and results document, the engines are deployed as high-performance modules in `src/forecasting.py` and interactive views in the `pages/` directory.

> **We implemented ALL FOUR options** from the challenge (B, C, D) plus a bonus conversational AI agent.

---

## Engine B: Win Rate Driver Analysis

**Primary Walkthrough:** [`02_Decision_Engine.ipynb`](../notebooks/02_Decision_Engine.ipynb) (Section 2.0)  
**Implementation:** `pages/1_🎯_Win_Rate_Drivers.py`  
**Core Logic:** `src/forecasting.py` (model training + SHAP)

### Problem Definition

*"Which factors are most responsible for deals winning or losing, and how can a sales leader act on each factor?"*

Standard win/loss reports tell you that "Region X has a lower win rate." But they don't tell you *why* — is it the deal size distribution in that region, the sales cycle, the lead source mix, or individual rep behavior? SHAP-based analysis decomposes individual predictions to reveal exactly which factors push each deal toward win or loss.

### Model Architecture

- **Algorithm:** XGBoost Classifier with 5-fold Stratified Cross-Validation
- **Features (11 engineered):**

| Feature | Engineering | Why |
|---------|------------|-----|
| `log_amount` | `log1p(deal_amount)` | Handles right-skewed deal sizes |
| `log_cycle` | `log1p(sales_cycle_days)` | Normalizes cycle distribution |
| `amt_vs_industry` | `deal_amount / expanding_mean(industry)` | Relative deal sizing within industry (uses expanding mean to prevent data leakage) |
| `amt_vs_rep` | `deal_amount / expanding_mean(rep)` | Is this deal large/small for this rep? |
| `region_enc` | Label encoded | Categorical |
| `industry_enc` | Label encoded | Categorical |
| `product_type_enc` | Label encoded | Categorical |
| `lead_source_enc` | Label encoded | Categorical |
| `created_month` | Month extracted from created_date | Seasonality |
| `is_quarter_end` | Binary: month ∈ {3,6,9,12} | Quarter-end push behavior |
| `days_since_start` | Days since earliest deal in dataset | Time trend |

- **Leakage Prevention:** We use `expanding_mean()` with `.shift(1)` for relative features (`amt_vs_industry`, `amt_vs_rep`). This ensures a deal's training features only use information available *before* that deal existed.
- **Class Imbalance:** `scale_pos_weight` is automatically calculated from win/loss ratio.

### Explainability: SHAP

- **Method:** `shap.TreeExplainer` on the final XGBoost model
- **Global Importance:** Mean |SHAP| per feature, displayed as a ranked horizontal bar chart
- **Visualization:** Interactive Plotly chart showing top 10 factors with color-coded impact intensity
- **Insight Card:** The dashboard identifies the #1 win driver and provides a CEO-level diagnostic summary

### How a Sales Leader Uses This

1. **Look at the top 3 SHAP drivers** — these are the levers with the most influence on win/loss
2. **Check if drivers are controllable** — "Deal Amount vs. Industry Avg" is controllable (repricing); "Region" is less so (requires strategy shift)
3. **Cross-reference with WRE and SMI** (shown on the same page) to understand whether the problem is deal-level or segment-level

---

## Engine C: Revenue Forecast

**Page:** `pages/2_📈_Revenue_Forecast.py`  
**Module:** `src/forecasting.py`

### Problem Definition

*"How much revenue should we expect in the next quarter, and how confident should we be in that number?"*

### Model Architecture: Hybrid Ensemble

We use two complementary models and blend their predictions:

| Model | Captures | Weakness | Role in Ensemble |
|-------|----------|----------|-----------------|
| **Prophet** | Long-term trend, quarterly seasonality, trend changepoints | Ignores deal-level signals; slow to react to recent pipeline changes | 50% weight (trend) |
| **XGBoost Regressor** | Short-term dynamics: lagged revenue, pipeline velocity, deals created | No seasonal decomposition; extrapolation degrades quickly | 50% weight (recency) |

**Weekly Feature Engineering for XGBoost:**

| Feature | Description |
|---------|-------------|
| `revenue_lag_1, _2, _4` | Revenue from 1, 2, 4 weeks ago |
| `revenue_ma_4, _ma_8` | 4-week and 8-week moving averages |
| `pipeline_created_lag1` | Pipeline value created last week (leading indicator) |
| `deals_created_lag1` | Number of deals created last week (leading indicator) |
| `time_index` | Linear trend |
| `month`, `is_quarter_end` | Seasonality features |

### Validation: Walk-Forward Cross-Validation

We do **not** use random train/test splits for time series. Instead, we use walk-forward CV:
1. Train on weeks 1–30, test on weeks 31–34
2. Train on weeks 1–34, test on weeks 35–38
3. ... and so on until the end of data

This produces realistic error estimates because the model only ever predicts *future* data that it has never seen.

**Metrics reported:** MAE, RMSE, MAPE (per fold and aggregate)

### User Controls

| Control | Options | Effect |
|---------|---------|--------|
| **Forecast Horizon** | 4–24 weeks (slider) | Adjusts how far into the future the ensemble predicts |
| **Scenario** | Most Likely / Optimistic (+15%) / Conservative (-15%) | Scales the forecast by a multiplier for planning purposes |
| **Show Components** | Toggle | Overlays the Prophet and XGBoost individual predictions on the chart |

### Output

- Interactive line chart: Historical revenue + forecast with confidence bands
- Headline metrics: Total forecasted revenue, average weekly, trend direction, model MAPE
- Expandable data table with per-week forecast values
- Revenue driver bar chart (SHAP from the deal-level model, shared with Engine B)

---

## Engine D: Pipeline Anomaly Detection

**Primary Walkthrough:** [`02_Decision_Engine.ipynb`](../notebooks/02_Decision_Engine.ipynb) (Section 4.0)  
**Implementation:** `app.py` (Pipeline Alerts tab)  

### Problem Definition

*"Are any pipeline metrics deviating significantly from historical norms? Which segments are behaving unusually?"*

### Method: Statistical Z-Score Analysis

**Overall Anomaly Detection:**
1. Calculate monthly values for a selected metric (win rate, deal count, avg deal size, avg cycle time)
2. Compute rolling mean and rolling standard deviation (configurable window, default 4 months)
3. Calculate Z-score: `(value - rolling_mean) / rolling_std`
4. Flag as anomaly if `|Z-score| > threshold`

**Segment-Level Anomaly Detection:**
1. For each segment value (e.g., each region), split data into recent months vs. historical months
2. Compare: `recent_win_rate vs. historical_win_rate`
3. Normalize by historical standard deviation to get a Z-score
4. Flag segments with `|Z-score| > 1.5` as anomalous
5. Classify severity: `|Z| > 2.5` = High, `|Z| > 2.0` = Medium, else Low

### User Controls

| Control | Options | Effect |
|---------|---------|--------|
| **Sensitivity** | Conservative (2.5σ) / Balanced (2.0σ) / Aggressive (1.5σ) | Controls how many anomalies are flagged |

### Output

- Color-coded alert cards for each detected anomaly, showing:
  - Segment name and type
  - Current vs. historical win rate
  - Change in percentage points
  - Severity badge (High / Medium / Low)
- Interactive line chart showing metric values, rolling average, and anomaly markers (red X's)

### How a Sales Leader Uses This

1. **Monday morning:** Open Alerts tab to see if anything unusual happened last week
2. **Drill down:** Click on a flagged segment to understand what changed
3. **Act:** If a region's win rate dropped 15pp and the anomaly is "High," investigate immediately — don't wait for the quarterly review

---

## SkyRalph — Conversational CRM Agent

**Page:** `pages/3_💬_CRM_Agent.py`  
**Module:** `src/agent/graph.py` + `src/agent/tools.py`

### Problem Definition

*"Can a CRO ask natural-language questions about their pipeline and get instant, data-backed answers?"*

### Architecture

- **Framework:** LangGraph (ReAct pattern: Reason → Act → Observe → Repeat)
- **LLM:** Gemini 2.5 Flash
- **4 Agent Tools:**

| Tool | Function | When Agent Uses It |
|------|----------|-------------------|
| `get_key_metrics_summary` | Returns PQS, WRE, SMI, DVI in JSON | "How is the business doing?" |
| `get_revenue_forecast` | Runs full forecast pipeline | "What's our Q3 projection?" |
| `explain_win_rate_trends` | Returns EDA insights | "Why are we losing more deals?" |
| `analyze_sales_data` | Ad-hoc pandas queries via LLM-powered `create_pandas_dataframe_agent` | "How many deals did Rep X close in January?", "Show me a chart of win rate by region" |

### Example Interactions

| User Says | Agent Does |
|-----------|-----------|
| "Give me a health check" | Calls `get_key_metrics_summary` → summarizes PQS, WRE, SMI |
| "Forecast next 8 weeks" | Calls `get_revenue_forecast(weeks=8)` → formats forecast table |
| "Why is APAC underperforming?" | Calls `explain_win_rate_trends` + `analyze_sales_data("APAC win rate by quarter")` → produces narrative |
| "Show me deal size distribution" | Calls `analyze_sales_data` → generates Plotly histogram → renders chart |

### How a Sales Leader Uses This

Instead of navigating dashboards, the CRO opens a chat and asks questions in plain English. The agent handles tool selection, data retrieval, computation, and response formatting automatically. This is the "last mile" of decision intelligence—reducing the cognitive load of interacting with analytics from "navigate 4 pages of charts" to "ask one question."
