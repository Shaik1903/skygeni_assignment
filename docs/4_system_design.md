# Part 4: Mini System Design — Sales Insight & Alert System

## 1. High-Level Architecture

If SkyGeni were to productize this, the system would be a modular, event-driven insight platform. Below is the target architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER LAYER                                  │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│   │ Executive     │  │ Win Rate     │  │ Revenue Forecast         │ │
│   │ Dashboard     │  │ Diagnostic   │  │ (Prophet + XGBoost)      │ │
│   │ (Overview +   │  │ Center       │  │                          │ │
│   │  Alerts +     │  │ (SHAP +      │  │                          │ │
│   │  Actions)     │  │  WRE + SMI)  │  │                          │ │
│   └──────┬───────┘  └──────┬───────┘  └───────────┬──────────────┘ │
│          │                 │                       │                 │
│   ┌──────┴─────────────────┴───────────────────────┴──────────────┐ │
│   │                   💬 SkyRalph CRM Agent                       │ │
│   │           (LangGraph ReAct + Gemini 2.5 Flash)                │ │
│   │      Natural language interface to ALL analytics tools        │ │
│   └───────────────────────────┬───────────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────┐
│                        INTELLIGENCE LAYER                           │
│                                │                                    │
│   ┌───────────────┐  ┌────────┴────────┐  ┌──────────────────────┐ │
│   │ EDA Engine    │  │ LLM Insights    │  │ Anomaly Detection    │ │
│   │ (Trends,      │  │ (Gemini 2.5     │  │ (Z-Score on          │ │
│   │  Segments,    │  │  Flash via      │  │  rolling metrics,    │ │
│   │  Rep analysis │  │  LangChain)     │  │  segment-level)      │ │
│   │  Lead source) │  │                 │  │                      │ │
│   └───────┬───────┘  └────────┬────────┘  └───────────┬──────────┘ │
│           │                   │                        │            │
│   ┌───────┴───────┐  ┌───────┴────────┐  ┌────────────┴─────────┐ │
│   │ Custom        │  │ Forecasting    │  │ Win/Loss Classifier  │ │
│   │ Metrics       │  │ Module         │  │ (XGBoost + SHAP)     │ │
│   │ (PQS, WRE,   │  │ (Prophet +     │  │                      │ │
│   │  SMI, DVI)    │  │  XGBoost       │  │ 5-fold Stratified CV │ │
│   │               │  │  Ensemble +    │  │ Feature Engineering  │ │
│   │               │  │  Walk-fwd CV)  │  │                      │ │
│   └───────┬───────┘  └───────┬────────┘  └────────────┬─────────┘ │
└───────────┼──────────────────┼─────────────────────────┼───────────┘
            │                  │                         │
┌───────────┼──────────────────┼─────────────────────────┼───────────┐
│           └──────────────────┴─────────────────────────┘           │
│                        DATA LAYER                                   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │  Data Loader + Preprocessor                                   │ │
│   │  • CSV/Database ingestion                                     │ │
│   │  • Validation report (missing values, duplicates, ranges)     │ │
│   │  • Feature engineering (time features, deal size buckets,     │ │
│   │    sales cycle categories, binary outcome conversion)         │ │
│   └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │  📁 skygeni_sales_data.csv (5,000+ deals)                    │ │
│   └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

### Step 1: Ingestion & Validation
```
CSV → data_loader.load_data() → validate_data() → preprocess_data()
```
- **Input:** Raw CSV with 11 fields (deal_id through outcome)
- **Validation:** Generates a report on missing values, duplicates, date ranges, and distributions
- **Feature Engineering:** Automatically creates `is_won`, `sales_cycle_days`, `closed_year_quarter`, `closed_year_month`, `deal_size_category`, `cycle_category`, and temporal features

### Step 2: Analysis Layer (Parallel)
Once data is loaded, three independent analysis pipelines execute:

| Pipeline | Module | Outputs |
|----------|--------|---------|
| **Custom Metrics** | `metrics.py` | PQS (with segment breakdown), WRE (with sweet spot), SMI (multi-segment), DVI |
| **EDA Engine** | `eda.py` | Trend analysis, segment performance, deal characteristics, rep deep-dives, time patterns |
| **Forecasting** | `forecasting.py` | Weekly revenue forecast (12-week horizon), Win/Loss classifier, SHAP values |

### Step 3: Insight Generation
- **Rule-Based Insights:** `metrics.identify_key_insights()` and `eda.generate_eda_insights()` produce structured business insights automatically
- **LLM-Powered Insights:** `llm_insights.py` sends metric summaries to Gemini 2.5 Flash to generate deeper, CRO-level narratives with severity ratings and actionable recommendations
- **Fallback:** If LLM is unavailable (no API key, rate limit), the system gracefully degrades to rule-based insights with zero downtime

### Step 4: Presentation
All results are cached (`st.cache_data`, `st.cache_resource`) and presented via Streamlit's multi-page application with interactive Plotly charts.

---

## 3. Example Alerts and Insights

### Anomaly Alerts (Z-Score Based)
| Alert | Trigger | Action |
|-------|---------|--------|
| 🔴 **Win Rate Collapse in APAC** | Win rate dropped from 45% → 28% (Z-score: -2.8) | Investigate: New competitor? Rep turnover? |
| 🟡 **Deal Cycle Spike in Enterprise** | Average cycle increased from 60 → 95 days (Z-score: 2.1) | Review: Are larger deals entering pipeline without executive sponsorship? |
| 🟢 **Mid-Market Acceleration** | Volume up 30%, win rate stable at 52% | Opportunity: Allocate more SDR capacity to mid-market leads |

### Custom Metric Insights (Auto-Generated)
| Metric | Insight | Severity |
|--------|---------|----------|
| **PQS** | "30% of pipeline deals have exceeded the statistical winning window (>75 days). These account for ~2,400 wasted rep-days this quarter." | 🔴 High |
| **WRE** | "Win rate drops from 58% to 31% as deal size crosses $50K. The sweet spot is $20K–$40K range where win rate peaks." | 🟡 Medium |
| **SMI** | "Technology vertical is in sharp decline (SMI: -0.42). Win rate, volume, and deal size are all trending down." | 🔴 High |

---

## 4. How Often It Runs

| Component | Frequency | Rationale |
|-----------|-----------|-----------|
| **Data Refresh** | Daily (or on CRM sync) | Deals close throughout the day |
| **Metric Calculation** | On-demand with 1-hour cache | Heavy computation; hourly is sufficient for executive use |
| **Model Retraining** | Weekly (Sunday night batch) | Enough new data to justify refit; not so frequent that it introduces instability |
| **Anomaly Detection** | Daily | Time-sensitive—an anomaly detected Friday is useless on Monday |
| **LLM Insight Refresh** | On-demand per session | Expensive API calls; regenerate when user opens the dashboard |

---

## 5. Failure Cases and Limitations

### Data Quality Failures
| Failure | Impact | Mitigation |
|---------|--------|------------|
| **Missing close dates** | PQS and DVI become unreliable | Validation report flags these; exclude from calculation |
| **Inconsistent stage names** | Segment analysis breaks | Standardize via preprocessing; alert if unknown stages appear |
| **Backdate manipulation** | Velocity metrics are garbage | Cross-validate with CRM audit logs |

### Model Failures
| Failure | Impact | Mitigation |
|---------|--------|------------|
| **Feature drift** (new product line) | Classifier predicts poorly for unseen categories | Monitor out-of-distribution scores; retrain when new products accumulate >100 deals |
| **Small segment sizes** | SMI on a region with 5 deals is noise, not signal | Minimum sample threshold (n ≥ 10) enforced in code |
| **Prophet seasonality on short data** | Seasonal decomposition unreliable with <2 years | Fall back to XGBoost-only forecast |

### Operational Failures
| Failure | Impact | Mitigation |
|---------|--------|------------|
| **LLM API unavailable** | No AI-powered insights | Graceful fallback to rule-based insights (implemented) |
| **Large data volume** | Slow UI response | Streamlit caching + session state caching of ML models |
| **Concurrent users** | Streamlit reruns block each other | Production: Deploy on Streamlit Cloud or Kubernetes with multiple workers |

---

## 6. Production Scaling Path

If SkyGeni productized this today:

1. **Data Backend:** Replace CSV with Snowflake/BigQuery. Use `dbt` for transformations.
2. **Compute:** Move model training to Vertex AI or SageMaker. Serve predictions via REST API.
3. **Orchestration:** Use Airflow or Prefect for daily ETL + weekly retraining.
4. **Monitoring:** MLflow for model versioning. Great Expectations for data quality gates.
5. **Multi-Tenancy:** Each customer gets isolated data + shared model architecture (fine-tuned per tenant).
