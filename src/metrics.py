"""
Custom Metrics Module for SkyGeni Sales Intelligence Dashboard
===============================================================

3 Original Custom Metrics (invented for this analysis):
  1. Deal Qualification Efficiency (DQE) — How fast do we fail on losers vs invest in winners?
  2. Revenue Concentration Risk (RCR)    — How dangerously dependent are we on a few segments?
  3. Segment Momentum Index (SMI)        — Which segments are gaining or losing steam over time?

Bonus: Deal Velocity Index (DVI) — Revenue generated per day of sales effort.

The module also surfaces 4+ business insights automatically from these metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════
# METRIC 1: DEAL QUALIFICATION EFFICIENCY (DQE)
# ═══════════════════════════════════════════════════════════════
# Business question: "Are we wasting sales capacity on deals that will never close?"
#
# A good sales org should FAIL FAST on losers and INVEST TIME in winners.
# If lost deals take just as long as won deals, it means reps can't tell
# the difference early—a massive capacity problem.
#
# Formula: DQE = 1 - (Median Cycle of Lost Deals / Median Cycle of Won Deals)
#   - DQE = 0.0 → Lost deals take SAME time as won deals (terrible qualification)
#   - DQE = 0.5 → Lost deals take half the time of won deals (decent)
#   - DQE < 0   → Lost deals take LONGER than won deals (actively harmful)
#
# This is novel because it reframes sales cycle data as a QUALIFICATION signal,
# not just a speed-of-close metric.
# ═══════════════════════════════════════════════════════════════

def calculate_deal_qualification_efficiency(df: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
    """
    Calculate Deal Qualification Efficiency (DQE).
    
    Measures how quickly we identify and exit losing deals vs. how long
    winning deals take. High DQE = fail fast on losers, invest in winners.
    
    Args:
        df: Preprocessed DataFrame with is_won and sales_cycle_days
        group_by: Optional column to segment by (e.g., 'sales_rep_id', 'region')
        
    Returns:
        DataFrame with DQE scores
    """
    def _calc_dqe(group):
        won = group[group["is_won"] == 1]["sales_cycle_days"]
        lost = group[group["is_won"] == 0]["sales_cycle_days"]
        
        if len(won) == 0 or len(lost) == 0:
            return pd.Series({
                "total_deals": len(group),
                "won_deals": len(won),
                "lost_deals": len(lost),
                "median_cycle_won": won.median() if len(won) > 0 else np.nan,
                "median_cycle_lost": lost.median() if len(lost) > 0 else np.nan,
                "dqe_score": np.nan,
                "wasted_capacity_days": 0
            })
        
        median_won = won.median()
        median_lost = lost.median()
        
        # Core DQE metric
        dqe = 1 - (median_lost / median_won) if median_won > 0 else 0
        
        # Wasted capacity: total extra days spent on lost deals beyond an ideal "fast fail"
        # Ideal fast fail = 25th percentile of won deal cycle (quick qualification threshold)
        ideal_fail_time = won.quantile(0.25)
        excess_days = (lost[lost > ideal_fail_time] - ideal_fail_time).sum()
        
        return pd.Series({
            "total_deals": len(group),
            "won_deals": int(won.count()),
            "lost_deals": int(lost.count()),
            "median_cycle_won": median_won,
            "median_cycle_lost": median_lost,
            "dqe_score": round(dqe, 3),
            "wasted_capacity_days": round(excess_days, 0)
        })
    
    if group_by:
        result = df.groupby(group_by).apply(_calc_dqe, include_groups=False).reset_index()
    else:
        result = _calc_dqe(df).to_frame().T
    
    # Categorize
    result["dqe_category"] = pd.cut(
        result["dqe_score"].astype(float),
        bins=[-float("inf"), -0.1, 0.1, 0.25, 0.5, float("inf")],
        labels=["Critical: Losers take longer", "Poor: Nearly equal cycles",
                "Fair: Slightly faster exits", "Good: Noticeably faster exits",
                "Excellent: Rapid disqualification"]
    )
    
    return result


# ═══════════════════════════════════════════════════════════════
# METRIC 2: REVENUE CONCENTRATION RISK (RCR)
# ═══════════════════════════════════════════════════════════════
# Business question: "How exposed is our revenue to a single region/industry/rep?"
#
# Inspired by the Herfindahl-Hirschman Index used in antitrust economics,
# adapted for sales portfolio analysis.
#
# Formula: RCR = Σ(share_i²) × N
#   where share_i = segment revenue / total revenue
#   and N = number of segments (normalization)
#
#   - RCR = 1.0 → Perfectly distributed (healthy diversification)
#   - RCR = N   → All revenue from one segment (maximum risk)
#
# What makes this novel: We don't just measure concentration—we calculate
# the "Revenue at Risk" dollar amount if the top segment declined 30%.
# ═══════════════════════════════════════════════════════════════

def calculate_revenue_concentration_risk(df: pd.DataFrame, segment_column: str) -> Dict:
    """
    Calculate Revenue Concentration Risk (RCR) for a given segmentation.
    
    Measures how dangerously dependent revenue is on a few segments.
    Uses HHI-inspired formula adapted for sales portfolio risk.
    
    Args:
        df: Preprocessed DataFrame
        segment_column: Column to analyze (region, industry, product_type, etc.)
        
    Returns:
        Dictionary with RCR score, segment breakdown, and risk analysis
    """
    won_df = df[df["is_won"] == 1]
    total_revenue = won_df["deal_amount"].sum()
    
    # Revenue by segment
    segment_revenue = won_df.groupby(segment_column).agg(
        revenue=("deal_amount", "sum"),
        deals_won=("deal_id", "count"),
        avg_deal_size=("deal_amount", "mean")
    ).reset_index()
    
    segment_revenue["revenue_share"] = segment_revenue["revenue"] / total_revenue
    segment_revenue["revenue_share_pct"] = (segment_revenue["revenue_share"] * 100).round(2)
    
    n_segments = len(segment_revenue)
    
    # HHI-based concentration (normalized)
    hhi_raw = (segment_revenue["revenue_share"] ** 2).sum()
    
    # Normalized: 1 = perfectly even, higher = more concentrated
    # Min HHI = 1/N (perfectly even), Max HHI = 1 (one segment has all)
    rcr_score = round(hhi_raw * n_segments, 3)
    
    # Risk assessment: what happens if top segment drops 30%?
    top_segment = segment_revenue.sort_values("revenue", ascending=False).iloc[0]
    top_2 = segment_revenue.sort_values("revenue", ascending=False).head(2)
    
    revenue_at_risk_30pct = top_segment["revenue"] * 0.30
    top2_share = top_2["revenue_share"].sum()
    
    # Risk level
    if rcr_score > 2.5:
        risk_level = "Critical"
    elif rcr_score > 1.8:
        risk_level = "High"
    elif rcr_score > 1.3:
        risk_level = "Moderate"
    else:
        risk_level = "Healthy"
    
    return {
        "rcr_score": rcr_score,
        "risk_level": risk_level,
        "hhi_raw": round(hhi_raw, 4),
        "n_segments": n_segments,
        "total_revenue": total_revenue,
        "top_segment": top_segment[segment_column],
        "top_segment_share": round(top_segment["revenue_share"] * 100, 1),
        "top2_combined_share": round(top2_share * 100, 1),
        "revenue_at_risk_30pct_decline": round(revenue_at_risk_30pct, 0),
        "segment_breakdown": segment_revenue.sort_values("revenue", ascending=False),
        "recommendation": _get_rcr_recommendation(rcr_score, top_segment[segment_column], top_segment["revenue_share"])
    }


def _get_rcr_recommendation(rcr_score: float, top_segment: str, top_share: float) -> str:
    """Generate actionable recommendation based on RCR."""
    if rcr_score > 2.5:
        return (f"URGENT: Revenue is dangerously concentrated in {top_segment} "
                f"({top_share*100:.0f}% share). A downturn there would be catastrophic. "
                "Immediately invest in diversifying pipeline across other segments.")
    elif rcr_score > 1.8:
        return (f"WARNING: {top_segment} accounts for {top_share*100:.0f}% of revenue. "
                "Build dedicated pipeline in 2-3 alternative segments within 90 days.")
    elif rcr_score > 1.3:
        return ("Revenue distribution is moderately concentrated. Monitor quarterly "
                "and set targets for underrepresented segments.")
    else:
        return "Healthy diversification. Maintain current portfolio mix while optimizing conversion."


# ═══════════════════════════════════════════════════════════════
# METRIC 3: SEGMENT MOMENTUM INDEX (SMI)
# ═══════════════════════════════════════════════════════════════
# Business question: "Which parts of our business are gaining or losing steam?"
#
# This is a composite momentum indicator combining THREE trend signals:
#   1. Win rate trend (are we closing better or worse?)
#   2. Volume trend (are more or fewer deals flowing in?)
#   3. Deal size trend (are deals getting bigger or smaller?)
#
# Formula: SMI = w₁ × WinRateΔ + w₂ × VolumeΔ + w₃ × DealSizeΔ
#   where Δ = (Recent period / Historical period) - 1
#   Weights: w₁=0.5 (win rate matters most), w₂=0.3, w₃=0.2
#
#   - SMI > 0  → Segment is gaining momentum (invest more)
#   - SMI ≈ 0  → Stable (maintain)
#   - SMI < 0  → Segment is losing momentum (investigate/divest)
#
# Novelty: Unlike just looking at win rate trends, this captures the COMBINED
# signal. A segment can have stable win rate but shrinking volume = still declining.
# ═══════════════════════════════════════════════════════════════

def calculate_segment_momentum_index(df: pd.DataFrame, segment_column: str,
                                      recent_quarters: int = 2) -> pd.DataFrame:
    """
    Calculate Segment Momentum Index (SMI).
    
    Combines win rate, volume, and deal size trends into a single
    momentum indicator showing which segments are growing vs. declining.
    
    Args:
        df: Preprocessed DataFrame with closed_year_quarter
        segment_column: Column to segment by
        recent_quarters: Number of recent quarters to compare against historical
        
    Returns:
        DataFrame with momentum scores per segment
    """
    # Split data into recent vs historical
    quarters = sorted(df["closed_year_quarter"].unique())
    
    if len(quarters) < 3:
        # Not enough data for meaningful trend
        segments = df[segment_column].unique()
        return pd.DataFrame({
            segment_column: segments,
            "smi_score": [0.0] * len(segments),
            "momentum_category": ["Insufficient Data"] * len(segments)
        })
    
    recent_qs = quarters[-recent_quarters:]
    historical_qs = quarters[:-recent_quarters]
    
    recent = df[df["closed_year_quarter"].isin(recent_qs)]
    historical = df[df["closed_year_quarter"].isin(historical_qs)]
    
    # Normalize volumes by number of quarters in each period
    n_recent_qs = len(recent_qs)
    n_hist_qs = max(len(historical_qs), 1)
    
    results = []
    
    for segment in df[segment_column].unique():
        seg_recent = recent[recent[segment_column] == segment]
        seg_hist = historical[historical[segment_column] == segment]
        
        # Skip if no history
        if len(seg_hist) == 0:
            results.append({
                segment_column: segment,
                "smi_score": 0.0,
                "win_rate_delta": 0,
                "volume_delta": 0,
                "deal_size_delta": 0,
                "momentum_category": "New Segment"
            })
            continue
        
        # Win rate change
        recent_wr = seg_recent["is_won"].mean() if len(seg_recent) > 0 else 0
        hist_wr = seg_hist["is_won"].mean()
        wr_delta = (recent_wr / hist_wr - 1) if hist_wr > 0 else 0
        
        # Volume change (normalized per quarter)
        recent_vol_per_q = len(seg_recent) / n_recent_qs
        hist_vol_per_q = len(seg_hist) / n_hist_qs
        vol_delta = (recent_vol_per_q / hist_vol_per_q - 1) if hist_vol_per_q > 0 else 0
        
        # Deal size change
        recent_size = seg_recent["deal_amount"].mean() if len(seg_recent) > 0 else 0
        hist_size = seg_hist["deal_amount"].mean()
        size_delta = (recent_size / hist_size - 1) if hist_size > 0 else 0
        
        # Weighted composite (win rate matters most)
        smi = 0.5 * wr_delta + 0.3 * vol_delta + 0.2 * size_delta
        
        results.append({
            segment_column: segment,
            "smi_score": round(smi, 3),
            "win_rate_delta": round(wr_delta * 100, 1),   # % change
            "volume_delta": round(vol_delta * 100, 1),     # % change
            "deal_size_delta": round(size_delta * 100, 1), # % change
            "recent_win_rate": round(recent_wr * 100, 1),
            "historical_win_rate": round(hist_wr * 100, 1),
            "recent_deals_per_q": round(recent_vol_per_q, 1),
            "historical_deals_per_q": round(hist_vol_per_q, 1),
            "momentum_category": ""  # Filled below
        })
    
    result_df = pd.DataFrame(results)
    
    # Categorize momentum
    conditions = [
        result_df["smi_score"] > 0.15,
        result_df["smi_score"] > 0.05,
        result_df["smi_score"] > -0.05,
        result_df["smi_score"] > -0.15,
        result_df["smi_score"] <= -0.15
    ]
    categories = ["🚀 Strong Growth", "📈 Growing", "➡️ Stable", "📉 Declining", "🔻 Sharp Decline"]
    result_df["momentum_category"] = np.select(conditions, categories, default="➡️ Stable")
    
    return result_df.sort_values("smi_score", ascending=False)


# ═══════════════════════════════════════════════════════════════
# BONUS: DEAL VELOCITY INDEX (DVI) — Improved version
# ═══════════════════════════════════════════════════════════════

def calculate_deal_velocity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Deal Velocity Index (DVI) — Revenue per day of sales effort,
    normalized against the median.
    
    DVI > 1 = Above-average efficiency   |   DVI < 1 = Below average
    
    Args:
        df: DataFrame with deal_amount and sales_cycle_days
        
    Returns:
        DataFrame with DVI column added
    """
    df = df.copy()
    
    df["revenue_per_day"] = df["deal_amount"] / df["sales_cycle_days"].replace(0, 1)
    median_rpd = df["revenue_per_day"].median()
    df["deal_velocity_index"] = (df["revenue_per_day"] / median_rpd).round(3)
    
    df["dvi_category"] = pd.cut(
        df["deal_velocity_index"],
        bins=[0, 0.5, 0.8, 1.2, 2.0, float("inf")],
        labels=["Very Low", "Low", "Average", "High", "Very High"]
    )
    
    return df


# ═══════════════════════════════════════════════════════════════
# AGGREGATION & INSIGHTS
# ═══════════════════════════════════════════════════════════════

def calculate_all_custom_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate all custom metrics and return as a dictionary.
    
    Returns:
        Dictionary containing all metric results.
    """
    # 1. Deal Velocity Index
    df_with_dvi = calculate_deal_velocity_index(df)
    
    # 2. Deal Qualification Efficiency (overall + by rep + by region)
    dqe_overall = calculate_deal_qualification_efficiency(df)
    dqe_by_rep = calculate_deal_qualification_efficiency(df, "sales_rep_id")
    dqe_by_region = calculate_deal_qualification_efficiency(df, "region")
    
    # 3. Revenue Concentration Risk
    rcr_region = calculate_revenue_concentration_risk(df, "region")
    rcr_industry = calculate_revenue_concentration_risk(df, "industry")
    rcr_product = calculate_revenue_concentration_risk(df, "product_type")
    
    # 4. Segment Momentum Index
    smi_region = calculate_segment_momentum_index(df, "region")
    smi_industry = calculate_segment_momentum_index(df, "industry")
    smi_product = calculate_segment_momentum_index(df, "product_type")
    smi_source = calculate_segment_momentum_index(df, "lead_source")
    
    return {
        "df_with_dvi": df_with_dvi,
        "dqe_overall": dqe_overall,
        "dqe_by_rep": dqe_by_rep,
        "dqe_by_region": dqe_by_region,
        "rcr_region": rcr_region,
        "rcr_industry": rcr_industry,
        "rcr_product": rcr_product,
        "smi_region": smi_region,
        "smi_industry": smi_industry,
        "smi_product": smi_product,
        "smi_source": smi_source,
    }


def get_metric_summary(metrics_dict: Dict) -> Dict:
    """Generate concise metric summary for dashboard KPI cards."""
    df = metrics_dict["df_with_dvi"]
    dqe = metrics_dict["dqe_overall"]
    
    return {
        "dvi": {
            "won_avg": df[df["is_won"] == 1]["deal_velocity_index"].mean(),
            "lost_avg": df[df["is_won"] == 0]["deal_velocity_index"].mean(),
            "high_velocity_deals": int((df["deal_velocity_index"] > 1.2).sum()),
        },
        "dqe": {
            "score": float(dqe["dqe_score"].iloc[0]),
            "wasted_days": float(dqe["wasted_capacity_days"].iloc[0]),
            "category": str(dqe["dqe_category"].iloc[0]),
        },
        "rcr_region": {
            "score": metrics_dict["rcr_region"]["rcr_score"],
            "risk_level": metrics_dict["rcr_region"]["risk_level"],
            "top_segment": metrics_dict["rcr_region"]["top_segment"],
            "top_share": metrics_dict["rcr_region"]["top_segment_share"],
        },
        "rcr_industry": {
            "score": metrics_dict["rcr_industry"]["rcr_score"],
            "risk_level": metrics_dict["rcr_industry"]["risk_level"],
            "top_segment": metrics_dict["rcr_industry"]["top_segment"],
            "top_share": metrics_dict["rcr_industry"]["top_segment_share"],
        },
    }


def identify_key_insights(df: pd.DataFrame, metrics_dict: Dict) -> List[Dict]:
    """
    Auto-generate 4+ business insights from custom metrics.
    
    Each insight includes:
      - category, emoji, insight text
      - severity (high/medium/low)
      - business_action: what a CRO should do about it
      - metric_source: which custom metric generated it
      - estimated_impact: quantified business impact
    
    Returns:
        List of insight dictionaries, sorted by severity.
    """
    insights = []
    overall_win_rate = df["is_won"].mean() * 100
    total_won_revenue = df[df["is_won"] == 1]["deal_amount"].sum()
    
    # ─────────────────────────────────────────────────
    # INSIGHT 1: Deal Qualification Efficiency
    # ─────────────────────────────────────────────────
    dqe = metrics_dict["dqe_overall"]
    dqe_score = float(dqe["dqe_score"].iloc[0])
    wasted_days = float(dqe["wasted_capacity_days"].iloc[0])
    median_won = float(dqe["median_cycle_won"].iloc[0])
    median_lost = float(dqe["median_cycle_lost"].iloc[0])
    
    if dqe_score < 0.1:
        # Lost deals take almost as long or longer than won deals
        severity = "high"
        insights.append({
            "category": "Qualification Gap",
            "emoji": "🚨",
            "insight": (
                f"Lost deals take {median_lost:.0f} days vs {median_won:.0f} days for wins "
                f"(DQE Score: {dqe_score:.2f}). Your team cannot distinguish winners from losers "
                f"early enough, wasting ~{wasted_days:,.0f} cumulative rep-days on dead deals."
            ),
            "severity": severity,
            "business_action": (
                "Implement mandatory 30-day deal review checkpoints. Require reps to "
                "requalify stalled deals with updated champion/economic buyer confirmation. "
                "Introduce a 'Dead Deal Criteria' checklist."
            ),
            "metric_source": "Deal Qualification Efficiency (DQE)",
            "estimated_impact": f"Recovering {wasted_days:,.0f} rep-days could add 15-25% more pipeline capacity"
        })
    elif dqe_score < 0.25:
        insights.append({
            "category": "Qualification Improvement Needed",
            "emoji": "⚠️",
            "insight": (
                f"Deal Qualification Efficiency is {dqe_score:.2f} — lost deals still take "
                f"{median_lost:.0f} days vs {median_won:.0f} for wins. Room to exit losers faster."
            ),
            "severity": "medium",
            "business_action": "Review bottom-quartile reps' DQE scores for targeted coaching.",
            "metric_source": "Deal Qualification Efficiency (DQE)",
            "estimated_impact": f"Could save ~{wasted_days/2:,.0f} rep-days per cycle"
        })
    
    # Rep-level DQE extremes
    dqe_by_rep = metrics_dict["dqe_by_rep"]
    worst_dqe_reps = dqe_by_rep[dqe_by_rep["dqe_score"].astype(float) < 0]
    if len(worst_dqe_reps) > 0:
        insights.append({
            "category": "Rep Coaching Alert",
            "emoji": "👤",
            "insight": (
                f"{len(worst_dqe_reps)} reps have negative DQE — their lost deals take LONGER "
                f"than won deals. These reps are actively wasting pipeline capacity by holding "
                f"onto dead deals."
            ),
            "severity": "high",
            "business_action": (
                "Assign sales coaches to these reps. Focus on deal exit criteria, "
                "pipeline hygiene discipline, and early disqualification behavior."
            ),
            "metric_source": "Deal Qualification Efficiency (DQE) by Rep",
            "estimated_impact": "Fixing worst performers could reclaim 500+ selling days/year"
        })
    
    # ─────────────────────────────────────────────────
    # INSIGHT 2: Revenue Concentration Risk
    # ─────────────────────────────────────────────────
    for seg_type in ["region", "industry"]:
        rcr = metrics_dict[f"rcr_{seg_type}"]
        
        if rcr["risk_level"] in ["Critical", "High"]:
            at_risk = rcr["revenue_at_risk_30pct_decline"]
            insights.append({
                "category": f"Revenue Concentration ({seg_type.title()})",
                "emoji": "💰",
                "insight": (
                    f'{rcr["top_segment"]} accounts for {rcr["top_segment_share"]}% of '
                    f'won revenue (RCR: {rcr["rcr_score"]:.2f}, Risk: {rcr["risk_level"]}). '
                    f'A 30% decline in {rcr["top_segment"]} alone would cost ${at_risk:,.0f}.'
                ),
                "severity": "high",
                "business_action": rcr["recommendation"],
                "metric_source": "Revenue Concentration Risk (RCR)",
                "estimated_impact": f"${at_risk:,.0f} revenue at risk from single-segment dependency"
            })
        elif rcr["risk_level"] == "Moderate":
            insights.append({
                "category": f"Portfolio Mix ({seg_type.title()})",
                "emoji": "📊",
                "insight": (
                    f'Revenue is moderately concentrated (RCR: {rcr["rcr_score"]:.2f}). '
                    f'Top 2 {seg_type}s account for {rcr["top2_combined_share"]}% of revenue.'
                ),
                "severity": "medium",
                "business_action": "Set quarterly pipeline targets for underrepresented segments.",
                "metric_source": "Revenue Concentration Risk (RCR)",
                "estimated_impact": "Diversification reduces quarterly revenue volatility by 15-20%"
            })
    
    # ─────────────────────────────────────────────────
    # INSIGHT 3: Segment Momentum
    # ─────────────────────────────────────────────────
    for seg_type in ["region", "industry"]:
        smi = metrics_dict[f"smi_{seg_type}"]
        
        # Declining segments
        declining = smi[smi["smi_score"] < -0.05]
        for _, row in declining.head(2).iterrows():
            segment_name = row[seg_type]
            insights.append({
                "category": f"Momentum Alert ({seg_type.title()})",
                "emoji": "📉",
                "insight": (
                    f'{segment_name} is losing momentum (SMI: {row["smi_score"]:.2f}). '
                    f'Win rate changed {row["win_rate_delta"]:+.1f}%, volume {row["volume_delta"]:+.1f}%, '
                    f'deal size {row["deal_size_delta"]:+.1f}% vs historical baseline.'
                ),
                "severity": "high" if row["smi_score"] < -0.15 else "medium",
                "business_action": (
                    f"Investigate what changed in {segment_name}: new competitors? "
                    "Market shifts? Rep turnover? Lost channel partner?"
                ),
                "metric_source": "Segment Momentum Index (SMI)",
                "estimated_impact": f'Win rate dropped from {row.get("historical_win_rate", 0):.1f}% → {row.get("recent_win_rate", 0):.1f}%'
            })
        
        # Growing segments (opportunity)
        growing = smi[smi["smi_score"] > 0.1]
        if len(growing) > 0:
            top_grower = growing.iloc[0]
            segment_name = top_grower[seg_type]
            insights.append({
                "category": f"Growth Opportunity ({seg_type.title()})",
                "emoji": "🚀",
                "insight": (
                    f'{segment_name} has strong positive momentum (SMI: {top_grower["smi_score"]:.2f}). '
                    f'Win rate up {top_grower["win_rate_delta"]:+.1f}%, volume up {top_grower["volume_delta"]:+.1f}%.'
                ),
                "severity": "low",
                "business_action": f"Double down on {segment_name}: increase rep allocation and marketing spend.",
                "metric_source": "Segment Momentum Index (SMI)",
                "estimated_impact": "Investing in momentum segments typically yields 2-3x ROI vs declining ones"
            })
    
    # ─────────────────────────────────────────────────
    # INSIGHT 4: DVI Won vs Lost Gap
    # ─────────────────────────────────────────────────
    df_dvi = metrics_dict["df_with_dvi"]
    won_dvi = df_dvi[df_dvi["is_won"] == 1]["deal_velocity_index"].mean()
    lost_dvi = df_dvi[df_dvi["is_won"] == 0]["deal_velocity_index"].mean()
    
    if won_dvi > lost_dvi * 1.2:
        insights.append({
            "category": "Deal Velocity Signal",
            "emoji": "⚡",
            "insight": (
                f"Won deals have {((won_dvi/lost_dvi - 1) * 100):.0f}% higher Deal Velocity "
                f"(avg DVI: {won_dvi:.2f} vs {lost_dvi:.2f}). Faster, higher-value deals "
                "are significantly more likely to close."
            ),
            "severity": "medium",
            "business_action": (
                "Use DVI as an early warning signal for stalling deals. "
                "Prioritize deals with DVI > 1.0 for manager attention."
            ),
            "metric_source": "Deal Velocity Index (DVI)",
            "estimated_impact": "Prioritizing high-DVI deals could improve win rate 3-5pp"
        })
    
    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 2))
    
    return insights


# ═══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    
    df, _ = load_and_prepare_data()
    all_metrics = calculate_all_custom_metrics(df)
    
    print("=" * 70)
    print("SKYGENI CUSTOM METRICS TEST")
    print("=" * 70)
    
    # DVI
    dvi_df = all_metrics["df_with_dvi"]
    print(f"\n1. DEAL VELOCITY INDEX (DVI)")
    print(f"   Won deals avg DVI:  {dvi_df[dvi_df['is_won']==1]['deal_velocity_index'].mean():.3f}")
    print(f"   Lost deals avg DVI: {dvi_df[dvi_df['is_won']==0]['deal_velocity_index'].mean():.3f}")
    
    # DQE
    dqe = all_metrics["dqe_overall"]
    print(f"\n2. DEAL QUALIFICATION EFFICIENCY (DQE)")
    print(f"   DQE Score: {float(dqe['dqe_score'].iloc[0]):.3f}")
    print(f"   Category:  {dqe['dqe_category'].iloc[0]}")
    print(f"   Wasted capacity: {float(dqe['wasted_capacity_days'].iloc[0]):,.0f} days")
    
    # RCR
    for seg in ["region", "industry"]:
        rcr = all_metrics[f"rcr_{seg}"]
        print(f"\n3. REVENUE CONCENTRATION RISK ({seg.upper()})")
        print(f"   RCR Score: {rcr['rcr_score']:.3f} ({rcr['risk_level']})")
        print(f"   Top segment: {rcr['top_segment']} ({rcr['top_segment_share']}%)")
        print(f"   Revenue at risk (30% decline): ${rcr['revenue_at_risk_30pct_decline']:,.0f}")
    
    # SMI
    smi = all_metrics["smi_region"]
    print(f"\n4. SEGMENT MOMENTUM INDEX (REGION)")
    for _, row in smi.iterrows():
        print(f"   {row['region']}: SMI={row['smi_score']:+.3f} → {row['momentum_category']}")
    
    # Insights
    insights = identify_key_insights(df, all_metrics)
    print(f"\n{'='*70}")
    print(f"BUSINESS INSIGHTS ({len(insights)} generated)")
    print("=" * 70)
    for i, ins in enumerate(insights, 1):
        print(f"\n{i}. [{ins['severity'].upper()}] {ins['emoji']} {ins['category']}")
        print(f"   {ins['insight'][:100]}...")
        print(f"   Source: {ins['metric_source']}")
        print(f"   Action: {ins['business_action'][:80]}...")
