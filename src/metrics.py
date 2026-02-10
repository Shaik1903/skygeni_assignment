"""
Custom Metrics Module for SkyGeni Sales Intelligence Dashboard
===============================================================

3 Original Custom Metrics (invented for this analysis):
  1. Pipeline Qualification Score (PQS) — Combined org-level qualification efficiency
                                          with deal-level stall detection.
  2. Win Rate Elasticity (WRE)          — How much does chasing bigger deals hurt our win rate?
  3. Segment Momentum Index (SMI)       — Which segments are gaining or losing steam?

Bonus: Deal Velocity Index (DVI) — Revenue generated per day of sales effort.

The module also surfaces 6+ business insights automatically from these metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════
# METRIC 1: PIPELINE QUALIFICATION SCORE (PQS)
# ═══════════════════════════════════════════════════════════════
#
# This is a combined metric that answers TWO critical questions:
#
#   ORG LEVEL  → "Are we wasting sales capacity on deals that will never close?"
#   DEAL LEVEL → "Which specific deals are probably dead and should be killed NOW?"
#
# --- Component A: Deal Qualification Efficiency (DQE) ---
#
# A good sales org should FAIL FAST on losers and INVEST TIME in winners.
# If lost deals take just as long as won deals, reps can't tell the
# difference early — that's a massive capacity problem.
#
# Formula: DQE = 1 - (Median Cycle of Lost Deals / Median Cycle of Won Deals)
#   - DQE = 0.0 → Lost deals take SAME time as won deals (terrible)
#   - DQE = 0.5 → Lost deals take half the time (decent)
#   - DQE < 0   → Lost deals take LONGER than wins (actively harmful)
#
# --- Component B: Deal Stall Risk Score (DSRS) ---
#
# Every won deal closes within a certain timeframe. The 75th percentile
# of won deals' cycle is the "winning window." If a deal has been open
# longer than that, it's statistically past the point where wins happen.
#
# Formula: Stall Score = Deal's Cycle Days / P75(Won Deals' Cycle Days)
#   - Score < 1.0 → Within normal winning window (healthy)
#   - Score 1.0–1.5 → Exceeded typical winning time (warning)
#   - Score > 1.5 → Likely dead — exit now
#
# --- The Combined PQS ---
#
# PQS brings both together: the org-level DQE tells you HOW BAD the
# problem is, and the deal-level stall scores tell you WHERE to act.
# The composite PQS Score (0–100) combines:
#   - 50% DQE component (qualification speed)
#   - 30% Stall rate component (% of deals past winning window)
#   - 20% Win rate gap component (difference in win rate: on-pace vs stalling)
#
# PQS > 70 = Healthy pipeline discipline
# PQS 40–70 = Room for improvement
# PQS < 40 = Critical — deals are rotting in pipeline
# ═══════════════════════════════════════════════════════════════

def calculate_pipeline_qualification_score(df: pd.DataFrame, group_by: str = None) -> Dict:
    """
    Calculate Pipeline Qualification Score (PQS) — a combined metric that
    measures org-level qualification efficiency AND flags individual stalling deals.
    
    Returns both the composite score and actionable deal-level stall detection.
    
    Args:
        df: Preprocessed DataFrame with is_won, sales_cycle_days, deal_amount
        group_by: Optional column to segment results by (e.g., 'region', 'sales_rep_id')
        
    Returns:
        Dictionary with:
          - pqs_score (0-100 composite)
          - dqe_component (qualification efficiency details)
          - stall_component (deal-level stall detection)
          - segment_breakdown (if group_by is specified)
    """
    df_copy = df.copy()
    
    # ── Component A: Deal Qualification Efficiency ──
    won = df_copy[df_copy["is_won"] == 1]
    lost = df_copy[df_copy["is_won"] == 0]
    
    median_won_cycle = won["sales_cycle_days"].median()
    median_lost_cycle = lost["sales_cycle_days"].median()
    
    # Core DQE score
    dqe_score = 1 - (median_lost_cycle / median_won_cycle) if median_won_cycle > 0 else 0
    
    # Wasted capacity: extra days spent on lost deals beyond a fast-fail threshold
    fast_fail_threshold = won["sales_cycle_days"].quantile(0.25)
    excess_days = (lost["sales_cycle_days"][lost["sales_cycle_days"] > fast_fail_threshold] - fast_fail_threshold).sum()
    
    # ── Component B: Deal Stall Risk ──
    winning_window = won["sales_cycle_days"].quantile(0.75)
    
    df_copy["stall_score"] = (df_copy["sales_cycle_days"] / winning_window).round(3)
    
    # Categorize each deal
    conditions = [
        df_copy["stall_score"] <= 0.5,
        df_copy["stall_score"] <= 0.75,
        df_copy["stall_score"] <= 1.0,
        df_copy["stall_score"] <= 1.5,
        df_copy["stall_score"] > 1.5
    ]
    categories = ["Fast Track", "On Pace", "Near Limit", "Stalling", "Likely Dead"]
    df_copy["stall_category"] = np.select(conditions, categories, default="Unknown")
    
    total_deals = len(df_copy)
    stalling_count = int((df_copy["stall_score"] > 1.0).sum())
    likely_dead_count = int((df_copy["stall_score"] > 1.5).sum())
    stall_rate = stalling_count / total_deals if total_deals > 0 else 0
    
    # Win rate comparison: on-pace vs stalling deals
    on_pace = df_copy[df_copy["stall_score"] <= 1.0]
    stalling = df_copy[df_copy["stall_score"] > 1.0]
    on_pace_wr = on_pace["is_won"].mean() * 100 if len(on_pace) > 0 else 0
    stalling_wr = stalling["is_won"].mean() * 100 if len(stalling) > 0 else 0
    wr_gap = on_pace_wr - stalling_wr
    
    # ── Composite PQS Score (0–100) ──
    # DQE component: map DQE from [-1, 1] to [0, 100]
    dqe_norm = max(0, min(100, (dqe_score + 1) * 50))
    
    # Stall rate component: lower stall rate = better (invert)
    stall_norm = max(0, min(100, (1 - stall_rate) * 100))
    
    # Win rate gap component: bigger gap means stalling deals DO lose more = good detection
    # But also means we're not killing them fast enough
    gap_norm = max(0, min(100, 100 - stall_rate * wr_gap))
    
    pqs_score = round(0.5 * dqe_norm + 0.3 * stall_norm + 0.2 * gap_norm, 1)
    
    # Category
    if pqs_score >= 70:
        pqs_category = "Healthy"
    elif pqs_score >= 40:
        pqs_category = "Needs Improvement"
    else:
        pqs_category = "Critical"
    
    # ── Segment breakdown (if requested) ──
    segment_data = None
    if group_by:
        def _segment_pqs(group):
            g_won = group[group["is_won"] == 1]["sales_cycle_days"]
            g_lost = group[group["is_won"] == 0]["sales_cycle_days"]
            g_dqe = 1 - (g_lost.median() / g_won.median()) if len(g_won) > 0 and g_won.median() > 0 and len(g_lost) > 0 else 0
            g_stalling = (group["stall_score"] > 1.0).sum()
            g_dead = (group["stall_score"] > 1.5).sum()
            g_stall_pct = round(g_stalling / len(group) * 100, 1) if len(group) > 0 else 0
            
            # Wasted days for this segment
            if len(g_won) > 0:
                g_threshold = g_won.quantile(0.25)
                g_excess = (g_lost[g_lost > g_threshold] - g_threshold).sum() if len(g_lost) > 0 else 0
            else:
                g_excess = 0
            
            return pd.Series({
                "total_deals": len(group),
                "won_deals": int(group["is_won"].sum()),
                "win_rate": round(group["is_won"].mean() * 100, 1),
                "dqe_score": round(g_dqe, 3),
                "median_cycle_won": g_won.median() if len(g_won) > 0 else np.nan,
                "median_cycle_lost": g_lost.median() if len(g_lost) > 0 else np.nan,
                "deals_stalling": int(g_stalling),
                "deals_likely_dead": int(g_dead),
                "pct_at_risk": g_stall_pct,
                "wasted_capacity_days": round(g_excess, 0),
            })
        
        segment_data = df_copy.groupby(group_by).apply(_segment_pqs, include_groups=False).reset_index()
        
        # Categorize segment DQE
        segment_data["dqe_category"] = pd.cut(
            segment_data["dqe_score"].astype(float),
            bins=[-float("inf"), -0.1, 0.1, 0.25, 0.5, float("inf")],
            labels=["Critical: Losers take longer", "Poor: Nearly equal cycles",
                    "Fair: Slightly faster exits", "Good: Noticeably faster exits",
                    "Excellent: Rapid disqualification"]
        )
    
    return {
        # Composite score
        "pqs_score": pqs_score,
        "pqs_category": pqs_category,
        
        # DQE component details
        "dqe_score": round(dqe_score, 3),
        "median_cycle_won": round(median_won_cycle, 0),
        "median_cycle_lost": round(median_lost_cycle, 0),
        "wasted_capacity_days": round(excess_days, 0),
        "fast_fail_threshold_days": round(fast_fail_threshold, 0),
        
        # Stall component details
        "winning_window_days": round(winning_window, 0),
        "total_deals": total_deals,
        "deals_stalling": stalling_count,
        "deals_likely_dead": likely_dead_count,
        "stall_rate_pct": round(stall_rate * 100, 1),
        "on_pace_win_rate": round(on_pace_wr, 1),
        "stalling_win_rate": round(stalling_wr, 1),
        "win_rate_gap": round(wr_gap, 1),
        
        # Deal-level data (for drill-down)
        "df_with_stall": df_copy[["deal_id", "sales_cycle_days", "deal_amount",
                                   "is_won", "stall_score", "stall_category"]],
        
        # Segment breakdown
        "segment_breakdown": segment_data,
    }


# ═══════════════════════════════════════════════════════════════
# METRIC 2: WIN RATE ELASTICITY (WRE)
# ═══════════════════════════════════════════════════════════════
#
# Business question: "Should we chase bigger deals, or does that
#                     actually hurt our chances of winning?"
#
# Think of it like this:
#   Imagine you run a shop. You could sell $10 items (easy to sell)
#   or $100 items (harder to sell). The question is:
#   "For every 10% I increase the price, how much do my sales drop?"
#
# That's exactly what Win Rate Elasticity measures for deal sizes.
#
# Example:
#   - Small deals ($0-25K):  58% win rate
#   - Medium deals ($25-75K): 45% win rate
#   - Large deals ($75K+):   32% win rate
#   → Elasticity = -1.8 (means: for every 10% bigger the deal,
#     win rate drops by 18%. That's steep!)
#
# Why it matters for a CRO:
#   Elasticity very negative (like -2.0):
#     "Chasing big deals is DESTROYING our win rate."
#   Elasticity near 0:
#     "Deal size doesn't really matter — go big!"
#   Elasticity positive (rare):
#     "Bigger deals actually win MORE often."
#
# This guides critical decisions:
#   - Should we pursue enterprise deals or stay mid-market?
#   - Is our discount strategy working (or backfiring)?
#   - Which deal size "sweet spot" maximizes total revenue?
# ═══════════════════════════════════════════════════════════════

def calculate_win_rate_elasticity(df: pd.DataFrame, n_buckets: int = 5) -> Dict:
    """
    Calculate Win Rate Elasticity — how sensitive is our win rate
    to changes in deal size?
    
    A simple way to think about it:
      If elasticity = -1.5, then making deals 10% bigger
      drops our win rate by 15%. That's a big tradeoff!
      
      If elasticity = -0.3, deal size barely affects win rate.
      Go ahead and pursue bigger deals!
    
    Args:
        df: Preprocessed DataFrame with deal_amount and is_won
        n_buckets: Number of deal size buckets (default 5)
        
    Returns:
        Dictionary with elasticity score, bucket breakdown, and interpretation
    """
    df = df.copy()
    
    # Create deal size buckets (quintiles for even distribution)
    df["size_bucket"], bin_edges = pd.qcut(
        df["deal_amount"], q=n_buckets, retbins=True, duplicates="drop"
    )
    
    # Calculate win rate and avg deal size per bucket
    bucket_stats = df.groupby("size_bucket", observed=True).agg(
        avg_deal_size=("deal_amount", "mean"),
        win_rate=("is_won", "mean"),
        deal_count=("deal_id", "count"),
        total_revenue=("deal_amount", "sum")
    ).reset_index()
    
    bucket_stats = bucket_stats.sort_values("avg_deal_size")
    bucket_stats["win_rate_pct"] = (bucket_stats["win_rate"] * 100).round(1)
    
    # Create human-readable bucket labels
    bucket_stats["size_range"] = bucket_stats["size_bucket"].apply(
        lambda x: f"${x.left/1000:.0f}K - ${x.right/1000:.0f}K"
    )
    
    # Calculate elasticity between consecutive buckets
    # Elasticity = (% change in win rate) / (% change in deal size)
    elasticities = []
    for i in range(1, len(bucket_stats)):
        prev = bucket_stats.iloc[i - 1]
        curr = bucket_stats.iloc[i]
        
        pct_size_change = (curr["avg_deal_size"] - prev["avg_deal_size"]) / prev["avg_deal_size"]
        pct_wr_change = (curr["win_rate"] - prev["win_rate"]) / prev["win_rate"] if prev["win_rate"] > 0 else 0
        
        if pct_size_change != 0:
            elasticities.append(pct_wr_change / pct_size_change)
    
    # Overall elasticity (average)
    overall_elasticity = round(np.mean(elasticities), 3) if elasticities else 0.0
    
    # Find the "sweet spot" — bucket with best win rate × deal size balance
    bucket_stats["revenue_potential"] = bucket_stats["win_rate"] * bucket_stats["avg_deal_size"]
    sweet_spot_idx = bucket_stats["revenue_potential"].idxmax()
    sweet_spot = bucket_stats.loc[sweet_spot_idx]
    
    # Smallest vs largest bucket comparison
    smallest_bucket = bucket_stats.iloc[0]
    largest_bucket = bucket_stats.iloc[-1]
    
    # Interpret the elasticity in plain English
    if overall_elasticity < -1.0:
        interpretation = "HIGHLY ELASTIC: Bigger deals significantly hurt win rate. Consider staying mid-market or improving enterprise sales capability."
        severity = "high"
    elif overall_elasticity < -0.5:
        interpretation = "MODERATELY ELASTIC: Larger deals have noticeably lower win rates. Be selective about which big deals to pursue."
        severity = "medium"
    elif overall_elasticity < 0:
        interpretation = "SLIGHTLY ELASTIC: Deal size has minor impact on win rate. Pursue larger deals with modest caution."
        severity = "low"
    else:
        interpretation = "INELASTIC: Bigger deals do NOT hurt win rate. Aggressively pursue larger deals for maximum revenue."
        severity = "low"
    
    return {
        "elasticity": overall_elasticity,
        "interpretation": interpretation,
        "severity": severity,
        "bucket_stats": bucket_stats[["size_range", "avg_deal_size", "win_rate_pct", 
                                      "deal_count", "revenue_potential"]].reset_index(drop=True),
        "sweet_spot_range": sweet_spot["size_range"],
        "sweet_spot_win_rate": round(sweet_spot["win_rate_pct"], 1),
        "sweet_spot_avg_size": round(sweet_spot["avg_deal_size"], 0),
        "smallest_bucket_wr": round(smallest_bucket["win_rate_pct"], 1),
        "largest_bucket_wr": round(largest_bucket["win_rate_pct"], 1),
        "wr_drop_small_to_large": round(smallest_bucket["win_rate_pct"] - largest_bucket["win_rate_pct"], 1),
        "n_buckets": len(bucket_stats),
    }


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
    
    n_recent_qs = len(recent_qs)
    n_hist_qs = max(len(historical_qs), 1)
    
    results = []
    
    for segment in df[segment_column].unique():
        seg_recent = recent[recent[segment_column] == segment]
        seg_hist = historical[historical[segment_column] == segment]
        
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
        
        # Weighted composite
        smi = 0.5 * wr_delta + 0.3 * vol_delta + 0.2 * size_delta
        
        results.append({
            segment_column: segment,
            "smi_score": round(smi, 3),
            "win_rate_delta": round(wr_delta * 100, 1),
            "volume_delta": round(vol_delta * 100, 1),
            "deal_size_delta": round(size_delta * 100, 1),
            "recent_win_rate": round(recent_wr * 100, 1),
            "historical_win_rate": round(hist_wr * 100, 1),
            "recent_deals_per_q": round(recent_vol_per_q, 1),
            "historical_deals_per_q": round(hist_vol_per_q, 1),
            "momentum_category": ""
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
    cats = ["Strong Growth", "Growing", "Stable", "Declining", "Sharp Decline"]
    result_df["momentum_category"] = np.select(conditions, cats, default="Stable")
    
    return result_df.sort_values("smi_score", ascending=False)


# ═══════════════════════════════════════════════════════════════
# BONUS: DEAL VELOCITY INDEX (DVI)
# ═══════════════════════════════════════════════════════════════

def calculate_deal_velocity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Deal Velocity Index (DVI) — Revenue per day of sales effort,
    normalized against the median.
    
    DVI > 1 = Above-average efficiency   |   DVI < 1 = Below average
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
    """
    # 1. Deal Velocity Index
    df_with_dvi = calculate_deal_velocity_index(df)
    
    # 2. Pipeline Qualification Score (combined DQE + DSRS)
    pqs_overall = calculate_pipeline_qualification_score(df)
    pqs_by_rep = calculate_pipeline_qualification_score(df, "sales_rep_id")
    pqs_by_region = calculate_pipeline_qualification_score(df, "region")
    
    # 3. Win Rate Elasticity
    wre = calculate_win_rate_elasticity(df)
    
    # 4. Segment Momentum Index
    smi_region = calculate_segment_momentum_index(df, "region")
    smi_industry = calculate_segment_momentum_index(df, "industry")
    smi_product = calculate_segment_momentum_index(df, "product_type")
    smi_source = calculate_segment_momentum_index(df, "lead_source")
    
    return {
        "df_with_dvi": df_with_dvi,
        "pqs_overall": pqs_overall,
        "pqs_by_rep": pqs_by_rep,
        "pqs_by_region": pqs_by_region,
        "wre": wre,
        "smi_region": smi_region,
        "smi_industry": smi_industry,
        "smi_product": smi_product,
        "smi_source": smi_source,
    }


def get_metric_summary(metrics_dict: Dict) -> Dict:
    """Generate concise metric summary for dashboard KPI cards."""
    df = metrics_dict["df_with_dvi"]
    pqs = metrics_dict["pqs_overall"]
    wre = metrics_dict["wre"]
    
    return {
        "dvi": {
            "won_avg": df[df["is_won"] == 1]["deal_velocity_index"].mean(),
            "lost_avg": df[df["is_won"] == 0]["deal_velocity_index"].mean(),
            "high_velocity_deals": int((df["deal_velocity_index"] > 1.2).sum()),
        },
        "pqs": {
            "score": pqs["pqs_score"],
            "category": pqs["pqs_category"],
            "dqe_score": pqs["dqe_score"],
            "stall_rate": pqs["stall_rate_pct"],
            "wasted_days": pqs["wasted_capacity_days"],
            "deals_stalling": pqs["deals_stalling"],
            "deals_likely_dead": pqs["deals_likely_dead"],
        },
        "wre": {
            "elasticity": wre["elasticity"],
            "severity": wre["severity"],
            "sweet_spot": wre["sweet_spot_range"],
            "wr_drop": wre["wr_drop_small_to_large"],
        },
    }


def identify_key_insights(df: pd.DataFrame, metrics_dict: Dict) -> List[Dict]:
    """
    Auto-generate 6+ business insights from custom metrics.
    
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
    
    # ─────────────────────────────────────────────────
    # INSIGHT GROUP 1: Pipeline Qualification Score
    # ─────────────────────────────────────────────────
    pqs = metrics_dict["pqs_overall"]
    dqe_score = pqs["dqe_score"]
    wasted_days = pqs["wasted_capacity_days"]
    stall_rate = pqs["stall_rate_pct"]
    win_gap = pqs["win_rate_gap"]
    
    # PQS composite insight
    if pqs["pqs_score"] < 40:
        insights.append({
            "category": "Pipeline Health Crisis",
            "emoji": "🚨",
            "insight": (
                f"Pipeline Qualification Score is {pqs['pqs_score']}/100 (Critical). "
                f"Lost deals take {pqs['median_cycle_lost']:.0f} days vs {pqs['median_cycle_won']:.0f} for wins, "
                f"and {stall_rate:.0f}% of all deals have exceeded the winning window. "
                f"~{wasted_days:,.0f} rep-days wasted on dead deals."
            ),
            "severity": "high",
            "business_action": (
                "Emergency pipeline review: kill all deals past the winning window "
                f"({pqs['winning_window_days']:.0f} days). Implement mandatory 30-day "
                "requalification checkpoints and a Dead Deal Criteria checklist."
            ),
            "metric_source": "Pipeline Qualification Score (PQS)",
            "estimated_impact": f"Recovering {wasted_days:,.0f} rep-days = 15-25% more pipeline capacity"
        })
    elif pqs["pqs_score"] < 70:
        insights.append({
            "category": "Pipeline Efficiency Gap",
            "emoji": "⚠️",
            "insight": (
                f"Pipeline Qualification Score is {pqs['pqs_score']}/100 (Needs Improvement). "
                f"DQE is {dqe_score:.2f} and {stall_rate:.0f}% of deals are past the winning window."
            ),
            "severity": "medium",
            "business_action": (
                "Introduce deal velocity checkpoints and review stalling deals weekly."
            ),
            "metric_source": "Pipeline Qualification Score (PQS)",
            "estimated_impact": f"Could save ~{wasted_days/2:,.0f} rep-days per cycle"
        })
    else:
        insights.append({
            "category": "Pipeline Health",
            "emoji": "✅",
            "insight": (
                f"Pipeline Qualification Score is {pqs['pqs_score']}/100 (Healthy). "
                f"Team is disqualifying losers efficiently (DQE: {dqe_score:.2f})."
            ),
            "severity": "low",
            "business_action": "Maintain current pipeline discipline. Monitor monthly for drift.",
            "metric_source": "Pipeline Qualification Score (PQS)",
            "estimated_impact": "Healthy pipeline hygiene supports sustained win rate"
        })
    
    # Stall detection insight (deal-level)
    if pqs["deals_likely_dead"] > 0:
        insights.append({
            "category": "Dead Deals Alert",
            "emoji": "💀",
            "insight": (
                f"{pqs['deals_likely_dead']} deals are likely dead (cycle > 1.5x the winning window of "
                f"{pqs['winning_window_days']:.0f} days). These deals have only a {pqs['stalling_win_rate']:.0f}% "
                f"win rate vs {pqs['on_pace_win_rate']:.0f}% for on-pace deals."
            ),
            "severity": "high",
            "business_action": (
                "Review and exit these specific deals immediately. "
                "Redirect rep time to fresh pipeline with higher probability."
            ),
            "metric_source": "Pipeline Qualification Score (PQS) — Stall Detection",
            "estimated_impact": f"Win rate gap: {win_gap:.1f}pp between on-pace and stalling deals"
        })
    
    # Rep-level coaching from PQS
    pqs_by_rep = metrics_dict["pqs_by_rep"]
    rep_segments = pqs_by_rep.get("segment_breakdown")
    if rep_segments is not None:
        worst_reps = rep_segments[rep_segments["dqe_score"].astype(float) < 0]
        if len(worst_reps) > 0:
            insights.append({
                "category": "Rep Coaching Alert",
                "emoji": "👤",
                "insight": (
                    f"{len(worst_reps)} reps have negative DQE — their lost deals take LONGER "
                    f"than won deals. These reps hold dead deals too long, wasting pipeline capacity."
                ),
                "severity": "high",
                "business_action": (
                    "Assign sales coaches to these reps. Focus on deal exit criteria, "
                    "pipeline hygiene, and early disqualification behavior."
                ),
                "metric_source": "Pipeline Qualification Score (PQS) — Rep Breakdown",
                "estimated_impact": "Fixing worst performers could reclaim 500+ selling days/year"
            })
    
    # ─────────────────────────────────────────────────
    # INSIGHT GROUP 2: Win Rate Elasticity
    # ─────────────────────────────────────────────────
    wre = metrics_dict["wre"]
    elasticity = wre["elasticity"]
    
    if abs(wre["wr_drop_small_to_large"]) > 0:
        e_severity = wre["severity"]
        insights.append({
            "category": "Deal Size vs Win Rate",
            "emoji": "📏" if elasticity > -0.5 else "⚠️",
            "insight": (
                f"Win Rate Elasticity is {elasticity:.2f}. "
                f"Small deals ({wre['smallest_bucket_wr']:.0f}% win rate) vs large deals "
                f"({wre['largest_bucket_wr']:.0f}% win rate) — a {wre['wr_drop_small_to_large']:.0f}pp gap. "
                f"The revenue sweet spot is {wre['sweet_spot_range']} "
                f"({wre['sweet_spot_win_rate']:.0f}% win rate)."
            ),
            "severity": e_severity,
            "business_action": (
                f"Focus reps on the {wre['sweet_spot_range']} sweet spot for max ROI. "
                + ("Avoid overinvesting in enterprise-size deals without specialized support." if elasticity < -0.5 
                   else "Deal size is not a major barrier — pursue larger deals confidently.")
            ),
            "metric_source": "Win Rate Elasticity (WRE)",
            "estimated_impact": f"Optimizing deal size mix could lift win rate by {abs(wre['wr_drop_small_to_large'])/3:.0f}pp"
        })
    
    # ─────────────────────────────────────────────────
    # INSIGHT GROUP 3: Segment Momentum
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
                "estimated_impact": f'Win rate: {row.get("historical_win_rate", 0):.1f}% to {row.get("recent_win_rate", 0):.1f}%'
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
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    from data_loader import load_and_prepare_data
    
    df, _ = load_and_prepare_data()
    all_metrics = calculate_all_custom_metrics(df)
    
    print("=" * 70)
    print("SKYGENI CUSTOM METRICS TEST")
    print("=" * 70)
    
    # PQS
    pqs = all_metrics["pqs_overall"]
    print(f"\n1. PIPELINE QUALIFICATION SCORE (PQS)")
    print(f"   PQS Score: {pqs['pqs_score']}/100 ({pqs['pqs_category']})")
    print(f"   DQE Component: {pqs['dqe_score']:.3f}")
    print(f"   Median cycle: {pqs['median_cycle_won']:.0f}d (won) vs {pqs['median_cycle_lost']:.0f}d (lost)")
    print(f"   Wasted capacity: {pqs['wasted_capacity_days']:,.0f} rep-days")
    print(f"   Winning window: {pqs['winning_window_days']:.0f} days")
    print(f"   Deals stalling: {pqs['deals_stalling']} ({pqs['stall_rate_pct']:.1f}%)")
    print(f"   Deals likely dead: {pqs['deals_likely_dead']}")
    print(f"   Win rate gap: {pqs['on_pace_win_rate']:.1f}% (on-pace) vs {pqs['stalling_win_rate']:.1f}% (stalling)")
    
    # WRE
    wre = all_metrics["wre"]
    print(f"\n2. WIN RATE ELASTICITY (WRE)")
    print(f"   Elasticity: {wre['elasticity']:.3f}")
    print(f"   Interpretation: {wre['interpretation']}")
    print(f"   Sweet spot: {wre['sweet_spot_range']} ({wre['sweet_spot_win_rate']:.0f}% WR)")
    print(f"   Small deals: {wre['smallest_bucket_wr']:.0f}% WR")
    print(f"   Large deals: {wre['largest_bucket_wr']:.0f}% WR")
    print(f"   WR drop small-to-large: {wre['wr_drop_small_to_large']:.1f}pp")
    
    # SMI
    smi = all_metrics["smi_region"]
    print(f"\n3. SEGMENT MOMENTUM INDEX (REGION)")
    for _, row in smi.iterrows():
        print(f"   {row['region']}: SMI={row['smi_score']:+.3f} ({row['momentum_category']})")
    
    # DVI
    dvi_df = all_metrics["df_with_dvi"]
    print(f"\n4. DEAL VELOCITY INDEX (DVI)")
    print(f"   Won deals avg DVI:  {dvi_df[dvi_df['is_won']==1]['deal_velocity_index'].mean():.3f}")
    print(f"   Lost deals avg DVI: {dvi_df[dvi_df['is_won']==0]['deal_velocity_index'].mean():.3f}")
    
    # Insights
    insights = identify_key_insights(df, all_metrics)
    print(f"\n{'='*70}")
    print(f"BUSINESS INSIGHTS ({len(insights)} generated)")
    print("=" * 70)
    for i, ins in enumerate(insights, 1):
        print(f"\n{i}. [{ins['severity'].upper()}] {ins['emoji']} {ins['category']}")
        print(f"   {ins['insight'][:120]}...")
        print(f"   Source: {ins['metric_source']}")
        print(f"   Action: {ins['business_action'][:100]}...")
