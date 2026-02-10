"""
EDA (Exploratory Data Analysis) Module for SkyGeni Sales Intelligence
======================================================================
Comprehensive data analysis functions for generating insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


def analyze_win_rate_trends(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze win rate trends over time to identify the decline pattern.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with trend analysis
    """
    # Quarterly trends
    quarterly = df.groupby("closed_year_quarter").agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum")
    ).reset_index()
    
    quarterly["win_rate"] = (quarterly["won_deals"] / quarterly["total_deals"] * 100).round(2)
    quarterly["win_rate_change"] = quarterly["win_rate"].diff()
    
    # Identify declining quarters
    declining_quarters = quarterly[quarterly["win_rate_change"] < 0]["closed_year_quarter"].tolist()
    
    # Calculate overall trend (regression slope)
    if len(quarterly) > 1:
        x = np.arange(len(quarterly))
        y = quarterly["win_rate"].values
        slope = np.polyfit(x, y, 1)[0]
        trend_direction = "declining" if slope < -0.5 else "stable" if abs(slope) <= 0.5 else "improving"
    else:
        slope = 0
        trend_direction = "insufficient data"
    
    # Peak and current
    peak_quarter = quarterly.loc[quarterly["win_rate"].idxmax(), "closed_year_quarter"]
    peak_rate = quarterly["win_rate"].max()
    current_quarter = quarterly.iloc[-1]["closed_year_quarter"]
    current_rate = quarterly.iloc[-1]["win_rate"]
    
    return {
        "quarterly_data": quarterly.to_dict("records"),
        "declining_quarters": declining_quarters,
        "trend_slope": round(slope, 2),
        "trend_direction": trend_direction,
        "peak_quarter": peak_quarter,
        "peak_rate": peak_rate,
        "current_quarter": current_quarter,
        "current_rate": current_rate,
        "decline_from_peak": round(peak_rate - current_rate, 2)
    }


def analyze_segment_performance(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze performance across all key segments.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with segment analysis DataFrames
    """
    overall_win_rate = df["is_won"].mean() * 100
    
    results = {}
    
    for segment in ["region", "industry", "product_type", "lead_source", "deal_stage"]:
        segment_df = df.groupby(segment).agg(
            total_deals=("deal_id", "count"),
            won_deals=("is_won", "sum"),
            total_revenue=("deal_amount", "sum"),
            avg_deal_size=("deal_amount", "mean"),
            avg_cycle_days=("sales_cycle_days", "mean")
        ).reset_index()
        
        segment_df["win_rate"] = (segment_df["won_deals"] / segment_df["total_deals"] * 100).round(2)
        segment_df["deal_share"] = (segment_df["total_deals"] / len(df) * 100).round(2)
        segment_df["vs_overall"] = (segment_df["win_rate"] - overall_win_rate).round(2)
        
        # Flag underperformers
        segment_df["performance"] = np.where(
            segment_df["win_rate"] > overall_win_rate + 5, "Above Average",
            np.where(segment_df["win_rate"] < overall_win_rate - 5, "Below Average", "Average")
        )
        
        results[segment] = segment_df.sort_values("win_rate", ascending=False)
    
    return results


def analyze_deal_characteristics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Compare characteristics of won vs lost deals.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with deal characteristic analysis
    """
    won = df[df["is_won"] == 1]
    lost = df[df["is_won"] == 0]
    
    # Get deal size category stats (handle categorical)
    size_stats = df.groupby("deal_size_category", observed=True).agg(
        total=("deal_id", "count"),
        won=("is_won", "sum")
    )
    size_stats["win_rate"] = (size_stats["won"] / size_stats["total"] * 100).round(2)
    
    # Get cycle category stats (handle categorical)
    cycle_stats = df.groupby("cycle_category", observed=True).agg(
        total=("deal_id", "count"),
        won=("is_won", "sum")
    )
    cycle_stats["win_rate"] = (cycle_stats["won"] / cycle_stats["total"] * 100).round(2)
    
    analysis = {
        "deal_size": {
            "won_avg": won["deal_amount"].mean(),
            "lost_avg": lost["deal_amount"].mean(),
            "won_median": won["deal_amount"].median(),
            "lost_median": lost["deal_amount"].median(),
            "difference_pct": ((won["deal_amount"].mean() / lost["deal_amount"].mean()) - 1) * 100
        },
        "sales_cycle": {
            "won_avg": won["sales_cycle_days"].mean(),
            "lost_avg": lost["sales_cycle_days"].mean(),
            "won_median": won["sales_cycle_days"].median(),
            "lost_median": lost["sales_cycle_days"].median(),
            "difference_days": won["sales_cycle_days"].mean() - lost["sales_cycle_days"].mean()
        },
        "by_deal_size_category": size_stats.to_dict("index"),
        "by_cycle_category": cycle_stats.to_dict("index")
    }
    
    return analysis


def analyze_lead_source_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deep analysis of lead source performance.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with lead source analysis
    """
    lead_analysis = df.groupby("lead_source").agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_pipeline=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean"),
        avg_cycle_days=("sales_cycle_days", "mean")
    ).reset_index()
    
    lead_analysis["win_rate"] = (lead_analysis["won_deals"] / lead_analysis["total_deals"] * 100).round(2)
    lead_analysis["revenue_won"] = df[df["is_won"] == 1].groupby("lead_source")["deal_amount"].sum().reindex(lead_analysis["lead_source"]).fillna(0).values
    lead_analysis["revenue_efficiency"] = (lead_analysis["revenue_won"] / lead_analysis["total_pipeline"] * 100).round(2)
    
    # Rank
    lead_analysis["win_rate_rank"] = lead_analysis["win_rate"].rank(ascending=False, method="min").astype(int)
    
    return lead_analysis.sort_values("win_rate", ascending=False)


def analyze_sales_rep_deep(df: pd.DataFrame) -> Dict[str, any]:
    """
    Deep analysis of sales rep performance.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with rep analysis
    """
    rep_stats = df.groupby("sales_rep_id").agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean"),
        avg_cycle_days=("sales_cycle_days", "mean")
    ).reset_index()
    
    rep_stats["win_rate"] = (rep_stats["won_deals"] / rep_stats["total_deals"] * 100).round(2)
    rep_stats["revenue_won"] = df[df["is_won"] == 1].groupby("sales_rep_id")["deal_amount"].sum().reindex(rep_stats["sales_rep_id"]).fillna(0).values
    
    overall_win_rate = df["is_won"].mean() * 100
    
    return {
        "rep_summary": rep_stats.sort_values("win_rate", ascending=False),
        "top_performers": rep_stats[rep_stats["win_rate"] > overall_win_rate + 10]["sales_rep_id"].tolist(),
        "underperformers": rep_stats[rep_stats["win_rate"] < overall_win_rate - 10]["sales_rep_id"].tolist(),
        "rep_count": len(rep_stats),
        "win_rate_variance": rep_stats["win_rate"].var(),
        "win_rate_std": rep_stats["win_rate"].std()
    }


def analyze_time_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze patterns over time (monthly, quarterly trends).
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with time-based analysis
    """
    # Monthly trends
    df_copy = df.copy()
    df_copy["closed_year_month"] = df_copy["closed_date"].dt.to_period("M").astype(str)
    
    monthly = df_copy.groupby("closed_year_month").agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean"),
        avg_cycle_days=("sales_cycle_days", "mean")
    ).reset_index()
    
    monthly["win_rate"] = (monthly["won_deals"] / monthly["total_deals"] * 100).round(2)
    monthly["revenue_won"] = df_copy[df_copy["is_won"] == 1].groupby("closed_year_month")["deal_amount"].sum().reindex(monthly["closed_year_month"]).fillna(0).values
    
    # Quarterly trends
    quarterly = df.groupby("closed_year_quarter").agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean")
    ).reset_index()
    
    quarterly["win_rate"] = (quarterly["won_deals"] / quarterly["total_deals"] * 100).round(2)
    quarterly["revenue_won"] = df[df["is_won"] == 1].groupby("closed_year_quarter")["deal_amount"].sum().reindex(quarterly["closed_year_quarter"]).fillna(0).values
    
    return {
        "monthly": monthly.to_dict("records"),
        "quarterly": quarterly.to_dict("records"),
        "best_month": monthly.loc[monthly["win_rate"].idxmax()].to_dict(),
        "worst_month": monthly.loc[monthly["win_rate"].idxmin()].to_dict()
    }


def generate_eda_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generate comprehensive EDA insights with actionable recommendations.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        List of insight dictionaries with category, insight, and action
    """
    insights = []
    overall_win_rate = df["is_won"].mean() * 100
    
    # 1. Win Rate Trend Analysis
    trend = analyze_win_rate_trends(df)
    # Always generate a trend insight
    if trend["trend_direction"] == "declining":
        insights.append({
            "category": "Trend Alert",
            "emoji": "📉",
            "insight": f"Win rate has declined {abs(trend['decline_from_peak']):.1f}pp from peak ({trend['peak_quarter']}: {trend['peak_rate']:.1f}% to {trend['current_quarter']}: {trend['current_rate']:.1f}%)",
            "severity": "high",
            "action": "Investigate what changed between peak and current periods - new competitors, process changes, or team turnover?"
        })
    else:
        insights.append({
            "category": "Win Rate Trend",
            "emoji": "📈",
            "insight": f"Win rate is {trend['trend_direction']} - currently {trend['current_rate']:.1f}% ({trend['current_quarter']}), peak was {trend['peak_rate']:.1f}%",
            "severity": "low",
            "action": "Monitor quarterly to detect any emerging downward trends early"
        })
    
    # 2. Segment Analysis
    segments = analyze_segment_performance(df)
    
    # Region insights
    region_df = segments["region"]
    underperforming_regions = region_df[region_df["vs_overall"] < -1]
    for _, row in underperforming_regions.head(2).iterrows():
        insights.append({
            "category": "Regional Issue",
            "emoji": "🌍",
            "insight": f"{row['region']} has {row['win_rate']:.1f}% win rate ({abs(row['vs_overall']):.1f}pp below average) with {row['total_deals']} deals",
            "severity": "high" if abs(row['vs_overall']) > 5 else "medium",
            "action": f"Audit {row['region']} sales process, local competition, and rep performance"
        })
    
    # Lead source insights
    source_df = segments["lead_source"]
    best_source = source_df.iloc[0]
    worst_source = source_df.iloc[-1]
    
    if worst_source["win_rate"] > 0:
        ratio = best_source['win_rate'] / worst_source['win_rate']
        insights.append({
            "category": "Lead Source",
            "emoji": "🎯",
            "insight": f"{best_source['lead_source']} leads convert at {best_source['win_rate']:.1f}% vs {worst_source['lead_source']} at {worst_source['win_rate']:.1f}% ({ratio:.1f}x better)",
            "severity": "high" if ratio > 1.5 else "medium",
            "action": f"Increase investment in {best_source['lead_source']} channel and review {worst_source['lead_source']} lead quality"
        })
    
    # 3. Deal Characteristics
    deal_chars = analyze_deal_characteristics(df)
    
    # Sales cycle insight
    cycle_diff = deal_chars["sales_cycle"]["lost_avg"] - deal_chars["sales_cycle"]["won_avg"]
    if cycle_diff > 10:
        insights.append({
            "category": "Sales Cycle",
            "emoji": "⏱️",
            "insight": f"Lost deals take {cycle_diff:.0f} days longer on average ({deal_chars['sales_cycle']['lost_avg']:.0f}d vs {deal_chars['sales_cycle']['won_avg']:.0f}d for wins)",
            "severity": "medium",
            "action": "Implement deal velocity checkpoints at 60 and 90 days to identify stalled deals early"
        })
    
    # Deal size insight
    if deal_chars["deal_size"]["won_avg"] < deal_chars["deal_size"]["lost_avg"]:
        insights.append({
            "category": "Deal Size",
            "emoji": "💰",
            "insight": f"Larger deals have lower win rates - avg won deal is ${deal_chars['deal_size']['won_avg']:,.0f} vs ${deal_chars['deal_size']['lost_avg']:,.0f} for lost",
            "severity": "medium",
            "action": "Review Enterprise deal qualification criteria and consider multi-threading strategy for large deals"
        })
    
    # 4. Rep Performance
    rep_analysis = analyze_sales_rep_deep(df)
    
    if len(rep_analysis["underperformers"]) > 0:
        insights.append({
            "category": "Rep Performance",
            "emoji": "👥",
            "insight": f"{len(rep_analysis['underperformers'])} reps are underperforming (10+ pp below average win rate)",
            "severity": "high" if len(rep_analysis["underperformers"]) > 3 else "medium",
            "action": "Identify common patterns among underperformers and pair with top performers for mentorship"
        })
    
    if rep_analysis["win_rate_std"] > 15:
        insights.append({
            "category": "Rep Variance",
            "emoji": "📊",
            "insight": f"High variance in rep performance (std dev: {rep_analysis['win_rate_std']:.1f}pp) suggests inconsistent execution",
            "severity": "medium",
            "action": "Standardize sales playbook and implement regular deal reviews to reduce variance"
        })
    
    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 2))
    
    return insights


def run_full_eda(df: pd.DataFrame) -> Dict[str, any]:
    """
    Run complete EDA analysis and return all results.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with all EDA results
    """
    return {
        "win_rate_trends": analyze_win_rate_trends(df),
        "segment_performance": analyze_segment_performance(df),
        "deal_characteristics": analyze_deal_characteristics(df),
        "lead_source_analysis": analyze_lead_source_performance(df),
        "rep_analysis": analyze_sales_rep_deep(df),
        "time_patterns": analyze_time_patterns(df),
        "insights": generate_eda_insights(df)
    }


# For direct testing
if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    
    df, _ = load_and_prepare_data()
    
    print("=" * 60)
    print("SKYGENI SALES INTELLIGENCE - EDA SUMMARY")
    print("=" * 60)
    
    # Run full EDA
    results = run_full_eda(df)
    
    # Print trend analysis
    trend = results["win_rate_trends"]
    print(f"\n📈 WIN RATE TREND: {trend['trend_direction'].upper()}")
    print(f"   Peak: {trend['peak_quarter']} at {trend['peak_rate']:.1f}%")
    print(f"   Current: {trend['current_quarter']} at {trend['current_rate']:.1f}%")
    print(f"   Change: {trend['decline_from_peak']:+.1f}pp")
    
    # Print insights
    print(f"\n💡 TOP INSIGHTS ({len(results['insights'])} found):")
    for i, insight in enumerate(results["insights"][:5], 1):
        print(f"\n{i}. [{insight['severity'].upper()}] {insight['emoji']} {insight['category']}")
        print(f"   {insight['insight']}")
        print(f"   → Action: {insight['action']}")
