"""
Custom Metrics Module for SkyGeni Sales Intelligence Dashboard
===============================================================
Implements custom business metrics:
- Deal Velocity Index (DVI)
- Win Pressure Score
- Rep Consistency Score
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_deal_velocity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Deal Velocity Index (DVI) for each deal.
    
    DVI = (Deal Amount / Sales Cycle Days) / Median(Amount/Cycle)
    
    Higher DVI = More revenue per day of effort
    DVI > 1 = Above average efficiency
    DVI < 1 = Below average efficiency
    
    Args:
        df: DataFrame with deal_amount and sales_cycle_days
        
    Returns:
        DataFrame with DVI column added
    """
    df = df.copy()
    
    # Calculate revenue per day for each deal
    df["revenue_per_day"] = df["deal_amount"] / df["sales_cycle_days"].replace(0, 1)
    
    # Calculate median revenue per day
    median_rpd = df["revenue_per_day"].median()
    
    # Calculate DVI (normalized to median)
    df["deal_velocity_index"] = (df["revenue_per_day"] / median_rpd).round(3)
    
    # Categorize DVI
    df["dvi_category"] = pd.cut(
        df["deal_velocity_index"],
        bins=[0, 0.5, 0.8, 1.2, 2.0, float("inf")],
        labels=["Very Low", "Low", "Average", "High", "Very High"]
    )
    
    return df


def calculate_win_pressure_score(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """
    Calculate Win Pressure Score for each segment.
    
    Win Pressure = (Segment Win Rate × Segment Volume) / Overall Win Rate
    
    Score > 1 = Segment is lifting overall win rate
    Score < 1 = Segment is dragging down overall win rate
    
    The magnitude shows impact - higher volume segments have more pressure.
    
    Args:
        df: DataFrame with is_won column
        segment_column: Column to segment by
        
    Returns:
        DataFrame with segment-level win pressure scores
    """
    overall_win_rate = df["is_won"].mean()
    total_deals = len(df)
    
    segment_stats = df.groupby(segment_column).agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum")
    ).reset_index()
    
    segment_stats["win_rate"] = segment_stats["won_deals"] / segment_stats["total_deals"]
    segment_stats["volume_share"] = segment_stats["total_deals"] / total_deals
    
    # Win Pressure Score
    segment_stats["win_pressure_score"] = (
        (segment_stats["win_rate"] * segment_stats["volume_share"]) / overall_win_rate
    ).round(3)
    
    # Impact direction
    segment_stats["impact"] = np.where(
        segment_stats["win_pressure_score"] > 1, 
        "Positive (Lifting)", 
        "Negative (Dragging)"
    )
    
    # Contribution to overall win rate (percentage points)
    segment_stats["win_rate_contribution"] = (
        (segment_stats["win_rate"] - overall_win_rate) * segment_stats["volume_share"] * 100
    ).round(2)
    
    return segment_stats.sort_values("win_pressure_score", ascending=False)


def calculate_rep_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Rep Consistency Score for each sales rep.
    
    Consistency = 1 - (StdDev of monthly win rates / Mean of monthly win rates)
    
    Score closer to 1 = Highly consistent performer
    Score closer to 0 = Highly inconsistent (volatile) performer
    
    Args:
        df: DataFrame with sales_rep_id, is_won, and date columns
        
    Returns:
        DataFrame with rep-level consistency scores
    """
    # Calculate monthly win rates per rep
    df_copy = df.copy()
    df_copy["year_month"] = df_copy["closed_date"].dt.to_period("M").astype(str)
    
    monthly_rates = df_copy.groupby(["sales_rep_id", "year_month"]).agg(
        deals=("deal_id", "count"),
        wins=("is_won", "sum")
    ).reset_index()
    
    monthly_rates["monthly_win_rate"] = monthly_rates["wins"] / monthly_rates["deals"]
    
    # Calculate consistency score per rep
    rep_consistency = monthly_rates.groupby("sales_rep_id").agg(
        months_active=("year_month", "count"),
        mean_win_rate=("monthly_win_rate", "mean"),
        std_win_rate=("monthly_win_rate", "std"),
        min_win_rate=("monthly_win_rate", "min"),
        max_win_rate=("monthly_win_rate", "max")
    ).reset_index()
    
    # Fill NaN std with 0 (for reps with only 1 month)
    rep_consistency["std_win_rate"] = rep_consistency["std_win_rate"].fillna(0)
    
    # Calculate consistency score (avoid division by zero)
    rep_consistency["consistency_score"] = np.where(
        rep_consistency["mean_win_rate"] > 0,
        1 - (rep_consistency["std_win_rate"] / rep_consistency["mean_win_rate"]),
        0
    )
    
    # Clip to 0-1 range (can go negative if std > mean)
    rep_consistency["consistency_score"] = rep_consistency["consistency_score"].clip(0, 1).round(3)
    
    # Categorize consistency
    rep_consistency["consistency_category"] = pd.cut(
        rep_consistency["consistency_score"],
        bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=["Very Inconsistent", "Inconsistent", "Moderate", "Consistent", "Highly Consistent"]
    )
    
    # Win rate volatility (range)
    rep_consistency["win_rate_range"] = (
        rep_consistency["max_win_rate"] - rep_consistency["min_win_rate"]
    ).round(3)
    
    return rep_consistency.sort_values("consistency_score", ascending=False)


def calculate_all_custom_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate all custom metrics and return as dictionary.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with all custom metric DataFrames
    """
    # Deal Velocity Index (add to main df)
    df_with_dvi = calculate_deal_velocity_index(df)
    
    # Win Pressure Scores by different segments
    win_pressure_region = calculate_win_pressure_score(df, "region")
    win_pressure_industry = calculate_win_pressure_score(df, "industry")
    win_pressure_product = calculate_win_pressure_score(df, "product_type")
    win_pressure_source = calculate_win_pressure_score(df, "lead_source")
    
    # Rep Consistency Scores
    rep_consistency = calculate_rep_consistency_score(df)
    
    return {
        "df_with_dvi": df_with_dvi,
        "win_pressure_region": win_pressure_region,
        "win_pressure_industry": win_pressure_industry,
        "win_pressure_product": win_pressure_product,
        "win_pressure_source": win_pressure_source,
        "rep_consistency": rep_consistency
    }


def get_metric_summary(metrics_dict: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """
    Generate summary of custom metrics for dashboard display.
    
    Args:
        metrics_dict: Output from calculate_all_custom_metrics
        
    Returns:
        Dictionary with metric summaries
    """
    df = metrics_dict["df_with_dvi"]
    
    summary = {
        "dvi": {
            "mean": df["deal_velocity_index"].mean(),
            "median": df["deal_velocity_index"].median(),
            "high_velocity_deals": (df["deal_velocity_index"] > 1.2).sum(),
            "low_velocity_deals": (df["deal_velocity_index"] < 0.8).sum(),
            "won_avg_dvi": df[df["is_won"] == 1]["deal_velocity_index"].mean(),
            "lost_avg_dvi": df[df["is_won"] == 0]["deal_velocity_index"].mean()
        },
        "top_dragging_segments": {
            "region": metrics_dict["win_pressure_region"][
                metrics_dict["win_pressure_region"]["impact"] == "Negative (Dragging)"
            ]["region"].tolist()[:3],
            "industry": metrics_dict["win_pressure_industry"][
                metrics_dict["win_pressure_industry"]["impact"] == "Negative (Dragging)"
            ]["industry"].tolist()[:3]
        },
        "rep_consistency": {
            "avg_consistency": metrics_dict["rep_consistency"]["consistency_score"].mean(),
            "highly_consistent_reps": (
                metrics_dict["rep_consistency"]["consistency_score"] > 0.7
            ).sum(),
            "inconsistent_reps": (
                metrics_dict["rep_consistency"]["consistency_score"] < 0.5
            ).sum()
        }
    }
    
    return summary


def identify_key_insights(df: pd.DataFrame, metrics_dict: Dict[str, pd.DataFrame]) -> list:
    """
    Auto-generate key insights from custom metrics.
    
    Args:
        df: Preprocessed DataFrame
        metrics_dict: Output from calculate_all_custom_metrics
        
    Returns:
        List of insight strings
    """
    insights = []
    
    # DVI insights
    df_dvi = metrics_dict["df_with_dvi"]
    won_dvi = df_dvi[df_dvi["is_won"] == 1]["deal_velocity_index"].mean()
    lost_dvi = df_dvi[df_dvi["is_won"] == 0]["deal_velocity_index"].mean()
    
    if won_dvi > lost_dvi * 1.2:
        insights.append(
            f"🎯 Won deals have {((won_dvi/lost_dvi - 1) * 100):.0f}% higher Deal Velocity Index - "
            "faster deals with higher values are more likely to close."
        )
    
    # Win Pressure insights
    for segment_type in ["region", "industry"]:
        wp_df = metrics_dict[f"win_pressure_{segment_type}"]
        dragging = wp_df[wp_df["win_rate_contribution"] < -1]
        
        for _, row in dragging.head(2).iterrows():
            insights.append(
                f"⚠️ {row[segment_type]} is dragging win rate by {abs(row['win_rate_contribution']):.1f} "
                f"percentage points (win rate: {row['win_rate']*100:.1f}% vs overall)."
            )
    
    # Rep consistency insights
    rep_df = metrics_dict["rep_consistency"]
    inconsistent = rep_df[rep_df["consistency_score"] < 0.5]
    
    if len(inconsistent) > 0:
        insights.append(
            f"📊 {len(inconsistent)} sales reps have inconsistent performance month-over-month. "
            "Consider targeted coaching for process adherence."
        )
    
    # High performers with consistency
    high_consistent = rep_df[
        (rep_df["consistency_score"] > 0.7) & 
        (rep_df["mean_win_rate"] > df["is_won"].mean())
    ]
    
    if len(high_consistent) > 0:
        insights.append(
            f"⭐ {len(high_consistent)} reps are both high-performing AND consistent - "
            "consider documenting their best practices."
        )
    
    return insights


# For direct testing
if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    
    df, _ = load_and_prepare_data()
    metrics = calculate_all_custom_metrics(df)
    
    print("=== Deal Velocity Index ===")
    print(f"Won deals avg DVI: {metrics['df_with_dvi'][metrics['df_with_dvi']['is_won']==1]['deal_velocity_index'].mean():.2f}")
    print(f"Lost deals avg DVI: {metrics['df_with_dvi'][metrics['df_with_dvi']['is_won']==0]['deal_velocity_index'].mean():.2f}")
    
    print("\n=== Win Pressure by Region ===")
    print(metrics["win_pressure_region"][["region", "win_rate", "win_pressure_score", "impact"]])
    
    print("\n=== Rep Consistency ===")
    print(metrics["rep_consistency"][["sales_rep_id", "mean_win_rate", "consistency_score", "consistency_category"]].head(10))
    
    print("\n=== Key Insights ===")
    for insight in identify_key_insights(df, metrics):
        print(insight)
