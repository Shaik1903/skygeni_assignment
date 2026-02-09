"""
Data Loader Module for SkyGeni Sales Intelligence Dashboard
============================================================
Handles data loading, validation, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load sales data from CSV file.
    
    Args:
        filepath: Path to CSV file. If None, uses default path.
        
    Returns:
        DataFrame with raw sales data
    """
    if filepath is None:
        # Default path relative to project root
        filepath = Path(__file__).parent.parent / "data" / "skygeni_sales_data.csv"
    
    df = pd.read_csv(filepath)
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return validation report.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Dictionary with validation results
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "date_range": {
            "created_min": df["created_date"].min(),
            "created_max": df["created_date"].max(),
            "closed_min": df["closed_date"].min(),
            "closed_max": df["closed_date"].max()
        },
        "unique_values": {
            "industries": df["industry"].nunique(),
            "regions": df["region"].nunique(),
            "product_types": df["product_type"].nunique(),
            "lead_sources": df["lead_source"].nunique(),
            "sales_reps": df["sales_rep_id"].nunique(),
            "deal_stages": df["deal_stage"].nunique()
        },
        "outcome_distribution": df["outcome"].value_counts().to_dict(),
        "is_valid": True
    }
    
    # Check for critical issues
    if df.isnull().sum().sum() > 0:
        report["warnings"] = ["Dataset contains missing values"]
    
    if report["duplicates"] > 0:
        report["warnings"] = report.get("warnings", []) + [f"{report['duplicates']} duplicate rows found"]
    
    return report


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data: parse dates, create derived features, clean data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame with additional features
    """
    df = df.copy()
    
    # Parse dates
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df["closed_date"])
    
    # Create time-based features
    df["created_year"] = df["created_date"].dt.year
    df["created_month"] = df["created_date"].dt.month
    df["created_quarter"] = df["created_date"].dt.quarter
    df["created_year_quarter"] = df["created_date"].dt.to_period("Q").astype(str)
    
    df["closed_year"] = df["closed_date"].dt.year
    df["closed_month"] = df["closed_date"].dt.month
    df["closed_quarter"] = df["closed_date"].dt.quarter
    df["closed_year_quarter"] = df["closed_date"].dt.to_period("Q").astype(str)
    
    # Create outcome binary
    df["is_won"] = (df["outcome"] == "Won").astype(int)
    
    # Deal size categories
    df["deal_size_category"] = pd.cut(
        df["deal_amount"],
        bins=[0, 10000, 25000, 50000, 100000, float("inf")],
        labels=["Small (<$10K)", "Medium ($10K-$25K)", "Large ($25K-$50K)", 
                "Enterprise ($50K-$100K)", "Strategic (>$100K)"]
    )
    
    # Sales cycle categories
    df["cycle_category"] = pd.cut(
        df["sales_cycle_days"],
        bins=[0, 30, 60, 90, 120, float("inf")],
        labels=["Fast (<30d)", "Normal (30-60d)", "Slow (60-90d)", 
                "Extended (90-120d)", "Very Long (>120d)"]
    )
    
    return df


def get_summary_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate summary statistics for the dataset.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    total_deals = len(df)
    won_deals = df["is_won"].sum()
    lost_deals = total_deals - won_deals
    
    stats = {
        "total_deals": total_deals,
        "won_deals": won_deals,
        "lost_deals": lost_deals,
        "overall_win_rate": won_deals / total_deals * 100,
        "total_revenue_won": df[df["is_won"] == 1]["deal_amount"].sum(),
        "total_revenue_lost": df[df["is_won"] == 0]["deal_amount"].sum(),
        "avg_deal_size": df["deal_amount"].mean(),
        "avg_deal_size_won": df[df["is_won"] == 1]["deal_amount"].mean(),
        "avg_deal_size_lost": df[df["is_won"] == 0]["deal_amount"].mean(),
        "avg_sales_cycle": df["sales_cycle_days"].mean(),
        "avg_sales_cycle_won": df[df["is_won"] == 1]["sales_cycle_days"].mean(),
        "avg_sales_cycle_lost": df[df["is_won"] == 0]["sales_cycle_days"].mean(),
        "median_deal_size": df["deal_amount"].median(),
        "median_sales_cycle": df["sales_cycle_days"].median(),
    }
    
    return stats


def get_segment_analysis(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """
    Calculate win rate and stats by segment.
    
    Args:
        df: Preprocessed DataFrame
        segment_column: Column to segment by
        
    Returns:
        DataFrame with segment-level statistics
    """
    segment_stats = df.groupby(segment_column).agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean"),
        avg_cycle_days=("sales_cycle_days", "mean")
    ).reset_index()
    
    segment_stats["win_rate"] = (segment_stats["won_deals"] / segment_stats["total_deals"] * 100).round(2)
    segment_stats["revenue_won"] = df[df["is_won"] == 1].groupby(segment_column)["deal_amount"].sum().values
    segment_stats["deal_share"] = (segment_stats["total_deals"] / len(df) * 100).round(2)
    
    return segment_stats.sort_values("win_rate", ascending=False)


def get_quarterly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate quarterly win rate and revenue trends.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with quarterly trends
    """
    quarterly = df.groupby("closed_year_quarter").agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean"),
        avg_cycle_days=("sales_cycle_days", "mean")
    ).reset_index()
    
    quarterly["win_rate"] = (quarterly["won_deals"] / quarterly["total_deals"] * 100).round(2)
    quarterly["revenue_won"] = df[df["is_won"] == 1].groupby("closed_year_quarter")["deal_amount"].sum().reindex(quarterly["closed_year_quarter"]).fillna(0).values
    
    # Calculate quarter-over-quarter changes
    quarterly["win_rate_change"] = quarterly["win_rate"].diff()
    quarterly["revenue_change_pct"] = quarterly["revenue_won"].pct_change() * 100
    
    return quarterly


def get_rep_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sales rep performance metrics.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with rep-level statistics
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
    
    # Rank reps
    rep_stats["win_rate_rank"] = rep_stats["win_rate"].rank(ascending=False, method="min").astype(int)
    rep_stats["revenue_rank"] = rep_stats["revenue_won"].rank(ascending=False, method="min").astype(int)
    
    return rep_stats.sort_values("win_rate", ascending=False)


def create_heatmap_data(df: pd.DataFrame, row_col: str, col_col: str, 
                         value_col: str = "win_rate") -> pd.DataFrame:
    """
    Create pivot table for heatmap visualization.
    
    Args:
        df: Preprocessed DataFrame
        row_col: Column for rows
        col_col: Column for columns
        value_col: Metric to display (win_rate, avg_deal_size, etc.)
        
    Returns:
        Pivot table DataFrame
    """
    # First calculate segment-level metrics
    segment_df = df.groupby([row_col, col_col]).agg(
        total_deals=("deal_id", "count"),
        won_deals=("is_won", "sum"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean")
    ).reset_index()
    
    segment_df["win_rate"] = (segment_df["won_deals"] / segment_df["total_deals"] * 100).round(2)
    
    # Create pivot
    pivot = segment_df.pivot(index=row_col, columns=col_col, values=value_col)
    
    return pivot


def load_and_prepare_data(filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete data loading pipeline: load, validate, preprocess.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (preprocessed DataFrame, validation report)
    """
    df = load_data(filepath)
    validation_report = validate_data(df)
    df = preprocess_data(df)
    
    return df, validation_report


# For direct testing
if __name__ == "__main__":
    df, report = load_and_prepare_data()
    print(f"Loaded {report['total_rows']} rows")
    print(f"Date range: {report['date_range']['created_min']} to {report['date_range']['closed_max']}")
    print(f"Outcome distribution: {report['outcome_distribution']}")
    
    stats = get_summary_stats(df)
    print(f"\nOverall Win Rate: {stats['overall_win_rate']:.1f}%")
    print(f"Total Revenue Won: ${stats['total_revenue_won']:,.0f}")
