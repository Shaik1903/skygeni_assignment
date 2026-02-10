"""
SkyGeni Sales Intelligence - Source Module
==========================================
"""

from .data_loader import (
    load_data,
    validate_data,
    preprocess_data,
    get_summary_stats,
    get_segment_analysis,
    get_quarterly_trends,
    get_rep_performance,
    create_heatmap_data,
    load_and_prepare_data
)

from .metrics import (
    calculate_deal_velocity_index,
    calculate_pipeline_qualification_score,
    calculate_win_rate_elasticity,
    calculate_segment_momentum_index,
    calculate_all_custom_metrics,
    get_metric_summary,
    identify_key_insights
)

from .eda import (
    analyze_win_rate_trends,
    analyze_segment_performance,
    analyze_deal_characteristics,
    analyze_lead_source_performance,
    analyze_sales_rep_deep,
    analyze_time_patterns,
    generate_eda_insights,
    run_full_eda
)

from .llm_insights import (
    generate_llm_metric_insights,
    generate_llm_eda_insights,
    generate_llm_recommendations
)

__all__ = [
    # Data loader
    "load_data",
    "validate_data", 
    "preprocess_data",
    "get_summary_stats",
    "get_segment_analysis",
    "get_quarterly_trends",
    "get_rep_performance",
    "create_heatmap_data",
    "load_and_prepare_data",
    # Metrics
    "calculate_deal_velocity_index",
    "calculate_pipeline_qualification_score",
    "calculate_win_rate_elasticity",
    "calculate_segment_momentum_index",
    "calculate_all_custom_metrics",
    "get_metric_summary",
    "identify_key_insights",
    # EDA
    "analyze_win_rate_trends",
    "analyze_segment_performance",
    "analyze_deal_characteristics",
    "analyze_lead_source_performance",
    "analyze_sales_rep_deep",
    "analyze_time_patterns",
    "generate_eda_insights",
    "run_full_eda",
    # LLM Insights
    "generate_llm_metric_insights",
    "generate_llm_eda_insights",
    "generate_llm_recommendations",
]
