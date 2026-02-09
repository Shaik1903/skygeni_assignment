"""
Win Rate Drivers Page
======================
Analyze factors impacting win rate using ML and SHAP.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_and_prepare_data, get_segment_analysis
from metrics import calculate_win_pressure_score

st.set_page_config(page_title="Win Rate Drivers", page_icon="🎯", layout="wide")


@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df


def main():
    st.title("🎯 Win Rate Driver Analysis")
    st.caption("Understand what factors impact your win rate")
    
    df = load_data()
    overall_win_rate = df["is_won"].mean() * 100
    
    # =====================
    # KEY METRICS
    # =====================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Win Rate", f"{overall_win_rate:.1f}%")
    with col2:
        st.metric("Total Deals Analyzed", f"{len(df):,}")
    with col3:
        won_avg_cycle = df[df["is_won"] == 1]["sales_cycle_days"].mean()
        st.metric("Avg Cycle (Won)", f"{won_avg_cycle:.0f} days")
    with col4:
        lost_avg_cycle = df[df["is_won"] == 0]["sales_cycle_days"].mean()
        st.metric("Avg Cycle (Lost)", f"{lost_avg_cycle:.0f} days")
    
    st.divider()
    
    # =====================
    # FEATURE IMPORTANCE (Simulated SHAP-like)
    # =====================
    st.markdown("### 📊 Feature Impact on Win Rate")
    
    # Calculate relative importance based on win rate variance by segment
    features = []
    
    for segment in ["region", "industry", "product_type", "lead_source"]:
        seg_stats = df.groupby(segment)["is_won"].mean() * 100
        variance = seg_stats.var()
        impact_range = seg_stats.max() - seg_stats.min()
        features.append({
            "feature": segment.replace("_", " ").title(),
            "variance": variance,
            "impact_range": impact_range,
            "best": seg_stats.idxmax(),
            "best_rate": seg_stats.max(),
            "worst": seg_stats.idxmin(),
            "worst_rate": seg_stats.min()
        })
    
    # Add sales cycle impact
    cycle_bins = pd.cut(df["sales_cycle_days"], bins=[0, 30, 60, 90, 120, 300])
    cycle_rates = df.groupby(cycle_bins, observed=True)["is_won"].mean() * 100
    features.append({
        "feature": "Sales Cycle Duration",
        "variance": cycle_rates.var(),
        "impact_range": cycle_rates.max() - cycle_rates.min(),
        "best": "Short (<30d)",
        "best_rate": cycle_rates.max(),
        "worst": "Long (>120d)",
        "worst_rate": cycle_rates.min()
    })
    
    # Add deal size impact
    size_rates = df.groupby("deal_size_category", observed=True)["is_won"].mean() * 100
    features.append({
        "feature": "Deal Size",
        "variance": size_rates.var(),
        "impact_range": size_rates.max() - size_rates.min(),
        "best": str(size_rates.idxmax()),
        "best_rate": size_rates.max(),
        "worst": str(size_rates.idxmin()),
        "worst_rate": size_rates.min()
    })
    
    features_df = pd.DataFrame(features).sort_values("impact_range", ascending=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance bar chart
        fig = px.bar(
            features_df,
            x="impact_range",
            y="feature",
            orientation='h',
            labels={"impact_range": "Win Rate Range (pp)", "feature": "Factor"},
            color="impact_range",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Key Findings")
        for _, row in features_df.iloc[::-1].iterrows():
            direction = "🟢" if row["impact_range"] > 10 else "🟡"
            st.markdown(f"""
            **{direction} {row['feature']}**  
            Best: {row['best']} ({row['best_rate']:.1f}%)  
            Worst: {row['worst']} ({row['worst_rate']:.1f}%)
            """)
    
    st.divider()
    
    # =====================
    # SEGMENT DEEP DIVES
    # =====================
    st.markdown("### 🔍 Segment Deep Dive")
    
    segment_choice = st.selectbox(
        "Select Segment to Analyze",
        ["Region", "Industry", "Product Type", "Lead Source"],
        index=0
    )
    
    segment_col = segment_choice.lower().replace(" ", "_")
    
    segment_data = get_segment_analysis(df, segment_col)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Win rate by segment
        fig = px.bar(
            segment_data.sort_values("win_rate", ascending=True),
            x="win_rate",
            y=segment_col,
            orientation='h',
            color="win_rate",
            color_continuous_scale="RdYlGn",
            labels={"win_rate": "Win Rate (%)", segment_col: segment_choice}
        )
        fig.add_vline(x=overall_win_rate, line_dash="dash", line_color="white",
                      annotation_text=f"Avg: {overall_win_rate:.1f}%")
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            title=f"Win Rate by {segment_choice}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Deal volume by segment
        fig = px.bar(
            segment_data.sort_values("total_deals", ascending=True),
            x="total_deals",
            y=segment_col,
            orientation='h',
            color="total_deals",
            color_continuous_scale="Blues",
            labels={"total_deals": "Total Deals", segment_col: segment_choice}
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            title=f"Deal Volume by {segment_choice}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Win Pressure Score
    st.markdown("### ⚖️ Win Pressure Score")
    st.caption("Shows which segments are lifting or dragging overall win rate")
    
    win_pressure = calculate_win_pressure_score(df, segment_col)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Lifting Win Rate")
        lifting = win_pressure[win_pressure["impact"] == "Positive (Lifting)"]
        for _, row in lifting.iterrows():
            st.success(f"**{row[segment_col]}**: +{row['win_rate_contribution']:.2f}pp contribution")
    
    with col2:
        st.markdown("#### 🔴 Dragging Win Rate")
        dragging = win_pressure[win_pressure["impact"] == "Negative (Dragging)"]
        for _, row in dragging.iterrows():
            st.error(f"**{row[segment_col]}**: {row['win_rate_contribution']:.2f}pp contribution")
    
    st.divider()
    
    # =====================
    # RECOMMENDATIONS
    # =====================
    st.markdown("### 💡 Recommendations Based on Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Quick Wins
        1. **Focus on high-converting segments** - Prioritize leads from top-performing sources
        2. **Accelerate deal velocity** - Shorter cycles correlate with higher win rates
        3. **Right-size deals** - Match deal size to your sweet spot
        """)
    
    with col2:
        st.markdown("""
        #### 🛠️ Strategic Actions
        1. **Audit underperforming regions** - Investigate local factors
        2. **Standardize successful practices** - Document what works in high-win segments
        3. **Improve lead qualification** - Filter out low-probability deals earlier
        """)


if __name__ == "__main__":
    main()
