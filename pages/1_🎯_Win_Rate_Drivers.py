"""
Win Rate Drivers Page
======================
Analyze factors impacting win rate using custom metrics.
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
from metrics import (
    calculate_all_custom_metrics,
    calculate_segment_momentum_index,
    calculate_deal_qualification_efficiency,
    calculate_revenue_concentration_risk
)

st.set_page_config(page_title="Win Rate Drivers", page_icon="🎯", layout="wide")


@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df


@st.cache_data
def get_metrics(_df):
    return calculate_all_custom_metrics(_df)


def main():
    st.title("🎯 Win Rate Driver Analysis")
    st.caption("Understand what factors impact your win rate")
    
    df = load_data()
    metrics = get_metrics(df)
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
        dqe_score = float(metrics['dqe_overall']['dqe_score'].iloc[0])
        st.metric("Deal Qualification Efficiency", f"{dqe_score:.2f}")
    
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
    
    # =====================
    # SEGMENT MOMENTUM INDEX
    # =====================
    st.markdown("### 📈 Segment Momentum Index (SMI)")
    st.caption("Composite indicator: win rate trend + volume trend + deal size trend")
    
    smi = calculate_segment_momentum_index(df, segment_col)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SMI bar chart
        fig = px.bar(
            smi.sort_values("smi_score"),
            x="smi_score",
            y=segment_col,
            orientation='h',
            color="smi_score",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"smi_score": "Momentum Score", segment_col: segment_choice}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            title="Segment Momentum (negative = declining)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Momentum Breakdown")
        for _, row in smi.iterrows():
            category = row['momentum_category']
            st.markdown(f"""
            **{category} {row[segment_col]}** (SMI: {row['smi_score']:+.3f})  
            - Win Rate: {row.get('win_rate_delta', 0):+.1f}% change  
            - Volume: {row.get('volume_delta', 0):+.1f}% change  
            - Deal Size: {row.get('deal_size_delta', 0):+.1f}% change
            """)
    
    st.divider()
    
    # =====================
    # DEAL QUALIFICATION EFFICIENCY
    # =====================
    st.markdown("### ⚖️ Deal Qualification Efficiency (DQE)")
    st.caption("How quickly we exit losing deals vs. invest in winning ones")
    
    dqe_by_region = metrics['dqe_by_region']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            dqe_by_region.sort_values("dqe_score"),
            x="dqe_score",
            y="region",
            orientation='h',
            color="dqe_score",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"dqe_score": "DQE Score", "region": "Region"}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white",
                      annotation_text="Equal cycles")
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            showlegend=False,
            title="DQE by Region (negative = losers take longer than winners)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Interpretation")
        overall_dqe = float(metrics['dqe_overall']['dqe_score'].iloc[0])
        wasted = float(metrics['dqe_overall']['wasted_capacity_days'].iloc[0])
        st.markdown(f"""
        **Overall DQE: {overall_dqe:.3f}** ({metrics['dqe_overall']['dqe_category'].iloc[0]})
        
        - **DQE > 0.25**: Excellent — we exit losers fast
        - **DQE ~ 0**: Poor — losers take as long as winners
        - **DQE < 0**: Critical — losers take LONGER than wins
        
        **Wasted capacity: ~{wasted:,.0f} rep-days** spent on deals
        that ultimately lost but took too long to disqualify.
        """)
    
    st.divider()
    
    # =====================
    # RECOMMENDATIONS
    # =====================
    st.markdown("### 💡 Recommendations Based on Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Quick Wins
        1. **Implement deal checkpoints** — Review stalled deals at 30 and 60 days
        2. **Focus on high-momentum segments** — Shift resources toward growing areas
        3. **Right-size deals** — Match deal size to your sweet spot
        """)
    
    with col2:
        st.markdown("""
        #### 🛠️ Strategic Actions
        1. **Train on fast disqualification** — Coach reps to exit losers early
        2. **Investigate declining segments** — Find root cause of negative SMI
        3. **Diversify pipeline** — Reduce concentration risk in top segments
        """)


if __name__ == "__main__":
    main()
