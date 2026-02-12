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
    calculate_pipeline_qualification_score,
    calculate_win_rate_elasticity
)
from forecasting import get_forecast_pipeline_cached

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
        pqs_score = metrics['pqs_overall']['pqs_score']
        st.metric("Pipeline Qualification Score", f"{pqs_score:.0f}/100")
    
    st.divider()
    
    # =====================
    # FEATURE IMPORTANCE (Real SHAP)
    # =====================
    st.markdown("### 🔑 What Actually Drives Wins? (AI Analysis)")
    st.caption("Based on XGBoost model analysis of 5,000 deals")
    
    # Get cached forecast data (includes SHAP)
    with st.spinner("Analyzing win drivers..."):
        try:
            forecast_data = get_forecast_pipeline_cached(df)
            shap_data = forecast_data["shap_analysis"]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # SHAP Bar Chart
                features = [shap_data["feature_names"][i] for i in shap_data["sorted_idx"]]
                importance = [shap_data["mean_abs_shap"][i] for i in shap_data["sorted_idx"]]
                
                # Top 10 features
                features = features[:10][::-1]
                importance = importance[:10][::-1]
                
                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    labels={'x': 'Mean |SHAP| Value (Impact)', 'y': 'Feature'},
                    color=importance,
                    color_continuous_scale='Bluered'
                )
                fig.update_layout(
                    title="Top Factors Influencing Win Probability",
                    height=400,
                    showlegend=False,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                st.markdown("#### Key Findings")
                top_factor = features[-1]
                st.info(f"💡 **Top Driver:**\n\nThe most critical factor driving wins is **{top_factor}**.")
                
                st.markdown(f"""
                **What this means:**
                The AI model identified {top_factor} as having the strongest influence on whether a deal closes won or lost.
                
                **Action:**
                Drill down into this metric below to understand the directional impact.
                """)
        except Exception as e:
            st.error(f"Could not load AI analysis: {e}")
            st.info("Falling back to standard metrics below.")
    
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
        st.plotly_chart(fig, width='stretch')
    
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
        st.plotly_chart(fig, width='stretch')
    
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
        st.plotly_chart(fig, width='stretch')
    
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
    st.markdown("### ⚖️ Pipeline Qualification Score (PQS)")
    st.caption("How quickly we exit losing deals & which deals are stalling past the winning window")
    
    pqs = metrics['pqs_overall']
    pqs_by_region = metrics['pqs_by_region'].get('segment_breakdown')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pqs_by_region is not None and len(pqs_by_region) > 0:
            fig = px.bar(
                pqs_by_region.sort_values("dqe_score"),
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
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Insufficient data for regional DQE breakdown.")
    
    with col2:
        st.markdown("#### Interpretation")
        st.markdown(f"""
        **PQS: {pqs['pqs_score']:.0f}/100** ({pqs['pqs_category']})
        
        **Qualification Efficiency (DQE): {pqs['dqe_score']:.3f}**
        - Won deals take {pqs['median_cycle_won']:.0f} days | Lost deals take {pqs['median_cycle_lost']:.0f} days
        - DQE > 0.25 = Excellent | DQE ~ 0 = Poor | DQE < 0 = Critical
        
        **Stall Detection:**
        - Winning window: {pqs['winning_window_days']:.0f} days (P75 of won deals)
        - {pqs['deals_stalling']} deals stalling ({pqs['stall_rate_pct']:.0f}% of pipeline)
        - {pqs['deals_likely_dead']} deals likely dead
        - Win rate gap: {pqs['on_pace_win_rate']:.0f}% (on-pace) vs {pqs['stalling_win_rate']:.0f}% (stalling)
        
        **Wasted capacity: ~{pqs['wasted_capacity_days']:,.0f} rep-days** spent on deals
        past the winning window that should have been disqualified.
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
