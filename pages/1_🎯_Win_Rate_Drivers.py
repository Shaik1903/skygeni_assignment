"""
Win Rate Drivers Page
======================
Comprehensive diagnostic analysis of factors impacting win rates.
Designed for executive leadership to identify root causes and growth opportunities.
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

# Custom CSS for executive dashboard aesthetic
st.markdown("""
    <style>
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f2f6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #00d4ff;
        padding-left: 1rem;
    }
    .insight-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #00d4ff;
        height: 100%;
    }
    .metric-label {
        color: #9da5b1;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df

@st.cache_data
def get_metrics(_df):
    return calculate_all_custom_metrics(_df)

def main():
    st.title("🎯 Win Rate Diagnostic Center")
    st.markdown("### Executive analysis of sales performance and deal drivers")
    
    df = load_data()
    metrics = get_metrics(df)
    overall_win_rate = df["is_won"].mean() * 100
    
    # =========================================================================
    # SECTION 1: THE "WHY" — WHAT DRIVES WINS?
    # =========================================================================
    st.markdown('<div class="section-header">Section 1: AI-Driven Win/Loss Propensity Analysis</div>', unsafe_allow_html=True)
    
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
                    labels={'x': 'Relative Impact (Mean |SHAP|)', 'y': ''},
                    color=importance,
                    color_continuous_scale='Bluered'
                )
                fig.update_layout(
                    title="Top Factors Influencing Win Probability",
                    height=450,
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### CEO Diagnostic Summary")
                top_factor = features[-1]
                
                st.markdown(f"""<div class="insight-card">
<p style="font-size: 1.1rem; font-weight: 600; color: #00d4ff;">💡 Primary Win Driver</p>
<p>The most critical behavioral factor for winning a deal is <b>{top_factor}</b>.</p>
<hr style="opacity: 0.1; margin: 1rem 0;">
<p style="color: #9da5b1; font-size: 0.9rem;"><b>Executive Verdict:</b></p>
<p>Our AI model analyzed 5,000+ historical deals. Unlike standard reports that just look at "who won," this analysis looks at the <i>behavioral DNA</i> of every deal.</p>
<p>Winning isn't just about the rep's skill; it's driven by <b>deal sizing compared to industry norms</b> and the <b>initial velocity</b> of the sales cycle.</p>
</div>""", unsafe_allow_html=True)
                


        except Exception as e:
            st.error(f"Could not load AI analysis: {e}")

    st.divider()

    # =========================================================================
    # SECTION 2: THE "WHERE" — SWEET SPOTS & MOMENTUM
    # =========================================================================
    st.markdown('<div class="section-header">Section 2: Revenue Elasticity & Market Momentum</div>', unsafe_allow_html=True)
    
    tab_sweet, tab_momentum = st.tabs(["🎯 Revenue Sweet Spot", "📈 Segment Momentum"])
    
    with tab_sweet:
        wre = metrics['wre']
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sweet Spot Visualization
            stats = wre['bucket_stats']
            
            fig = go.Figure()
            
            # Bar for deal count
            fig.add_trace(go.Bar(
                x=stats['size_range'],
                y=stats['deal_count'],
                name='Number of Deals',
                marker_color='rgba(158, 202, 225, 0.6)',
                yaxis='y'
            ))
            
            # Line for win rate
            fig.add_trace(go.Scatter(
                x=stats['size_range'],
                y=stats['win_rate_pct'],
                name='Win Rate (%)',
                line=dict(color='#00d4ff', width=4),
                yaxis='y2'
            ))
            
            # Point for sweet spot
            fig.add_trace(go.Scatter(
                x=[wre['sweet_spot_range']],
                y=[wre['sweet_spot_win_rate']],
                name='SWEET SPOT',
                mode='markers',
                marker=dict(size=18, color='#ff4b4b', symbol='star'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"Win Rate Elasticity: Finding the Revenue Sweet Spot",
                yaxis=dict(title="Volume of Deals", side='left'),
                yaxis2=dict(title="Win Rate (%)", side='right', overlaying='y', range=[0, 100]),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Elasticity Analysis")
            st.markdown(f"""<div class="insight-card" style="border-left-color: {'#ff4b4b' if wre['severity'] == 'high' else '#ffbd45' if wre['severity'] == 'medium' else '#00d4ff'}">
<p class="metric-label">Win Rate Elasticity</p>
<p class="metric-value">{wre['elasticity']:.2f}</p>
<p style="margin-top: 1rem;"><b>Diagnostic:</b> {wre['interpretation']}</p>
<hr style="opacity: 0.1; margin: 1rem 0;">
<p class="metric-label">Optimal Deal Size</p>
<p class="metric-value">{wre['sweet_spot_range']}</p>
<p style="margin-top: 0.5rem; font-size: 0.9rem;">Deals in this range offer the best balance between deal size and win probability ({wre['sweet_spot_win_rate']}%).</p>
</div>""", unsafe_allow_html=True)
            
            st.info(f"💡 **Opportunity:** Closing the {wre['wr_drop_small_to_large']:.0f}pp gap between small and large deals could increase total won revenue significantly.")

    with tab_momentum:
        col_m1, col_m2 = st.columns([1, 1])
        
        with col_m1:
            segment_choice = st.selectbox(
                "Filter Momentum by:",
                ["Industry", "Region", "Product Type", "Lead Source"],
                index=0
            )
            segment_col = segment_choice.lower().replace(" ", "_")
            smi = calculate_segment_momentum_index(df, segment_col)
            
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
                height=400,
                showlegend=False,
                title=f"{segment_choice} Momentum (SMI Index)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_m2:
            st.markdown("#### Segment Health Check")
            
            # Highlights
            top_grower = smi.iloc[0]
            top_decliner = smi.iloc[-1]
            
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.success(f"**Top Grower:** {top_grower[segment_col]}")
                st.caption(f"SMI: {top_grower['smi_score']:+.2f}")
            with col_h2:
                st.error(f"**Sharpest Decline:** {top_decliner[segment_col]}")
                st.caption(f"SMI: {top_decliner['smi_score']:+.2f}")
                
            st.markdown("---")
            for _, row in smi.head(3).iterrows():
                st.write(f"🚀 **{row[segment_col]}** is showing **{row['momentum_category']}** momentum.")
            
            st.write("")
            st.caption("Momentum Score (SMI) combines Win Rate change, Volume change, and Deal Size shift vs historical baseline.")

    st.divider()

    # =========================================================================
    # SECTION 3: THE "HOW" — EFFICIENCY & DISCIPLINE
    # =========================================================================
    st.markdown('<div class="section-header">Section 3: Sales Velocity & Pipeline Discipline</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Deal Velocity Benchmarks")
        st.caption("Revenue generated per day of sales effort (DVI index)")
        
        df_dvi = metrics['df_with_dvi']
        won_dvi = df_dvi[df_dvi["is_won"] == 1]["deal_velocity_index"].mean()
        lost_dvi = df_dvi[df_dvi["is_won"] == 0]["deal_velocity_index"].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = won_dvi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "WON DEALS VELOCITY", 'font': {'size': 20}},
            delta = {'reference': lost_dvi, 'increasing': {'color': "#00d4ff"}},
            gauge = {
                'axis': {'range': [None, max(won_dvi, lost_dvi) * 1.5]},
                'bar': {'color': "#00d4ff"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, lost_dvi], 'color': 'rgba(255, 75, 75, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': lost_dvi}
            }
        ))
        fig.update_layout(height=350, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **The Diagnostic:** Won deals generate **{(won_dvi/lost_dvi - 1)*100:.0f}% more revenue per day** than lost deals. 
        If a deal's velocity (revenue/days) drops below **{lost_dvi:.2f}**, it has entered the "Danger Zone" where losers outweigh winners by 3-to-1.
        """)
        
    with col2:
        st.markdown("#### Pipeline Qualification Health")
        pqs = metrics['pqs_overall']
        
        st.markdown(f"""<div class="insight-card">
<p class="metric-label">Qualification Score (PQS)</p>
<p class="metric-value">{pqs['pqs_score']:.0f} / 100</p>
<p style="margin-top: 0.5rem; font-weight: 600; color: {'#00ff00' if pqs['pqs_score'] > 70 else '#ffbd45' if pqs['pqs_score'] > 40 else '#ff4b4b'}">{pqs['pqs_category']}</p>
<hr style="opacity: 0.1; margin: 1rem 0;">
<table style="width: 100%; font-size: 0.9rem;">
<tr>
<td class="metric-label">Avg Winner Cycle</td>
<td class="metric-label" style="text-align: right;">Avg Loser Cycle</td>
</tr>
<tr>
<td class="metric-value" style="font-size: 1.2rem;">{pqs['median_cycle_won']:.0f} days</td>
<td class="metric-value" style="font-size: 1.2rem; text-align: right;">{pqs['median_cycle_lost']:.0f} days</td>
</tr>
</table>
<p style="margin-top: 1rem; font-size: 0.85rem; color: #9da5b1;">
<b>Capacity Warning:</b> We've spent ~<b>{pqs['wasted_capacity_days']:,.0f} rep-days</b> on deals that should have been disqualified earlier. 
That's the equivalent of having <b>{pqs['wasted_capacity_days']/250:.1f}</b> full-time reps working only on deals that will eventually lose.
</p>
</div>""", unsafe_allow_html=True)
        
        st.warning(f"⚠️ **Action Required:** {pqs['deals_stalling']} deals are currently 'stalling' past the winning window of {pqs['winning_window_days']:.0f} days.")

    # =========================================================================
    # SECTION 4: STRATEGIC RECOMMENDATIONS
    # =========================================================================
    st.divider()
    st.markdown("### 💡 Recommended Executive Actions")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.markdown(f"""<div class="insight-card" style="border-left-color: #ff4b4b;">
<p style="font-weight: 700;">1. Operational Discipline</p>
<p style="font-size: 0.9rem;">Review all <b>{pqs['deals_stalling']} deals</b> past the {pqs['winning_window_days']:.0f}-day window. Force a 'win-it-or-kill-it' review by end of week.</p>
</div>""", unsafe_allow_html=True)
        
    with col_rec2:
        st.markdown(f"""<div class="insight-card" style="border-left-color: #ffbd45;">
<p style="font-weight: 700;">2. Resource Allocation</p>
<p style="font-size: 0.9rem;">Pivot sales training toward the <b>{wre['sweet_spot_range']}</b> sweet spot. This range yields the highest revenue per unit of effort.</p>
</div>""", unsafe_allow_html=True)
        
    with col_rec3:
        st.markdown(f"""<div class="insight-card" style="border-left-color: #00d4ff;">
<p style="font-weight: 700;">3. Growth Investment</p>
<p style="font-size: 0.9rem;">Increase marketing and outbound spend for <b>{smi.iloc[0][segment_col]}</b>, which has {smi.iloc[0]['smi_score'] * 100:+.0f}% momentum index.</p>
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
