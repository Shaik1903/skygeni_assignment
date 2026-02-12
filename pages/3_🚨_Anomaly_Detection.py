"""
Anomaly Detection Page
=======================
Detect unusual patterns in sales pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_and_prepare_data

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")


@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df


def detect_anomalies(series: pd.Series, threshold: float = 2.0):
    """Detect anomalies using Z-score method."""
    z_scores = np.abs(stats.zscore(series.fillna(series.mean())))
    return z_scores > threshold


def calculate_rolling_anomalies(df: pd.DataFrame, metric: str, window: int = 4):
    """Calculate rolling anomalies for a metric."""
    # Group by month
    df['closed_year_month'] = df['closed_date'].dt.to_period('M').astype(str)
    
    if metric == 'win_rate':
        monthly = df.groupby('closed_year_month').agg(
            value=('is_won', 'mean')
        ).reset_index()
        monthly['value'] *= 100
    elif metric == 'deal_count':
        monthly = df.groupby('closed_year_month').agg(
            value=('deal_id', 'count')
        ).reset_index()
    elif metric == 'avg_deal_size':
        monthly = df.groupby('closed_year_month').agg(
            value=('deal_amount', 'mean')
        ).reset_index()
    elif metric == 'avg_cycle':
        monthly = df.groupby('closed_year_month').agg(
            value=('sales_cycle_days', 'mean')
        ).reset_index()
    else:
        monthly = df.groupby('closed_year_month').agg(
            value=('deal_amount', 'sum')
        ).reset_index()
    
    # Calculate rolling stats
    monthly['rolling_mean'] = monthly['value'].rolling(window=window, min_periods=2).mean()
    monthly['rolling_std'] = monthly['value'].rolling(window=window, min_periods=2).std()
    monthly['z_score'] = (monthly['value'] - monthly['rolling_mean']) / monthly['rolling_std'].replace(0, 1)
    monthly['is_anomaly'] = np.abs(monthly['z_score']) > 2.0
    
    return monthly


def get_segment_anomalies(df: pd.DataFrame, segment_col: str):
    """Detect anomalies within segments over time."""
    df['closed_year_month'] = df['closed_date'].dt.to_period('M').astype(str)
    
    # Recent vs historical comparison
    recent_months = df['closed_year_month'].unique()[-2:]
    historical_months = df['closed_year_month'].unique()[:-2]
    
    recent = df[df['closed_year_month'].isin(recent_months)]
    historical = df[df['closed_year_month'].isin(historical_months)]
    
    anomalies = []
    
    for segment in df[segment_col].unique():
        recent_seg = recent[recent[segment_col] == segment]
        hist_seg = historical[historical[segment_col] == segment]
        
        if len(recent_seg) < 5 or len(hist_seg) < 10:
            continue
        
        recent_rate = recent_seg['is_won'].mean() * 100
        hist_rate = hist_seg['is_won'].mean() * 100
        hist_std = hist_seg.groupby('closed_year_month')['is_won'].mean().std() * 100
        
        if hist_std == 0:
            hist_std = 5  # Default
        
        z_score = (recent_rate - hist_rate) / hist_std
        
        if abs(z_score) > 1.5:
            anomalies.append({
                'segment': segment,
                'segment_type': segment_col,
                'current_rate': recent_rate,
                'historical_rate': hist_rate,
                'change': recent_rate - hist_rate,
                'z_score': z_score,
                'severity': 'High' if abs(z_score) > 2.5 else 'Medium' if abs(z_score) > 2.0 else 'Low',
                'direction': 'up' if z_score > 0 else 'down',
                'deals_affected': len(recent_seg)
            })
    
    return pd.DataFrame(anomalies)


def main():
    st.title("🚨 Anomaly Detection")
    st.caption("Identify unusual patterns in your sales pipeline")
    
    df = load_data()
    
    # =====================
    # CONTROLS
    # =====================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sensitivity = st.select_slider(
            "Detection Sensitivity",
            options=["Conservative", "Balanced", "Aggressive"],
            value="Balanced"
        )
        threshold = {"Conservative": 2.5, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
    
    with col2:
        metric_choice = st.selectbox(
            "Metric to Monitor",
            ["Win Rate", "Deal Count", "Avg Deal Size", "Avg Cycle Length", "Total Revenue"]
        )
        metric_key = metric_choice.lower().replace(" ", "_").replace("avg_", "avg_")
        if metric_key == "win_rate":
            pass
        elif metric_key == "deal_count":
            pass
        elif metric_key == "avg_deal_size":
            metric_key = "avg_deal_size"
        elif metric_key == "avg_cycle_length":
            metric_key = "avg_cycle"
        else:
            metric_key = "total_revenue"
    
    with col3:
        window_size = st.slider("Rolling Window (months)", 2, 6, 4)
    
    st.divider()
    
    # =====================
    # ACTIVE ALERTS
    # =====================
    st.markdown("### 🚨 Active Anomalies")
    
    # Calculate segment anomalies
    region_anomalies = get_segment_anomalies(df, 'region')
    industry_anomalies = get_segment_anomalies(df, 'industry')
    all_anomalies = pd.concat([region_anomalies, industry_anomalies], ignore_index=True)
    
    if len(all_anomalies) > 0:
        all_anomalies = all_anomalies.sort_values('z_score', key=abs, ascending=False)
        
        col1, col2, col3 = st.columns(3)
        
        high_alerts = all_anomalies[all_anomalies['severity'] == 'High']
        medium_alerts = all_anomalies[all_anomalies['severity'] == 'Medium']
        low_alerts = all_anomalies[all_anomalies['severity'] == 'Low']
        
        with col1:
            st.metric("🔴 High Severity", len(high_alerts))
        with col2:
            st.metric("🟠 Medium Severity", len(medium_alerts))
        with col3:
            st.metric("🟡 Low Severity", len(low_alerts))
        
        # Display alerts
        for _, alert in all_anomalies.head(5).iterrows():
            severity_emoji = "🔴" if alert['severity'] == 'High' else "🟠" if alert['severity'] == 'Medium' else "🟡"
            direction_emoji = "📈" if alert['direction'] == 'up' else "📉"
            
            color = "#f56565" if alert['severity'] == 'High' else "#ed8936" if alert['severity'] == 'Medium' else "#ecc94b"
            
            st.markdown(f"""
            <div style="background: #1a202c; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
                <strong>{severity_emoji} {alert['segment']} ({alert['segment_type'].title()})</strong> {direction_emoji}<br/>
                <span style="color: #e2e8f0;">
                    Win rate: {alert['current_rate']:.1f}% (was {alert['historical_rate']:.1f}%) — 
                    Change: <span style="color: {'#48bb78' if alert['change'] > 0 else '#f56565'};">{alert['change']:+.1f}pp</span>
                </span><br/>
                <small style="color: #718096;">{alert['deals_affected']} deals affected in recent period</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No significant anomalies detected!")
    
    st.divider()
    
    # =====================
    # METRIC TREND WITH ANOMALIES
    # =====================
    st.markdown(f"### 📊 {metric_choice} Trend with Anomaly Detection")
    
    monthly_data = calculate_rolling_anomalies(df, metric_key, window_size)
    
    fig = go.Figure()
    
    # Main metric line
    fig.add_trace(go.Scatter(
        x=monthly_data['closed_year_month'],
        y=monthly_data['value'],
        mode='lines+markers',
        name=metric_choice,
        line=dict(color='#4299e1', width=2),
        marker=dict(size=8)
    ))
    
    # Rolling average
    fig.add_trace(go.Scatter(
        x=monthly_data['closed_year_month'],
        y=monthly_data['rolling_mean'],
        mode='lines',
        name='Rolling Avg',
        line=dict(color='#805ad5', width=2, dash='dash')
    ))
    
    # Upper/Lower bounds
    upper_bound = monthly_data['rolling_mean'] + 2 * monthly_data['rolling_std']
    lower_bound = monthly_data['rolling_mean'] - 2 * monthly_data['rolling_std']
    
    fig.add_trace(go.Scatter(
        x=pd.concat([monthly_data['closed_year_month'], monthly_data['closed_year_month'][::-1]]),
        y=pd.concat([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 90, 213, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Normal Range (±2σ)'
    ))
    
    # Highlight anomalies
    anomaly_points = monthly_data[monthly_data['is_anomaly']]
    if len(anomaly_points) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_points['closed_year_month'],
            y=anomaly_points['value'],
            mode='markers',
            name='Anomaly',
            marker=dict(size=15, color='#f56565', symbol='x')
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title='Month',
        yaxis_title=metric_choice,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # =====================
    # SEGMENT DEVIATION HEATMAP
    # =====================
    st.markdown("### 🗺️ Segment Performance Deviation")
    st.caption("Shows how each segment's current performance compares to historical average")
    
    col1, col2 = st.columns(2)
    
    for col, seg_type in [(col1, 'region'), (col2, 'industry')]:
        with col:
            anomalies_df = get_segment_anomalies(df, seg_type)
            
            if len(anomalies_df) > 0:
                fig = px.bar(
                    anomalies_df.sort_values('change'),
                    x='change',
                    y='segment',
                    orientation='h',
                    color='change',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    labels={'change': 'Win Rate Change (pp)', 'segment': seg_type.title()},
                    title=f"{seg_type.title()} Deviations"
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info(f"No significant {seg_type} deviations detected")
    
    st.divider()
    
    # =====================
    # METHODOLOGY
    # =====================
    with st.expander("📝 Detection Methodology"):
        st.markdown("""
        ### How Anomalies Are Detected
        
        **Z-Score Method**
        - Calculate rolling mean and standard deviation
        - Flag points where |Z-score| > threshold
        - Default threshold: 2.0 (95% confidence)
        
        **Segment Comparison**
        - Compare recent 2-month performance vs historical average
        - Calculate deviation in standard deviation units
        - Flag segments with significant changes
        
        **Severity Levels**
        - 🔴 **High**: |Z-score| > 2.5 (typically < 1% chance random)
        - 🟠 **Medium**: |Z-score| > 2.0 (typically < 5% chance random)
        - 🟡 **Low**: |Z-score| > 1.5 (early warning)
        
        **Recommended Actions**
        1. Investigate high-severity anomalies immediately
        2. Set up alerts for critical metrics
        3. Review segment-specific issues with regional managers
        """)


if __name__ == "__main__":
    main()
