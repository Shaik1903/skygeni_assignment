"""
Revenue Forecast Page
======================
Time series forecasting for revenue predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_and_prepare_data

st.set_page_config(page_title="Revenue Forecast", page_icon="📈", layout="wide")


@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df


def create_simple_forecast(monthly_data: pd.DataFrame, periods: int = 3):
    """
    Create a simple linear trend forecast.
    For production, use Prophet or ARIMA.
    """
    # Get historical trend
    monthly_data = monthly_data.copy()
    monthly_data['period_num'] = range(len(monthly_data))
    
    # Simple linear regression for trend
    x = monthly_data['period_num'].values
    y = monthly_data['revenue_won'].values
    
    # Calculate trend
    slope = np.polyfit(x, y, 1)[0]
    intercept = np.polyfit(x, y, 1)[1]
    
    # Generate forecast
    last_period = monthly_data['period_num'].max()
    forecast_periods = range(last_period + 1, last_period + periods + 1)
    
    forecast_values = [slope * p + intercept for p in forecast_periods]
    
    # Add confidence intervals (simple approach: based on historical std)
    std = monthly_data['revenue_won'].std()
    
    # Generate forecast dates
    last_date = pd.to_datetime(monthly_data['closed_year_month'].iloc[-1])
    forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast_values,
        'lower_80': [v - 1.28 * std for v in forecast_values],
        'upper_80': [v + 1.28 * std for v in forecast_values],
        'lower_95': [v - 1.96 * std for v in forecast_values],
        'upper_95': [v + 1.96 * std for v in forecast_values]
    })
    
    return forecast_df, slope


def main():
    st.title("📈 Revenue Forecast")
    st.caption("Predict future revenue using time series analysis")
    
    df = load_data()
    
    # Prepare monthly revenue data
    df['closed_year_month'] = df['closed_date'].dt.to_period('M').astype(str)
    
    monthly = df[df['is_won'] == 1].groupby('closed_year_month').agg(
        revenue_won=('deal_amount', 'sum'),
        deals_won=('deal_id', 'count'),
        avg_deal_size=('deal_amount', 'mean')
    ).reset_index()
    
    # =====================
    # CONTROLS
    # =====================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.slider("Forecast Horizon (Months)", 1, 6, 3)
    
    with col2:
        scenario = st.selectbox("Scenario", ["Most Likely", "Optimistic", "Conservative"])
    
    with col3:
        show_confidence = st.checkbox("Show Confidence Bands", value=True)
    
    # Scenario adjustments
    scenario_multiplier = {"Most Likely": 1.0, "Optimistic": 1.15, "Conservative": 0.85}[scenario]
    
    # Generate forecast
    forecast_df, trend_slope = create_simple_forecast(monthly, forecast_periods)
    forecast_df['forecast'] *= scenario_multiplier
    forecast_df['lower_80'] *= scenario_multiplier
    forecast_df['upper_80'] *= scenario_multiplier
    forecast_df['lower_95'] *= scenario_multiplier
    forecast_df['upper_95'] *= scenario_multiplier
    
    st.divider()
    
    # =====================
    # KEY METRICS
    # =====================
    col1, col2, col3, col4 = st.columns(4)
    
    total_forecast = forecast_df['forecast'].sum()
    last_period_revenue = monthly['revenue_won'].iloc[-1]
    
    with col1:
        st.metric(
            f"Forecasted Revenue ({forecast_periods}mo)",
            f"${total_forecast/1e6:.2f}M"
        )
    
    with col2:
        avg_monthly = forecast_df['forecast'].mean()
        st.metric("Avg Monthly Forecast", f"${avg_monthly/1e3:.0f}K")
    
    with col3:
        trend_direction = "📈 Growing" if trend_slope > 0 else "📉 Declining"
        st.metric("Trend", trend_direction)
    
    with col4:
        confidence_range = (forecast_df['upper_80'].iloc[-1] - forecast_df['lower_80'].iloc[-1]) / forecast_df['forecast'].iloc[-1] * 100
        st.metric("Uncertainty (80% CI)", f"±{confidence_range/2:.0f}%")
    
    st.divider()
    
    # =====================
    # FORECAST CHART
    # =====================
    st.markdown("### 📊 Revenue Forecast Chart")
    
    # Prepare historical data for chart
    historical = monthly.copy()
    historical['date'] = pd.to_datetime(historical['closed_year_month'])
    historical['type'] = 'Historical'
    
    # Create figure
    fig = go.Figure()
    
    # Historical line
    fig.add_trace(go.Scatter(
        x=historical['date'],
        y=historical['revenue_won'],
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color='#4299e1', width=3),
        marker=dict(size=8)
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#48bb78', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Confidence bands
    if show_confidence:
        # 95% CI
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
            y=pd.concat([forecast_df['upper_95'], forecast_df['lower_95'][::-1]]),
            fill='toself',
            fillcolor='rgba(72, 187, 120, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% Confidence'
        ))
        
        # 80% CI
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
            y=pd.concat([forecast_df['upper_80'], forecast_df['lower_80'][::-1]]),
            fill='toself',
            fillcolor='rgba(72, 187, 120, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='80% Confidence'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =====================
    # FORECAST TABLE
    # =====================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Monthly Forecast Details")
        
        display_df = forecast_df.copy()
        display_df['Month'] = display_df['date'].dt.strftime('%B %Y')
        display_df['Forecast'] = display_df['forecast'].apply(lambda x: f"${x/1e3:.0f}K")
        display_df['80% Range'] = display_df.apply(
            lambda r: f"${r['lower_80']/1e3:.0f}K - ${r['upper_80']/1e3:.0f}K", axis=1
        )
        
        st.dataframe(
            display_df[['Month', 'Forecast', '80% Range']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("### 📊 Scenario Comparison")
        
        scenarios = {
            "Conservative": forecast_df['forecast'].sum() * 0.85,
            "Most Likely": forecast_df['forecast'].sum(),
            "Optimistic": forecast_df['forecast'].sum() * 1.15
        }
        
        fig = px.bar(
            x=list(scenarios.values()),
            y=list(scenarios.keys()),
            orientation='h',
            labels={'x': 'Total Revenue', 'y': 'Scenario'},
            color=list(scenarios.keys()),
            color_discrete_map={
                'Conservative': '#f56565',
                'Most Likely': '#4299e1',
                'Optimistic': '#48bb78'
            }
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # =====================
    # ASSUMPTIONS
    # =====================
    with st.expander("📝 Forecast Assumptions & Methodology"):
        st.markdown("""
        ### Methodology
        - **Model**: Linear trend extrapolation (for production, recommend Prophet or ARIMA)
        - **Data**: Monthly aggregated won revenue
        - **Confidence Intervals**: Based on historical standard deviation
        
        ### Key Assumptions
        1. Historical trend continues at similar rate
        2. No major market disruptions or seasonality shifts
        3. Sales capacity remains constant
        4. Product/pricing mix stays consistent
        
        ### Limitations
        - Simple linear model doesn't capture complex patterns
        - No external factors (market conditions, competition) included
        - Limited historical data may reduce accuracy
        
        ### Recommended Actions
        - Review forecast monthly and adjust based on pipeline changes
        - Consider scenario planning for strategic decisions
        - Monitor leading indicators (pipeline, conversion rates)
        """)


if __name__ == "__main__":
    main()
