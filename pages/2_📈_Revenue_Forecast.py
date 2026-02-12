"""
Revenue Forecast Page
======================
Hybrid time series forecasting (Prophet + XGBoost) for revenue predictions.
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
from forecasting import get_forecast_pipeline_cached

st.set_page_config(page_title="Revenue Forecast", page_icon="📈", layout="wide")


@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df


def main():
    st.title("📈 Hybrid Revenue Forecast")
    st.caption("Advanced forecasting using Prophet (Trends) + XGBoost (Deal Drivers)")
    
    df = load_data()
    
    # =====================
    # CONTROLS
    # =====================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_weeks = st.slider("Forecast Horizon (Weeks)", 4, 24, 12)
    
    with col2:
        scenario = st.selectbox("Scenario", ["Most Likely", "Optimistic (+15%)", "Conservative (-15%)"])
    
    with col3:
        show_components = st.checkbox("Show Model Components", value=False)
    
    # Run Forecast Pipeline
    with st.spinner("Generating hybrid forecast..."):
        try:
            pipeline_results = get_forecast_pipeline_cached(df, forecast_weeks=forecast_weeks)
            forecast_df = pipeline_results["forecast"]["forecast_df"]
            historical_df = pipeline_results["forecast"]["historical_df"]
            cv_results = pipeline_results["forecast"]["cv_results"]
            shap = pipeline_results["shap_analysis"]
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")
            return

    # Apply Scenario Multiplier
    multiplier = 1.0
    if "Optimistic" in scenario: multiplier = 1.15
    if "Conservative" in scenario: multiplier = 0.85
    
    forecast_df["final_forecast"] = forecast_df["ensemble_forecast"] * multiplier
    forecast_df["final_lower"] = forecast_df["ensemble_lower"] * multiplier
    forecast_df["final_upper"] = forecast_df["ensemble_upper"] * multiplier
    
    # =====================
    # HEADLINE METRICS
    # =====================
    col1, col2, col3, col4 = st.columns(4)
    
    total_forecast = forecast_df["final_forecast"].sum()
    avg_weekly = forecast_df["final_forecast"].mean()
    
    with col1:
        st.metric(
            f"Total Forecast ({forecast_weeks} wks)",
            f"${total_forecast/1e6:.2f}M"
        )
    
    with col2:
        st.metric("Avg Weekly Revenue", f"${avg_weekly/1e3:.0f}K")
    
    with col3:
        # Trend calculation (last 4 weeks vs first 4 weeks of forecast)
        first_4 = forecast_df["final_forecast"].iloc[:4].mean()
        last_4 = forecast_df["final_forecast"].iloc[-4:].mean()
        trend_pct = (last_4 - first_4) / first_4 * 100
        
        st.metric(
            "Forecast Trend",
            f"{trend_pct:+.1f}%",
            delta_color="normal"
        )
    
    with col4:
        # Model Accuracy (MAPE from CV)
        mape = cv_results.get("mape_mean")
        if mape:
            st.metric("Model Error (MAPE)", f"{mape:.1%}", help="Mean Absolute Percentage Error from walk-forward validation")
        else:
            st.metric("Model Error", "N/A")
            
    st.divider()

    # =====================
    # FORECAST CHART
    # =====================
    st.markdown("### 📊 Weekly Revenue Forecast")
    
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=historical_df["week_start"],
        y=historical_df["revenue_won"],
        mode="lines+markers",
        name="Historical Revenue",
        line=dict(color="#4299e1", width=2),
        marker=dict(size=6, opacity=0.7)
    ))
    
    # Forecast Data
    fig.add_trace(go.Scatter(
        x=forecast_df["week_start"],
        y=forecast_df["final_forecast"],
        mode="lines+markers",
        name="Hybrid Forecast",
        line=dict(color="#48bb78", width=3, dash="dot"),
        marker=dict(size=6)
    ))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["week_start"], forecast_df["week_start"][::-1]]),
        y=pd.concat([forecast_df["final_upper"], forecast_df["final_lower"][::-1]]),
        fill='toself',
        fillcolor='rgba(72, 187, 120, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='80% Confidence Band',
        showlegend=True
    ))
    
    # Optional: Show Components
    if show_components:
        if "prophet_forecast" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["week_start"],
                y=forecast_df["prophet_forecast"] * multiplier,
                mode="lines",
                name="Prophet Trend",
                line=dict(color="orange", width=1, dash="dash"),
                opacity=0.6
            ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df["week_start"],
            y=forecast_df["xgb_forecast"] * multiplier,
            mode="lines",
            name="XGBoost Prediction",
            line=dict(color="purple", width=1, dash="dash"),
            opacity=0.6
        ))
        
    fig.update_layout(
        template='plotly_dark',
        height=500,
        hovermode="x unified",
        xaxis_title="Week",
        yaxis_title="Revenue ($)",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # =====================
    # DRIVERS & INSIGHTS
    # =====================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔍 Forecast Logic Explained")
        st.info("""
        **Hybrid Approach:**
        1. **Prophet Model:** Captures seasonality (e.g., end-of-quarter spikes) and long-term trends.
        2. **XGBoost Regressor:** Uses recent signals (pipeline velocity, win rates, momentum) to adjust the short-term forecast.
        
        The final forecast is a **weighted ensemble** of both models.
        """)
        
        # Validation stats
        with st.expander("See Model Validation Stats"):
            st.json(cv_results)
            
    with col2:
        st.markdown("### 🔑 Global Revenue Drivers")
        st.caption("Top factors influencing deal success (SHAP)")
        
        # Drivers Bar Chart (Condensed version of Page 1)
        features = [shap["feature_names"][i] for i in shap["sorted_idx"]][:8][::-1]
        importance = [shap["mean_abs_shap"][i] for i in shap["sorted_idx"]][:8][::-1]
        
        fig_drivers = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title=None,
            labels={'x': 'Impact', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='Bluered'
        )
        fig_drivers.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_drivers, width='stretch')

    st.divider()
    
    # =====================
    # FORECAST DATA TABLE
    # =====================
    with st.expander("📋 Detailed Forecast Data"):
        display_df = forecast_df.copy()
        display_df["Week"] = display_df["week_start"].dt.strftime("%Y-%m-%d")
        display_df = display_df[["Week", "final_forecast", "final_lower", "final_upper"]].rename(columns={
            "final_forecast": "Forecast ($)",
            "final_lower": "Lower Bound ($)",
            "final_upper": "Upper Bound ($)"
        })
        st.dataframe(display_df, width='stretch')


if __name__ == "__main__":
    main()
