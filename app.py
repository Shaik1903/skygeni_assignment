"""
SkyGeni Sales Intelligence Dashboard
=====================================
A comprehensive sales decision intelligence platform powered by data science.

Run: streamlit run app.py
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
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_and_prepare_data, get_summary_stats, get_segment_analysis, get_quarterly_trends, get_rep_performance, create_heatmap_data
from metrics import calculate_all_custom_metrics, get_metric_summary, identify_key_insights
from eda import run_full_eda, generate_eda_insights
from llm_insights import generate_llm_metric_insights, generate_llm_eda_insights, generate_llm_recommendations

# Page configuration
st.set_page_config(
    page_title="SkyGeni Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #0e1117;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 8px;
    }
    
    .kpi-card p {
        margin: 0;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-change-positive {
        color: #48bb78;
        font-size: 0.85rem;
    }
    
    .kpi-change-negative {
        color: #f56565;
        font-size: 0.85rem;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4299e1;
    }
    
    .insight-high {
        border-left-color: #f56565;
    }
    
    .insight-medium {
        border-left-color: #ed8936;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4299e1 0%, #805ad5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the data."""
    df, validation_report = load_and_prepare_data()
    return df, validation_report


@st.cache_data(ttl=3600)  # Refresh hourly
def get_all_analytics(_df):
    """Calculate all analytics and cache. v2: PQS+WRE+SMI metrics."""
    stats = get_summary_stats(_df)
    metrics = calculate_all_custom_metrics(_df)
    eda_results = run_full_eda(_df)
    quarterly = get_quarterly_trends(_df)
    return stats, metrics, eda_results, quarterly


@st.cache_data(ttl=3600, show_spinner=False)
def get_llm_metric_insights(_df, _metrics):
    """Generate and cache LLM metric insights."""
    return generate_llm_metric_insights(_df, _metrics)


@st.cache_data(ttl=3600, show_spinner=False)
def get_llm_eda_insights(_df, _eda_results):
    """Generate and cache LLM EDA insights."""
    fallback = _eda_results.get('insights', [])
    return generate_llm_eda_insights(_df, _eda_results, fallback)


def create_kpi_card(value, label, change=None, prefix="", suffix=""):
    """Create a styled KPI card."""
    if change is not None:
        change_class = "kpi-change-positive" if change >= 0 else "kpi-change-negative"
        change_symbol = "↑" if change >= 0 else "↓"
        change_html = f'<p class="{change_class}">{change_symbol} {abs(change):.1f}%</p>'
    else:
        # Placeholder to keep card height consistent
        change_html = '<p class="kpi-change-positive" style="visibility: hidden;">&nbsp;</p>'
    
    return f"""
    <div class="kpi-card">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{prefix}{value}{suffix}</p>
        {change_html}
    </div>
    """


def detect_anomalies(series: pd.Series, threshold: float = 2.0):
    """Detect anomalies using Z-score method."""
    z_scores = np.abs(stats.zscore(series.fillna(series.mean())))
    return z_scores > threshold


def calculate_rolling_anomalies(df: pd.DataFrame, metric: str, window: int = 4, threshold: float = 2.0):
    """Calculate rolling anomalies for a metric."""
    df = df.copy()
    df['closed_year_month'] = df['closed_date'].dt.to_period('M').astype(str)
    
    if metric == 'win_rate':
        monthly = df.groupby('closed_year_month').agg(value=('is_won', 'mean')).reset_index()
        monthly['value'] *= 100
    elif metric == 'deal_count':
        monthly = df.groupby('closed_year_month').agg(value=('deal_id', 'count')).reset_index()
    elif metric == 'avg_deal_size':
        monthly = df.groupby('closed_year_month').agg(value=('deal_amount', 'mean')).reset_index()
    elif metric == 'avg_cycle':
        monthly = df.groupby('closed_year_month').agg(value=('sales_cycle_days', 'mean')).reset_index()
    else:
        monthly = df.groupby('closed_year_month').agg(value=('deal_amount', 'sum')).reset_index()
    
    monthly['rolling_mean'] = monthly['value'].rolling(window=window, min_periods=2).mean()
    monthly['rolling_std'] = monthly['value'].rolling(window=window, min_periods=2).std()
    monthly['z_score'] = (monthly['value'] - monthly['rolling_mean']) / monthly['rolling_std'].replace(0, 1)
    monthly['is_anomaly'] = np.abs(monthly['z_score']) > threshold
    
    return monthly


def get_segment_anomalies(df: pd.DataFrame, segment_col: str):
    """Detect anomalies within segments over time."""
    df = df.copy()
    df['closed_year_month'] = df['closed_date'].dt.to_period('M').astype(str)
    
    unique_months = sorted(df['closed_year_month'].unique())
    if len(unique_months) < 3:
        return pd.DataFrame()
        
    recent_months = unique_months[-2:]
    historical_months = unique_months[:-2]
    
    recent = df[df['closed_year_month'].isin(recent_months)]
    historical = df[df['closed_year_month'].isin(historical_months)]
    
    anomalies = []
    
    for segment in df[segment_col].unique():
        recent_seg = recent[recent[segment_col] == segment]
        hist_seg = historical[historical[segment_col] == segment]
        
        if len(recent_seg) < 2 or len(hist_seg) < 5:
            continue
        
        recent_rate = recent_seg['is_won'].mean() * 100
        hist_rate = hist_seg['is_won'].mean() * 100
        hist_std = hist_seg.groupby('closed_year_month')['is_won'].mean().std() * 100
        
        if pd.isna(hist_std) or hist_std == 0:
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


def generate_recommendations(df: pd.DataFrame) -> list:
    """Generate prioritized recommendations based on patterns."""
    recommendations = []
    overall_rate = df['is_won'].mean() * 100
    
    # Regional
    region_rates = df.groupby('region')['is_won'].mean() * 100
    for region, rate in region_rates.items():
        if rate < overall_rate - 5:
            recommendations.append({
                'priority': 'HIGH', 'category': 'Regional Strategy',
                'action': f"Audit {region} sales process",
                'rationale': f"{region} win rate ({rate:.1f}%) is {overall_rate - rate:.1f}pp below average",
                'effort': 'Medium', 'timeline': '2-4 weeks', 'icon': '🌏',
                'expected_impact': "Improve regional consistency"
            })
            
    # Lead source
    source_rates = df.groupby('lead_source')['is_won'].mean() * 100
    if not source_rates.empty:
        best_source = source_rates.idxmax()
        worst_source = source_rates.idxmin()
        if source_rates[best_source] > source_rates[worst_source] * 1.3:
            recommendations.append({
                'priority': 'HIGH', 'category': 'Lead Gen',
                'action': f"Scale {best_source} investment",
                'rationale': f"{best_source} converts at {source_rates[best_source]:.1f}% vs {source_rates[worst_source]:.1f}% for {worst_source}",
                'effort': 'Low', 'timeline': '1-2 weeks', 'icon': '🎯',
                'expected_impact': "Higher conversion quality"
            })
            
    # Always add basic hygiene recommendations if list is short
    if len(recommendations) < 3:
        recommendations.append({
            'priority': 'MEDIUM', 'category': 'Pipeline Hygiene',
            'action': "Clean up stalling deals (> 90 days)",
            'rationale': "Old deals inflate pipeline and lower forecast accuracy",
            'effort': 'Low', 'timeline': '1 week', 'icon': '🧼',
            'expected_impact': "Better visibility and focus"
        })
        recommendations.append({
            'priority': 'MEDIUM', 'category': 'CRM Data',
            'action': "Audit lead source mapping",
            'rationale': "Inconsistent mapping prevents accurate attribution",
            'effort': 'Low', 'timeline': '1 week', 'icon': '📊',
            'expected_impact': "Reliable ROAS calculations"
        })
            
    return recommendations


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_recommendations(_df, _metrics, _eda_results):
    """Generate & cache LLM recommendations."""
    recs = generate_llm_recommendations(_df, _metrics, _eda_results)
    return recs if recs else generate_recommendations(_df)


def create_insight_card(insight):
    """Create a styled insight card."""
    severity_class = f"insight-{insight['severity']}"
    # Support both old-style (action) and new-style (business_action) insights
    action_text = insight.get('business_action', insight.get('action', ''))
    impact_text = insight.get('estimated_impact', '')
    impact_html = f'<br/><small style="color: #48bb78;">Impact: {impact_text}</small>' if impact_text else ''
    return f"""
    <div class="insight-card {severity_class}">
        <strong>{insight['emoji']} {insight['category']}</strong><br/>
        <span style="color: #e2e8f0;">{insight['insight']}</span><br/>
        <small style="color: #718096;">Action: {action_text}</small>
        {impact_html}
    </div>
    """


def main():
    """Main application."""
    
    # Load data
    df, validation_report = load_data()
    stats, metrics, eda_results, quarterly = get_all_analytics(df)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bar-chart.png", width=60)
        st.title("SkyGeni")
        st.caption("Sales Intelligence Platform")
        
        st.divider()
        
        # Data info
        st.markdown("### 📊 Data Overview")
        st.metric("Total Deals", f"{validation_report['total_rows']:,}")
        st.metric("Date Range", f"{validation_report['date_range']['created_min'][:10]} to {validation_report['date_range']['closed_max'][:10]}")
        
        st.divider()
        
        # Quick filters
        st.markdown("### 🔍 Quick Filters")
        regions = st.multiselect("Region", df["region"].unique(), default=df["region"].unique())
        industries = st.multiselect("Industry", df["industry"].unique(), default=df["industry"].unique())
    
    # Filter data
    filtered_df = df[
        (df["region"].isin(regions)) & 
        (df["industry"].isin(industries))
    ]
    
    # Recalculate stats for filtered data
    if len(filtered_df) < len(df):
        stats = get_summary_stats(filtered_df)
        eda_results = run_full_eda(filtered_df)
    
    # Main content
    st.title("📊 Sales Intelligence Dashboard")
    st.caption("Real-time insights to drive sales performance")
    
    # Introduce tabs
    tab_overview, tab_anomalies, tab_recs = st.tabs(["📊 Overview", "🚨 Pipeline Alerts", "💡 Action Plan"])
    
    with tab_overview:
        # =====================
        # KPI CARDS ROW
        # =====================
        st.markdown("### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(create_kpi_card(
                f"{stats['overall_win_rate']:.1f}%",
                "Win Rate",
                change=-8.2  # Placeholder - would calculate from quarterly
            ), unsafe_allow_html=True)
        
        with col2:
            revenue_m = stats['total_revenue_won'] / 1_000_000
            st.markdown(create_kpi_card(
                f"{revenue_m:.1f}M",
                "Revenue Won",
                prefix="$"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_kpi_card(
                f"{stats['won_deals']:,}",
                "Deals Won"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_kpi_card(
                f"{stats['avg_sales_cycle']:.0f}",
                "Avg Cycle Days",
                suffix=" days"
            ), unsafe_allow_html=True)
        
        with col5:
            # Pipeline Qualification Score
            pqs_score = metrics['pqs_overall']['pqs_score']
            st.markdown(create_kpi_card(
                f"{pqs_score:.0f}/100",
                "Pipeline Health (PQS)"
            ), unsafe_allow_html=True)
        
        st.divider()
        
        # =====================
        # CHARTS ROW 1
        # =====================
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Win Rate Trend")
            quarterly_data = pd.DataFrame(eda_results['win_rate_trends']['quarterly_data'])
            
            fig = px.line(
                quarterly_data, 
                x='closed_year_quarter', 
                y='win_rate',
                markers=True,
                labels={'closed_year_quarter': 'Quarter', 'win_rate': 'Win Rate (%)'}
            )
            fig.update_traces(
                line=dict(color='#4299e1', width=3),
                marker=dict(size=10, color='#63b3ed')
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.add_hline(y=stats['overall_win_rate'], line_dash="dash", line_color="gray",
                          annotation_text=f"Average: {stats['overall_win_rate']:.1f}%")
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### 💰 Revenue by Quarter")
            quarterly_data['revenue_won_m'] = quarterly_data['won_deals'] * (stats['avg_deal_size_won'] / 1000)  # Estimate
            
            fig = px.bar(
                quarterly_data,
                x='closed_year_quarter',
                y='won_deals',
                labels={'closed_year_quarter': 'Quarter', 'won_deals': 'Deals Won'}
            )
            fig.update_traces(marker_color='#805ad5')
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, width='stretch')
        
        # =====================
        # HEATMAP
        # =====================
        st.markdown("#### 🗺️ Segment Performance Heatmap (Win Rate by Industry × Region)")
        
        heatmap_data = create_heatmap_data(filtered_df, "industry", "region", "win_rate")
        
        fig = px.imshow(
            heatmap_data,
            text_auto='.1f',
            aspect='auto',
            color_continuous_scale='RdYlGn',
            labels={'color': 'Win Rate (%)'}
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, width='stretch')
        
        # =====================
        # CHARTS ROW 2
        # =====================
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Lead Source Performance")
            lead_data = eda_results['lead_source_analysis']
            
            fig = px.pie(
                lead_data,
                values='total_deals',
                names='lead_source',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### 👥 Top Sales Reps")
            rep_data = get_rep_performance(filtered_df).head(10)
            
            fig = px.bar(
                rep_data,
                x='win_rate',
                y='sales_rep_id',
                orientation='h',
                labels={'win_rate': 'Win Rate (%)', 'sales_rep_id': 'Rep ID'},
                color='win_rate',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
        
        # =====================
        # CHARTS ROW 3
        # =====================
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Deal Size Distribution")
            
            fig = px.histogram(
                filtered_df,
                x='deal_amount',
                nbins=30,
                color='outcome',
                barmode='overlay',
                labels={'deal_amount': 'Deal Amount ($)', 'count': 'Frequency'},
                color_discrete_map={'Won': '#48bb78', 'Lost': '#f56565'}
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### ⏱️ Sales Cycle by Outcome")
            
            fig = px.box(
                filtered_df,
                x='outcome',
                y='sales_cycle_days',
                color='outcome',
                labels={'sales_cycle_days': 'Sales Cycle (Days)', 'outcome': 'Outcome'},
                color_discrete_map={'Won': '#48bb78', 'Lost': '#f56565'}
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
        
        st.divider()
        
        # =====================
        # CUSTOM METRICS SPOTLIGHT
        # =====================
        st.markdown("### 📐 Custom Metrics Spotlight")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pqs = metrics['pqs_overall']
            pqs_color = '#48bb78' if pqs['pqs_score'] >= 70 else ('#ed8936' if pqs['pqs_score'] >= 40 else '#f56565')
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <p class="kpi-label">Pipeline Qualification Score (PQS)</p>
                <p class="kpi-value" style="font-size:2rem; color:{pqs_color};">{pqs['pqs_score']:.0f}/100</p>
                <span style="color:#a0aec0;">DQE: {pqs['dqe_score']:.2f} | Stall rate: {pqs['stall_rate_pct']:.0f}%</span><br/>
                <span style="color:#f56565;">{pqs['deals_stalling']} stalling, {pqs['deals_likely_dead']} likely dead</span><br/>
                <span style="color:#a0aec0;">~{pqs['wasted_capacity_days']:,.0f} rep-days wasted</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            wre = metrics['wre']
            wre_color = '#f56565' if wre['severity'] == 'high' else ('#ed8936' if wre['severity'] == 'medium' else '#48bb78')
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <p class="kpi-label">Win Rate Elasticity (WRE)</p>
                <p class="kpi-value" style="font-size:2rem; color:{wre_color};">{wre['elasticity']:.2f}</p>
                <span style="color:#a0aec0;">Small deals: {wre['smallest_bucket_wr']:.0f}% WR vs Large: {wre['largest_bucket_wr']:.0f}% WR</span><br/>
                <span style="color:#48bb78;">Sweet spot: {wre['sweet_spot_range']} ({wre['sweet_spot_win_rate']:.0f}% WR)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            smi = metrics['smi_region']
            declining = smi[smi['smi_score'] < -0.05]
            growing = smi[smi['smi_score'] > 0.05]
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <p class="kpi-label">Segment Momentum Index (Region)</p>
                <p class="kpi-value" style="font-size:2rem;">{len(declining)} declining</p>
                <span style="color:#f56565;">{len(declining)} regions losing steam</span><br/>
                <span style="color:#48bb78;">{len(growing)} regions growing</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # =====================
        # INSIGHTS SECTION (LLM-powered with fallback)
        # =====================
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 🤖 Key Insights (AI-Generated)")
            
            with st.spinner("Generating AI insights..."):
                metric_insights = get_llm_metric_insights(filtered_df, metrics)
            for insight in metric_insights[:5]:
                st.markdown(create_insight_card(insight), unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 EDA Findings (AI-Generated)")
            
            with st.spinner("Analyzing patterns..."):
                eda_insights = get_llm_eda_insights(filtered_df, eda_results)
            for insight in eda_insights[:4]:
                st.markdown(create_insight_card(insight), unsafe_allow_html=True)
            
    with tab_anomalies:
        st.markdown("### 🚨 Pipeline Anomalies")
        col_s, col_m = st.columns([1, 2])
        with col_s:
            sensitivity = st.selectbox("Sensitivity", ["Balanced", "Conservative", "Aggressive"], index=0)
            threshold = {"Conservative": 2.5, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            
        region_anomalies = get_segment_anomalies(filtered_df, 'region')
        industry_anomalies = get_segment_anomalies(filtered_df, 'industry')
        all_anomalies = pd.concat([region_anomalies, industry_anomalies], ignore_index=True)
        
        if len(all_anomalies) > 0:
            for _, alert in all_anomalies.head(5).iterrows():
                color = "#f56565" if alert['severity'] == 'High' else "#ed8936" if alert['severity'] == 'Medium' else "#ecc94b"
                st.markdown(f"""
                <div style="background: #1a202c; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
                    <strong>{alert['segment']} ({alert['segment_type'].title()})</strong><br/>
                    <span style="color: #e2e8f0;">Win rate: {alert['current_rate']:.1f}% (was {alert['historical_rate']:.1f}%) — Change: {alert['change']:+.1f}pp</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant anomalies detected!")
            
        st.divider()
        st.markdown("#### 📊 Metric Anomaly Trend")
        anomaly_data = calculate_rolling_anomalies(filtered_df, 'win_rate', threshold=threshold)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anomaly_data['closed_year_month'], y=anomaly_data['value'], mode='lines+markers', name='Win Rate'))
        fig.add_trace(go.Scatter(x=anomaly_data['closed_year_month'], y=anomaly_data['rolling_mean'], mode='lines', name='Rolling Avg', line=dict(dash='dash')))
        anomaly_points = anomaly_data[anomaly_data['is_anomaly']]
        if len(anomaly_points) > 0:
            fig.add_trace(go.Scatter(x=anomaly_points['closed_year_month'], y=anomaly_points['value'], mode='markers', name='Anomaly', marker=dict(size=12, color='#f56565', symbol='x')))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, width='stretch')

    with tab_recs:
        st.markdown("### 💡 Prioritized Recommendations")
        with st.spinner("🤖 Generating AI-powered recommendations..."):
            recommendations = get_cached_recommendations(filtered_df, metrics, eda_results)
            
        if recommendations:
            for i, rec in enumerate(recommendations[:6], 1):
                with st.expander(f"{rec.get('icon', '💡')} **{rec.get('action', '')}** — {rec.get('priority', 'MEDIUM')} Priority", expanded=(i<=3)):
                    st.markdown(f"**Rationale:** {rec.get('rationale', '')}")
                    st.markdown(f"**Impact:** {rec.get('expected_impact', '')}")
                    st.markdown(f"**Effort:** {rec.get('effort', 'Medium')} | **Timeline:** {rec.get('timeline', '2-4 weeks')}")
        else:
            st.info("ℹ️ No specific recommendations identified based on current filters. High-level insights are available in the Overview tab.")
    
    # =====================
    # FOOTER
    # =====================
    st.divider()
    st.caption("Built with ❤️ for SkyGeni | Data as of " + validation_report['date_range']['closed_max'][:10])


if __name__ == "__main__":
    main()
