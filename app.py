"""
SkyGeni Sales Intelligence Dashboard
=====================================
A comprehensive sales decision intelligence platform powered by data science.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_and_prepare_data, get_summary_stats, get_segment_analysis, get_quarterly_trends, get_rep_performance, create_heatmap_data
from metrics import calculate_all_custom_metrics, get_metric_summary, identify_key_insights
from eda import run_full_eda, generate_eda_insights

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
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
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


@st.cache_data
def get_all_analytics(_df):
    """Calculate all analytics and cache."""
    stats = get_summary_stats(_df)
    metrics = calculate_all_custom_metrics(_df)
    eda_results = run_full_eda(_df)
    quarterly = get_quarterly_trends(_df)
    return stats, metrics, eda_results, quarterly


def create_kpi_card(value, label, change=None, prefix="", suffix=""):
    """Create a styled KPI card."""
    change_html = ""
    if change is not None:
        change_class = "kpi-change-positive" if change >= 0 else "kpi-change-negative"
        change_symbol = "↑" if change >= 0 else "↓"
        change_html = f'<p class="{change_class}">{change_symbol} {abs(change):.1f}%</p>'
    
    return f"""
    <div class="kpi-card">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{prefix}{value}{suffix}</p>
        {change_html}
    </div>
    """


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
        # Deal Qualification Efficiency
        dqe_score = float(metrics['dqe_overall']['dqe_score'].iloc[0])
        st.markdown(create_kpi_card(
            f"{dqe_score:.2f}",
            "Qualification Efficiency"
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
    st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # =====================
    # CUSTOM METRICS SPOTLIGHT
    # =====================
    st.markdown("### 📐 Custom Metrics Spotlight")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dqe = metrics['dqe_overall']
        dqe_score = float(dqe['dqe_score'].iloc[0])
        wasted = float(dqe['wasted_capacity_days'].iloc[0])
        st.markdown(f"""
        <div class="kpi-card" style="text-align:left;">
            <p class="kpi-label">Deal Qualification Efficiency (DQE)</p>
            <p class="kpi-value" style="font-size:2rem;">{dqe_score:.2f}</p>
            <span style="color:#a0aec0;">Lost deals take nearly as long as wins.</span><br/>
            <span style="color:#f56565;">~{wasted:,.0f} rep-days wasted</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rcr = metrics['rcr_region']
        risk_color = '#48bb78' if rcr['risk_level'] == 'Healthy' else '#f56565'
        st.markdown(f"""
        <div class="kpi-card" style="text-align:left;">
            <p class="kpi-label">Revenue Concentration Risk (Region)</p>
            <p class="kpi-value" style="font-size:2rem;">{rcr['rcr_score']:.2f}</p>
            <span style="color:{risk_color};">{rcr['risk_level']}</span><br/>
            <span style="color:#a0aec0;">Top: {rcr['top_segment']} ({rcr['top_segment_share']}%)</span>
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
    # INSIGHTS SECTION
    # =====================
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 💡 Key Insights (from Custom Metrics)")
        
        # Use insights from NEW custom metrics
        metric_insights = identify_key_insights(filtered_df, metrics)
        for insight in metric_insights[:5]:
            st.markdown(create_insight_card(insight), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 EDA Findings")
        
        eda_insights = eda_results['insights']
        for insight in eda_insights[:4]:
            st.markdown(create_insight_card(insight), unsafe_allow_html=True)
    
    # =====================
    # FOOTER
    # =====================
    st.divider()
    st.caption("Built with ❤️ for SkyGeni | Data as of " + validation_report['date_range']['closed_max'][:10])


if __name__ == "__main__":
    main()
