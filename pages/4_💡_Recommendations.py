"""
Recommendations Page
=====================
Actionable insights and prioritized recommendations.
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

from data_loader import load_and_prepare_data, get_summary_stats
from metrics import calculate_all_custom_metrics, identify_key_insights
from eda import generate_eda_insights, run_full_eda
from llm_insights import generate_llm_recommendations

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")


@st.cache_data
def load_data():
    df, _ = load_and_prepare_data()
    return df


def generate_recommendations(df: pd.DataFrame, insights: list) -> list:
    """Generate prioritized recommendations based on insights."""
    recommendations = []
    
    # 1. Regional performance recommendations
    region_rates = df.groupby('region')['is_won'].mean() * 100
    overall_rate = df['is_won'].mean() * 100
    
    for region, rate in region_rates.items():
        if rate < overall_rate - 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Regional Strategy',
                'action': f"Audit {region} sales process",
                'rationale': f"{region} win rate ({rate:.1f}%) is {overall_rate - rate:.1f}pp below average",
                'expected_impact': f"Could improve overall win rate by {(overall_rate - rate) * df[df['region']==region].shape[0] / len(df) * 0.5:.1f}pp",
                'effort': 'Medium',
                'timeline': '2-4 weeks',
                'icon': '🌏'
            })
    
    # 2. Lead source optimization
    source_rates = df.groupby('lead_source')['is_won'].mean() * 100
    best_source = source_rates.idxmax()
    worst_source = source_rates.idxmin()
    
    if source_rates[best_source] > source_rates[worst_source] * 1.3:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Lead Generation',
            'action': f"Increase {best_source} lead investment",
            'rationale': f"{best_source} converts at {source_rates[best_source]:.1f}% vs {source_rates[worst_source]:.1f}% for {worst_source}",
            'expected_impact': f"Shifting 20% budget could add {((source_rates[best_source] - source_rates[worst_source]) * 0.2):.1f}pp to win rate",
            'effort': 'Low',
            'timeline': '1-2 weeks',
            'icon': '🎯'
        })
    
    # 3. Sales cycle optimization
    won_cycle = df[df['is_won'] == 1]['sales_cycle_days'].mean()
    lost_cycle = df[df['is_won'] == 0]['sales_cycle_days'].mean()
    
    if lost_cycle > won_cycle + 15:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Process Improvement',
            'action': "Implement deal velocity checkpoints",
            'rationale': f"Lost deals average {lost_cycle:.0f} days vs {won_cycle:.0f} days for wins",
            'expected_impact': "Early stall detection could save 20-30% of struggling deals",
            'effort': 'Medium',
            'timeline': '2-3 weeks',
            'icon': '⏱️'
        })
    
    # 4. Rep coaching
    rep_rates = df.groupby('sales_rep_id')['is_won'].mean() * 100
    underperformers = rep_rates[rep_rates < overall_rate - 10]
    
    if len(underperformers) > 0:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Team Development',
            'action': f"Coach {len(underperformers)} underperforming reps",
            'rationale': f"These reps are 10+ pp below average win rate",
            'expected_impact': f"Bringing to average could add ${len(underperformers) * 50000:,.0f}+ in revenue",
            'effort': 'High',
            'timeline': '4-8 weeks',
            'icon': '👥'
        })
    
    # 5. Product mix optimization
    product_rates = df.groupby('product_type')['is_won'].mean() * 100
    best_product = product_rates.idxmax()
    
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Product Strategy',
        'action': f"Bundle or upsell {best_product} offerings",
        'rationale': f"{best_product} has highest win rate at {product_rates[best_product]:.1f}%",
        'expected_impact': "Could improve attach rate and overall deal value",
        'effort': 'Low',
        'timeline': '1-2 weeks',
        'icon': '📦'
    })
    
    # 6. Deal sizing
    size_rates = df.groupby('deal_size_category', observed=True)['is_won'].mean() * 100
    best_size = size_rates.idxmax()
    
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Deal Strategy',
        'action': f"Focus on {best_size} deals",
        'rationale': f"This segment has {size_rates[best_size]:.1f}% win rate",
        'expected_impact': "Better resource allocation and higher conversion",
        'effort': 'Low',
        'timeline': '1 week',
        'icon': '💰'
    })
    
    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 2))
    
    return recommendations

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_recommendations(_df, _metrics, _eda_results):
    """Generate & cache LLM recommendations."""
    recs = generate_llm_recommendations(
        _df, _metrics, _eda_results,
        fallback_fn=lambda d, _: generate_recommendations(d, [])
    )
    return recs if recs else generate_recommendations(_df, [])


def main():
    st.title("💡 Recommendations")
    st.caption("Prioritized actions to improve sales performance")
    
    df = load_data()
    stats = get_summary_stats(df)
    metrics = calculate_all_custom_metrics(df)
    eda_results = run_full_eda(df)
    
    with st.spinner("🤖 Generating AI-powered recommendations..."):
        recommendations = get_cached_recommendations(df, metrics, eda_results)
    
    # =====================
    # SUMMARY METRICS
    # =====================
    col1, col2, col3, col4 = st.columns(4)
    
    high_priority = len([r for r in recommendations if r['priority'] == 'HIGH'])
    medium_priority = len([r for r in recommendations if r['priority'] == 'MEDIUM'])
    
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    with col2:
        st.metric("🔴 High Priority", high_priority)
    with col3:
        st.metric("🟠 Medium Priority", medium_priority)
    with col4:
        potential_lift = sum([1.5 if r['priority'] == 'HIGH' else 0.5 for r in recommendations])
        st.metric("Potential Win Rate Lift", f"+{potential_lift:.1f}pp")
    
    st.divider()
    
    # =====================
    # PRIORITY FILTER
    # =====================
    priority_filter = st.multiselect(
        "Filter by Priority",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM"]
    )
    
    filtered_recs = [r for r in recommendations if r['priority'] in priority_filter]
    
    # =====================
    # RECOMMENDATIONS CARDS
    # =====================
    st.markdown("### 🎯 Action Plan")
    
    for i, rec in enumerate(filtered_recs, 1):
        priority_color = {
            'HIGH': '#f56565',
            'MEDIUM': '#ed8936',
            'LOW': '#48bb78'
        }.get(rec['priority'], '#4299e1')
        
        effort_emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(rec['effort'], '⚪')
        
        with st.expander(f"{rec['icon']} **{rec['action']}** — {rec['priority']} Priority", expanded=(i <= 3)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **Category:** {rec['category']}
                
                **Why:** {rec['rationale']}
                
                **Expected Impact:** {rec['expected_impact']}
                """)
            
            with col2:
                st.markdown(f"""
                **Effort:** {effort_emoji} {rec['effort']}
                
                **Timeline:** {rec['timeline']}
                """)
    
    st.divider()
    
    # =====================
    # IMPACT MATRIX
    # =====================
    st.markdown("### 📊 Impact vs Effort Matrix")
    
    # Convert to numeric for plotting
    effort_map = {'Low': 1, 'Medium': 2, 'High': 3}
    priority_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    
    matrix_data = pd.DataFrame([{
        'action': r['action'][:30] + '...' if len(r['action']) > 30 else r['action'],
        'effort': effort_map.get(r['effort'], 2),
        'impact': priority_map.get(r['priority'], 2),
        'priority': r['priority'],
        'category': r['category']
    } for r in recommendations])
    
    fig = px.scatter(
        matrix_data,
        x='effort',
        y='impact',
        color='priority',
        size=[20] * len(matrix_data),
        text='action',
        color_discrete_map={'HIGH': '#f56565', 'MEDIUM': '#ed8936', 'LOW': '#48bb78'},
        labels={'effort': 'Effort Required', 'impact': 'Potential Impact'}
    )
    
    fig.update_traces(textposition='top center', textfont_size=10)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis=dict(tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        yaxis=dict(tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        showlegend=True
    )
    
    # Add quadrant annotations
    fig.add_annotation(x=1, y=3, text="Quick Wins ⭐", showarrow=False, font=dict(size=12, color='#48bb78'))
    fig.add_annotation(x=3, y=3, text="Strategic", showarrow=False, font=dict(size=12, color='#4299e1'))
    fig.add_annotation(x=1, y=1, text="Low Priority", showarrow=False, font=dict(size=12, color='#718096'))
    fig.add_annotation(x=3, y=1, text="Reconsider", showarrow=False, font=dict(size=12, color='#718096'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # =====================
    # IMPLEMENTATION TIMELINE
    # =====================
    st.markdown("### 📅 Suggested Implementation Timeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Week 1-2: Quick Wins")
        quick_wins = [r for r in recommendations if r['effort'] == 'Low']
        for r in quick_wins:
            st.markdown(f"- {r['icon']} {r['action']}")
    
    with col2:
        st.markdown("#### Week 3-4: Medium Effort")
        medium_effort = [r for r in recommendations if r['effort'] == 'Medium']
        for r in medium_effort:
            st.markdown(f"- {r['icon']} {r['action']}")
    
    with col3:
        st.markdown("#### Month 2+: Strategic")
        high_effort = [r for r in recommendations if r['effort'] == 'High']
        for r in high_effort:
            st.markdown(f"- {r['icon']} {r['action']}")
    
    st.divider()
    
    # =====================
    # EXPORT
    # =====================
    st.markdown("### 📥 Export Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create export dataframe
        export_df = pd.DataFrame(recommendations)
        csv = export_df.to_csv(index=False)
        st.download_button(
            "📄 Download as CSV",
            csv,
            "skygeni_recommendations.csv",
            "text/csv"
        )
    
    with col2:
        # Summary text
        summary = f"""
# SkyGeni Sales Intelligence - Recommendations Summary

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Key Metrics
- Current Win Rate: {stats['overall_win_rate']:.1f}%
- Total Recommendations: {len(recommendations)}
- High Priority Actions: {high_priority}

## Top 3 Recommendations

"""
        for i, r in enumerate(recommendations[:3], 1):
            summary += f"{i}. **{r['action']}** ({r['priority']})\n   - {r['rationale']}\n   - Impact: {r['expected_impact']}\n\n"
        
        st.download_button(
            "📝 Download Summary (Markdown)",
            summary,
            "skygeni_recommendations_summary.md",
            "text/markdown"
        )


if __name__ == "__main__":
    main()
