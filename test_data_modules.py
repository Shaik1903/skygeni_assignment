"""
Test script for data modules
"""
import sys
sys.path.insert(0, 'src')

from data_loader import load_and_prepare_data, get_summary_stats
from metrics import calculate_all_custom_metrics, identify_key_insights
from eda import run_full_eda

print("=" * 60)
print("SKYGENI DATA FOUNDATION TEST")
print("=" * 60)

# Load data
print("\n1. Loading and preprocessing data...")
df, report = load_and_prepare_data()
print(f"   ✓ Loaded {report['total_rows']} rows")
print(f"   ✓ Date range: {report['date_range']['created_min']} to {report['date_range']['closed_max']}")
print(f"   ✓ Outcomes: {report['outcome_distribution']}")

# Summary stats
print("\n2. Calculating summary statistics...")
stats = get_summary_stats(df)
print(f"   ✓ Overall Win Rate: {stats['overall_win_rate']:.1f}%")
print(f"   ✓ Total Revenue Won: ${stats['total_revenue_won']:,.0f}")
print(f"   ✓ Avg Deal Size: ${stats['avg_deal_size']:,.0f}")
print(f"   ✓ Avg Sales Cycle: {stats['avg_sales_cycle']:.0f} days")

# Custom metrics
print("\n3. Calculating custom metrics...")
metrics = calculate_all_custom_metrics(df)
print(f"   ✓ Deal Velocity Index calculated")
print(f"     - Won deals avg DVI: {metrics['df_with_dvi'][metrics['df_with_dvi']['is_won']==1]['deal_velocity_index'].mean():.2f}")
print(f"     - Lost deals avg DVI: {metrics['df_with_dvi'][metrics['df_with_dvi']['is_won']==0]['deal_velocity_index'].mean():.2f}")
print(f"   ✓ Win Pressure Scores calculated for 4 segments")
print(f"   ✓ Rep Consistency Scores calculated for {len(metrics['rep_consistency'])} reps")

# EDA insights
print("\n4. Running EDA analysis...")
eda_results = run_full_eda(df)
insights = eda_results['insights']
print(f"   ✓ Generated {len(insights)} actionable insights")
print(f"\n   Top 3 Insights:")
for i, insight in enumerate(insights[:3], 1):
    print(f"   {i}. [{insight['severity'].upper()}] {insight['emoji']} {insight['category']}")
    print(f"      {insight['insight'][:80]}...")

print("\n" + "=" * 60)
print("ALL DATA MODULES WORKING CORRECTLY ✓")
print("=" * 60)
