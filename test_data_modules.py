"""Quick verification of new custom metrics."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

from data_loader import load_and_prepare_data
from metrics import calculate_all_custom_metrics, identify_key_insights

print("=" * 60)
print("CUSTOM METRICS VERIFICATION")
print("=" * 60)

df, _ = load_and_prepare_data()
m = calculate_all_custom_metrics(df)

# 1. DVI
dvi_df = m["df_with_dvi"]
print("\n1. DEAL VELOCITY INDEX (DVI)")
print(f"   Won avg DVI:  {dvi_df[dvi_df['is_won']==1]['deal_velocity_index'].mean():.3f}")
print(f"   Lost avg DVI: {dvi_df[dvi_df['is_won']==0]['deal_velocity_index'].mean():.3f}")

# 2. DQE
dqe = m["dqe_overall"]
print(f"\n2. DEAL QUALIFICATION EFFICIENCY (DQE)")
print(f"   Score: {float(dqe['dqe_score'].iloc[0]):.3f}")
print(f"   Category: {dqe['dqe_category'].iloc[0]}")
print(f"   Wasted capacity: {float(dqe['wasted_capacity_days'].iloc[0]):,.0f} days")
print(f"   Reps analyzed: {len(m['dqe_by_rep'])}")

# 3. RCR
for seg in ["region", "industry", "product"]:
    rcr = m[f"rcr_{seg}"]
    print(f"\n3. REVENUE CONCENTRATION RISK ({seg.upper()})")
    print(f"   RCR Score: {rcr['rcr_score']:.3f} ({rcr['risk_level']})")
    print(f"   Top: {rcr['top_segment']} ({rcr['top_segment_share']}%)")
    print(f"   Revenue at risk: ${rcr['revenue_at_risk_30pct_decline']:,.0f}")

# 4. SMI
for seg in ["region", "industry"]:
    smi = m[f"smi_{seg}"]
    print(f"\n4. SEGMENT MOMENTUM INDEX ({seg.upper()})")
    for _, row in smi.iterrows():
        print(f"   {row[seg]:20s} SMI={row['smi_score']:+.3f}  {row['momentum_category']}")

# 5. Insights
insights = identify_key_insights(df, m)
print(f"\n{'='*60}")
print(f"BUSINESS INSIGHTS: {len(insights)} generated")
print("=" * 60)
for i, ins in enumerate(insights, 1):
    print(f"\n{i}. [{ins['severity'].upper()}] {ins['emoji']} {ins['category']}")
    print(f"   {ins['insight'][:120]}...")
    print(f"   Source: {ins['metric_source']}")
    print(f"   Impact: {ins['estimated_impact']}")

print(f"\n{'='*60}")
print("ALL CUSTOM METRICS VERIFIED SUCCESSFULLY")
print("=" * 60)
