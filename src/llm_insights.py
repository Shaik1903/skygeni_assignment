"""
LLM-Powered Insights Module
============================
Uses Gemini 3 Flash (via LangChain) to generate richer, CRO-level insights
from our custom metrics and EDA results.

Falls back silently to rule-based insights if the LLM call fails for any reason
(no API key, network error, rate limit, etc.).
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# Load .env for GOOGLE_API_KEY
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def _get_llm():
    """Lazy-initialize the Gemini model. Returns None if unavailable."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[LLM] No GOOGLE_API_KEY found, falling back to rule-based insights.")
            return None

        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.01,
            max_retries=2,
        )
    except Exception as e:
        print(f"[LLM] Failed to initialize Gemini: {e}")
        return None


def _build_metrics_context(df: pd.DataFrame, metrics_dict: Dict) -> str:
    """Serialize key metrics data into a compact text block for the LLM prompt."""
    pqs = metrics_dict["pqs_overall"]
    wre = metrics_dict["wre"]
    
    # Summary stats
    total = len(df)
    won = int(df["is_won"].sum())
    win_rate = won / total * 100 if total > 0 else 0
    avg_deal = df[df["is_won"] == 1]["deal_amount"].mean()
    avg_cycle_won = df[df["is_won"] == 1]["sales_cycle_days"].mean()
    avg_cycle_lost = df[df["is_won"] == 0]["sales_cycle_days"].mean()

    # Region breakdown
    region_wr = df.groupby("region")["is_won"].mean().mul(100).round(1).to_dict()
    
    # Industry breakdown
    industry_wr = df.groupby("industry")["is_won"].mean().mul(100).round(1).to_dict()
    
    # Lead source breakdown
    source_wr = df.groupby("lead_source")["is_won"].mean().mul(100).round(1).to_dict()

    # SMI top movers
    smi = metrics_dict.get("smi_region")
    smi_summary = ""
    if smi is not None and len(smi) > 0:
        top_growing = smi.nlargest(3, "smi_score")[["region", "smi_score", "momentum_category"]].to_dict("records")
        top_declining = smi.nsmallest(3, "smi_score")[["region", "smi_score", "momentum_category"]].to_dict("records")
        smi_summary = f"""
Segment Momentum Index (SMI) — Region:
  Growing: {json.dumps(top_growing, default=str)}
  Declining: {json.dumps(top_declining, default=str)}"""

    # DVI
    dvi_df = metrics_dict.get("df_with_dvi")
    dvi_summary = ""
    if dvi_df is not None:
        won_dvi = dvi_df[dvi_df["is_won"] == 1]["deal_velocity_index"].mean()
        lost_dvi = dvi_df[dvi_df["is_won"] == 0]["deal_velocity_index"].mean()
        dvi_summary = f"""
Deal Velocity Index (DVI):
  Won deals avg DVI: {won_dvi:.2f} | Lost deals avg DVI: {lost_dvi:.2f}
  (DVI = deal_amount / cycle_days, normalized so 1.0 = median)"""

    # WRE bucket detail
    wre_buckets_str = ""
    if "bucket_details" in wre:
        buckets = wre["bucket_details"]
        wre_buckets_str = "\n  Buckets: " + json.dumps(
            [{k: v for k, v in b.items() if k in ("range", "win_rate", "count")} for b in buckets],
            default=str
        )

    return f"""=== SALES DATA OVERVIEW ===
Total deals: {total} | Won: {won} | Win Rate: {win_rate:.1f}%
Avg won deal size: ${avg_deal:,.0f}
Avg cycle: {avg_cycle_won:.0f} days (won) | {avg_cycle_lost:.0f} days (lost)

Win rate by region: {json.dumps(region_wr)}
Win rate by industry: {json.dumps(industry_wr)}
Win rate by lead source: {json.dumps(source_wr)}

=== PIPELINE QUALIFICATION SCORE (PQS): {pqs['pqs_score']:.0f}/100 ({pqs['pqs_category']}) ===
  DQE (qualification speed): {pqs['dqe_score']:.3f}
  Median cycle — Won: {pqs['median_cycle_won']:.0f} days | Lost: {pqs['median_cycle_lost']:.0f} days
  Winning window (P75): {pqs['winning_window_days']:.0f} days
  Stalling deals: {pqs['deals_stalling']} ({pqs['stall_rate_pct']:.0f}% of open pipeline)
  Likely dead deals: {pqs['deals_likely_dead']}
  On-pace win rate: {pqs['on_pace_win_rate']:.0f}% vs Stalling win rate: {pqs['stalling_win_rate']:.0f}%
  Wasted capacity: {pqs['wasted_capacity_days']:,.0f} rep-days

=== WIN RATE ELASTICITY (WRE): {wre['elasticity']:.3f} ({wre['severity']}) ===
  Smallest bucket WR: {wre['smallest_bucket_wr']:.1f}% | Largest bucket WR: {wre['largest_bucket_wr']:.1f}%
  Sweet spot: {wre['sweet_spot_range']} ({wre['sweet_spot_win_rate']:.1f}% WR)
  WR drop (small→large): {wre['wr_drop_small_to_large']:.1f}pp{wre_buckets_str}
{smi_summary}
{dvi_summary}"""


def _build_eda_context(eda_results: Dict) -> str:
    """Serialize EDA results into a compact text block."""
    parts = []

    # Win rate trends
    if "win_rate_trends" in eda_results:
        wrt = eda_results["win_rate_trends"]
        if "quarterly_data" in wrt:
            qd = wrt["quarterly_data"]
            if isinstance(qd, pd.DataFrame):
                last_4 = qd.tail(4)[["closed_year_quarter", "win_rate", "total_deals"]].to_dict("records")
            elif isinstance(qd, list):
                last_4 = qd[-4:]
            else:
                last_4 = []
            parts.append(f"Quarterly win rate trend (recent 4): {json.dumps(last_4, default=str)}")

    # Segment performance
    if "segment_analysis" in eda_results:
        sa = eda_results["segment_analysis"]
        for key in ["region_analysis", "industry_analysis", "product_analysis"]:
            if key in sa:
                data = sa[key]
                if isinstance(data, pd.DataFrame):
                    parts.append(f"{key}: {data.head(5).to_dict('records')}")
                elif isinstance(data, list):
                    parts.append(f"{key}: {data[:5]}")

    # Deal characteristics
    if "deal_characteristics" in eda_results:
        dc = eda_results["deal_characteristics"]
        if isinstance(dc, dict):
            for k, v in dc.items():
                if isinstance(v, (int, float, str)):
                    parts.append(f"  {k}: {v}")

    # Lead source
    if "lead_source_analysis" in eda_results:
        ls = eda_results["lead_source_analysis"]
        if isinstance(ls, pd.DataFrame):
            parts.append(f"Lead source analysis: {ls.to_dict('records')}")

    return "\n".join(parts) if parts else "No EDA data available."


INSIGHT_SYSTEM_PROMPT = """You are a senior sales analytics advisor generating insights for a CRO (Chief Revenue Officer).

RULES:
- Be specific with numbers. Never be vague.
- Every insight must have a concrete, actionable next step.
- Focus on what's SURPRISING or ACTIONABLE — skip obvious findings.
- Use the exact output format specified. No extra text outside the JSON.
"""


import re
import ast

def _extract_json_from_text(text: str) -> List[Dict]:
    """
    Robustly extract JSON array from text, handling markdown, 
    single quotes, and surrounding text.
    """
    if isinstance(text, list):
        text = "".join([str(item) for item in text])
    text = str(text)

    # 1. Try to find a JSON array block with regex
    # Looks for [ ... ] across multiple lines
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        json_candidate = match.group(0)
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            # If json.loads fails (e.g. single quotes), try ast.literal_eval
            try:
                return ast.literal_eval(json_candidate)
            except (ValueError, SyntaxError):
                pass
    
    # 2. Fallback: Try identifying markdown blocks
    # (Existing logic as backup, though regex usually catches this)
    clean_text = text.strip()
    if clean_text.startswith("```"):
        try:
            first_newline = clean_text.index("\n")
            clean_text = clean_text[first_newline+1:]
        except ValueError:
            pass
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # 3. Last resort: Try ast on the whole cleaned text
    try:
        return ast.literal_eval(clean_text)
    except (ValueError, SyntaxError):
        pass
        
    print(f"[LLM] Failed to parse JSON from response: {text[:100]}...")
    return []


def _parse_insights_json(llm_text: str) -> List[Dict]:
    """Parse LLM JSON output into insight dicts."""
    items = _extract_json_from_text(llm_text)

    if not isinstance(items, list):
        # Handle case where LLM returns a single object instead of list
        if isinstance(items, dict):
            items = [items]
        else:
            return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            result.append({
                "emoji": item.get("emoji", "💡"),
                "category": item.get("category", "Insight"),
                "insight": item.get("insight", ""),
                "severity": item.get("severity", "medium"),
                "business_action": item.get("business_action", item.get("action", "")),
                "estimated_impact": item.get("estimated_impact", ""),
            })
        except Exception:
            continue
            
    return result


def generate_llm_metric_insights(
    df: pd.DataFrame,
    metrics_dict: Dict,
    fallback_fn=None,
) -> List[Dict]:
    """
    Generate Custom Metrics insights using Gemini 3 Flash.
    Falls back to rule-based insights on any failure.
    """
    llm = _get_llm()
    if llm is None:
        if fallback_fn:
            return fallback_fn(df, metrics_dict)
        from metrics import identify_key_insights
        return identify_key_insights(df, metrics_dict)

    try:
        context = _build_metrics_context(df, metrics_dict)
        prompt = f"""{context}

---
Based on the metrics above, generate 5–7 HIGH-IMPACT insights for a CRO.

Return ONLY a JSON array where each item has:
- "emoji": one relevant emoji
- "category": short label (e.g. "Pipeline Health", "Deal Sizing", "Segment Shift")
- "insight": the finding in 1-2 sentences with specific numbers
- "severity": "high", "medium", or "low"
- "business_action": concrete next step the CRO should take
- "estimated_impact": quantified expected benefit (e.g. "Could save ~2,400 rep-days/quarter")

Focus on:
1. Pipeline qualification problems (PQS components: DQE + stall detection)
2. Deal sizing vs win rate trade-offs (WRE)
3. Segment momentum shifts (SMI)
4. Deal velocity patterns (DVI)
5. Cross-metric correlations (e.g. stalling deals in declining regions)
"""
        response = llm.invoke([
            ("system", INSIGHT_SYSTEM_PROMPT),
            ("human", prompt),
        ])
        insights = _parse_insights_json(response.content)
        if len(insights) >= 3:
            return insights
    except Exception as e:
        print(f"[LLM] Metric insights generation failed: {e}")
        # traceback.print_exc() # Optional: reduce log noise

    # Fallback
    if fallback_fn:
        return fallback_fn(df, metrics_dict)
    from metrics import identify_key_insights
    return identify_key_insights(df, metrics_dict)


def generate_llm_eda_insights(
    df: pd.DataFrame,
    eda_results: Dict,
    fallback_insights: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Generate EDA insights using Gemini 3 Flash.
    Falls back to rule-based EDA insights on any failure.
    """
    llm = _get_llm()
    if llm is None:
        return fallback_insights or eda_results.get("insights", [])

    try:
        context = _build_eda_context(eda_results)
        
        # Add basic stats for extra context
        total = len(df)
        win_rate = df["is_won"].mean() * 100
        stats_line = f"Overall: {total} deals, {win_rate:.1f}% win rate"

        prompt = f"""{stats_line}

{context}

---
Based on the exploratory data analysis above, generate 4–5 KEY FINDINGS for a CRO.

Return ONLY a JSON array where each item has:
- "emoji": one relevant emoji
- "category": short label (e.g. "Trend Alert", "Regional Gap", "Lead Quality")
- "insight": the finding in 1-2 sentences with specific numbers
- "severity": "high", "medium", or "low"
- "business_action": concrete next step
- "estimated_impact": quantified benefit if possible

Focus on PATTERNS, TRENDS, and ANOMALIES — things not obvious from the raw numbers.
"""
        response = llm.invoke([
            ("system", INSIGHT_SYSTEM_PROMPT),
            ("human", prompt),
        ])
        insights = _parse_insights_json(response.content)
        if len(insights) >= 2:
            return insights
    except Exception as e:
        print(f"[LLM] EDA insights generation failed: {e}")
        # traceback.print_exc()

    return fallback_insights or eda_results.get("insights", [])


def generate_llm_recommendations(
    df: pd.DataFrame,
    metrics_dict: Dict,
    eda_results: Dict,
    fallback_fn=None,
) -> List[Dict]:
    """
    Generate strategic recommendations using Gemini 3 Flash.
    Falls back to rule-based recommendations on any failure.
    """
    llm = _get_llm()
    if llm is None:
        if fallback_fn:
            return fallback_fn(df, [])
        return []

    try:
        metrics_context = _build_metrics_context(df, metrics_dict)
        eda_context = _build_eda_context(eda_results)

        prompt = f"""{metrics_context}

{eda_context}

---
You are a VP of Sales Strategy. Based on ALL the data above, generate 6-8 PRIORITIZED RECOMMENDATIONS.

Return ONLY a JSON array where each item has:
- "priority": "HIGH", "MEDIUM", or "LOW"
- "category": short label (e.g. "Pipeline Hygiene", "Territory Strategy")
- "action": the specific action to take (1 sentence)
- "rationale": why this matters, with supporting numbers (1-2 sentences)
- "expected_impact": quantified benefit
- "effort": "Low", "Medium", or "High"
- "timeline": realistic implementation time (e.g. "1-2 weeks")
- "icon": one relevant emoji

Order by priority (HIGH first). Mix quick wins (Low effort) with strategic moves (High effort).
Recommendations should cover: pipeline management, deal sizing, territory investment,
lead source optimization, rep coaching, and process improvements.
"""
        response = llm.invoke([
            ("system", "You are a VP of Sales Strategy advising a CRO. Be specific, data-driven, and action-oriented. Output valid JSON only."),
            ("human", prompt),
        ])
        recs = _parse_recommendations_json(response.content)
        if len(recs) >= 3:
            return recs
    except Exception as e:
        print(f"[LLM] Recommendations generation failed: {e}")
        # traceback.print_exc()

    if fallback_fn:
        return fallback_fn(df, [])
    return []


def _parse_recommendations_json(llm_text: str) -> List[Dict]:
    """Parse LLM JSON output into recommendation dicts."""
    items = _extract_json_from_text(llm_text)

    if not isinstance(items, list):
        if isinstance(items, dict):
            items = [items]
        else:
            return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            result.append({
                "priority": item.get("priority", "MEDIUM"),
                "category": item.get("category", "General"),
                "action": item.get("action", ""),
                "rationale": item.get("rationale", ""),
                "expected_impact": item.get("expected_impact", ""),
                "effort": item.get("effort", "Medium"),
                "timeline": item.get("timeline", "2-4 weeks"),
                "icon": item.get("icon", "💡"),
            })
        except Exception:
            continue
            
    return result
