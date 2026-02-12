from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.loader import get_sales_data
import src.metrics as metrics
import src.forecasting as forecasting
import src.eda as eda
import json
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def get_key_metrics_summary() -> str:
    """
    Get a summary of key sales metrics including Win Rate, Pipeline Qualification Score (PQS),
    Win Rate Elasticity (WRE), and Segment Momentum.
    Useful for answering high-level questions about business health, such as "How is the business doing?" or "Give me a summary of our performance".
    """
    df = get_sales_data()
    all_metrics = metrics.calculate_all_custom_metrics(df)
    summary = metrics.get_metric_summary(all_metrics)
    
    # Add high-level win rate data directly
    summary["overall"] = {
        "total_deals": len(df),
        "won_deals": int(df["is_won"].sum()),
        "win_rate_pct": round(df["is_won"].mean() * 100, 2),
        "total_revenue": float(df["deal_amount"].sum()),
        "avg_deal_size": float(df["deal_amount"].mean())
    }
    
    # Helper to convert numpy/pandas types to native python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        if hasattr(obj, 'item'): 
            return obj.item()
        raise TypeError
        
    return json.dumps(summary, indent=2, default=convert_types)

@tool
def get_revenue_forecast(weeks: int = 12) -> str:
    """
    Generate a revenue forecast for the next N weeks.
    Returns predicted revenue and upper/lower bounds.
    Useful for questions like "What is our projected revenue?", "Forecast for next quarter", or "How much will we close next month?".
    """
    df = get_sales_data()
    results = forecasting.run_forecast_pipeline(df, forecast_weeks=weeks)
    forecast_df = results["forecast"]["forecast_df"]
    
    # Format for LLM - limit to key columns and convert timestamp
    simple_forecast = forecast_df[["week_start", "ensemble_forecast", "ensemble_lower", "ensemble_upper"]].copy()
    simple_forecast["week_start"] = simple_forecast["week_start"].dt.strftime("%Y-%m-%d")
    
    return json.dumps(simple_forecast.to_dict(orient="records"), indent=2)

@tool
def explain_win_rate_trends() -> str:
    """
    Analyze why the win rate is changing (e.g. dropping or rising). 
    Identifies trends, specific underperforming segments (region, industry), and key deal characteristics driving the change.
    Useful for "Why are we losing deals?", "Explain the drop in win rate", or "What's affecting our win rate?".
    """
    df = get_sales_data()
    eda_results = eda.run_full_eda(df)
    insights = eda_results["insights"]
    return json.dumps(insights, indent=2)


@tool
def analyze_sales_data(query: str) -> str:
    """
    Analyze the sales data using Python logic to answer specific, ad-hoc questions.
    Use this for aggregations, filtering, or specific counts that are not covered by other tools.
    Examples: "How many deals created in Q3?", "Average deal size for Tech industry", "List top 5 reps by revenue".
    The tool has access to the full sales dataframe.
    """
    df = get_sales_data()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True,
        # Reduce max iterations to avoid long loops
        max_iterations=5
    )
    return agent_executor.invoke({"input": query})["output"]
