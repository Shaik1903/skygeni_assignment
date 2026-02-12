"""
Forecasting Module for SkyGeni Sales Intelligence
===================================================

Two-model hybrid system:
  Model 1 — Weekly revenue forecast (Prophet + XGBoost Regressor ensemble)
  Model 2 — Deal-level win/loss classifier (XGBoost + SHAP) for feature importance

Model 2 is shared across the Revenue Forecast page AND the Win Rate Drivers page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
)
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate deal-level data into weekly revenue time series.
    
    Returns DataFrame with columns:
        week_start, revenue_won, deals_won, deals_lost, deals_total,
        win_rate, avg_deal_size, avg_cycle_won, avg_cycle_lost,
        deals_created, pipeline_value_created
    """
    df = df.copy()
    df["closed_week"] = df["closed_date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["created_week"] = df["created_date"].dt.to_period("W").apply(lambda r: r.start_time)

    # Won revenue per week
    won = df[df["is_won"] == 1].groupby("closed_week").agg(
        revenue_won=("deal_amount", "sum"),
        deals_won=("deal_id", "count"),
        avg_deal_size=("deal_amount", "mean"),
        avg_cycle_won=("sales_cycle_days", "mean"),
    ).reset_index()

    lost = df[df["is_won"] == 0].groupby("closed_week").agg(
        deals_lost=("deal_id", "count"),
        avg_cycle_lost=("sales_cycle_days", "mean"),
    ).reset_index()

    total = df.groupby("closed_week").agg(
        deals_total=("deal_id", "count"),
    ).reset_index()

    # Pipeline created per week (leading indicator)
    created = df.groupby("created_week").agg(
        deals_created=("deal_id", "count"),
        pipeline_value_created=("deal_amount", "sum"),
    ).reset_index().rename(columns={"created_week": "closed_week"})

    # Merge
    weekly = won.merge(lost, on="closed_week", how="outer") \
               .merge(total, on="closed_week", how="outer") \
               .merge(created, on="closed_week", how="outer") \
               .sort_values("closed_week") \
               .reset_index(drop=True)

    weekly = weekly.fillna(0)
    weekly["win_rate"] = np.where(
        weekly["deals_total"] > 0,
        weekly["deals_won"] / weekly["deals_total"],
        0
    )

    # Drop first 2 and last 2 weeks (partial)
    if len(weekly) > 4:
        weekly = weekly.iloc[2:-2].reset_index(drop=True)

    weekly.rename(columns={"closed_week": "week_start"}, inplace=True)
    return weekly


def prepare_deal_level(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], LabelEncoder]:
    """
    Prepare deal-level features for XGBoost classifier.
    
    Returns:
        features_df: DataFrame with encoded features
        feature_names: list of feature column names
        label_encoders: dict of LabelEncoders for categorical columns
    """
    df = df.copy()

    # Temporal features
    df["created_month"] = df["created_date"].dt.month
    df["created_quarter"] = df["created_date"].dt.quarter
    df["created_month_sin"] = np.sin(2 * np.pi * df["created_month"] / 12)
    df["created_month_cos"] = np.cos(2 * np.pi * df["created_month"] / 12)
    df["is_quarter_end"] = df["created_month"].isin([3, 6, 9, 12]).astype(int)

    # Rep trailing win rate (up to that point in time — no leakage)
    df = df.sort_values("created_date").reset_index(drop=True)
    df["rep_win_rate_trailing"] = (
        df.groupby("sales_rep_id")["is_won"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["rep_win_rate_trailing"] = df["rep_win_rate_trailing"].fillna(0.5)  # prior

    # Label encode categoricals
    cat_cols = ["region", "industry", "product_type", "lead_source"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Feature columns
    feature_cols = [
        "deal_amount", "sales_cycle_days",
        "region_enc", "industry_enc", "product_type_enc", "lead_source_enc",
        "created_month_sin", "created_month_cos", "created_quarter",
        "is_quarter_end", "rep_win_rate_trailing",
    ]

    # Human-readable names for SHAP
    feature_display_names = [
        "Deal Amount ($)", "Sales Cycle (days)",
        "Region", "Industry", "Product Type", "Lead Source",
        "Month (sin)", "Month (cos)", "Quarter",
        "Quarter-End Deal", "Rep Historical Win Rate",
    ]

    return df, feature_cols, feature_display_names, label_encoders


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (Weekly)
# ═══════════════════════════════════════════════════════════════

def engineer_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, lag, rolling, and custom metric features to weekly series."""
    df = weekly.copy()

    # Temporal
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_quarter_end"] = df["month"].isin([3, 6, 9, 12]).astype(int)
    df["time_index"] = np.arange(len(df))

    # Lag features
    for lag in [1, 2, 4]:
        df[f"revenue_lag_{lag}"] = df["revenue_won"].shift(lag)
        df[f"win_rate_lag_{lag}"] = df["win_rate"].shift(lag)

    # Rolling features
    df["revenue_ma_4"] = df["revenue_won"].rolling(4, min_periods=2).mean()
    df["revenue_ma_8"] = df["revenue_won"].rolling(8, min_periods=4).mean()
    df["revenue_std_4"] = df["revenue_won"].rolling(4, min_periods=2).std()

    # Leading indicator lags
    df["pipeline_created_lag1"] = df["deals_created"].shift(1)
    df["pipeline_value_lag1"] = df["pipeline_value_created"].shift(1)

    # Custom Metric 1: Sales Velocity Coefficient
    df["svc"] = np.where(
        df["avg_cycle_won"] > 0,
        (df["deals_won"] * df["avg_deal_size"]) / df["avg_cycle_won"],
        0
    )

    # Custom Metric 2: Pipeline Conversion Momentum
    df["pcm"] = (df["win_rate"] - df["win_rate"].shift(4)) * df["deals_created"]

    # Custom Metric 3: Deal Quality Drift
    overall_avg = df["avg_deal_size"].mean()
    df["dqd"] = np.where(
        overall_avg > 0,
        df["avg_deal_size"] / overall_avg - 1,
        0
    )

    return df


# ═══════════════════════════════════════════════════════════════
# 3. MODEL 2: DEAL-LEVEL WIN/LOSS CLASSIFIER (shared)
# ═══════════════════════════════════════════════════════════════

def train_win_loss_model(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost classifier for win/loss prediction.
    Uses 5-fold stratified CV for validation.
    
    Returns:
        model: trained XGBClassifier
        results: dict with metrics, feature importance, CV scores
    """
    X = df[feature_cols].values
    y = df["is_won"].values

    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {"auc": [], "f1": []}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        cv_scores["auc"].append(roc_auc_score(y_val, y_pred_proba))
        cv_scores["f1"].append(f1_score(y_val, y_pred))

    # Train final model on all data
    final_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importance (gain-based)
    importance = final_model.feature_importances_

    results = {
        "cv_auc_mean": np.mean(cv_scores["auc"]),
        "cv_auc_std": np.std(cv_scores["auc"]),
        "cv_f1_mean": np.mean(cv_scores["f1"]),
        "cv_f1_std": np.std(cv_scores["f1"]),
        "feature_importance": importance,
        "cv_scores": cv_scores,
    }

    return final_model, results


def compute_shap_values(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Compute SHAP values for the trained model.
    
    Returns dict with:
        shap_values: array of SHAP values
        mean_abs_shap: mean |SHAP| per feature (global importance)
        feature_names: list of feature names
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        return {
            "shap_values": shap_values,
            "expected_value": explainer.expected_value,
            "mean_abs_shap": mean_abs_shap,
            "sorted_idx": sorted_idx,
            "feature_names": feature_names,
            "X": X,
        }
    except ImportError:
        print("[Forecast] shap not installed, using feature_importances_ instead")
        return {
            "shap_values": None,
            "mean_abs_shap": model.feature_importances_,
            "sorted_idx": np.argsort(model.feature_importances_)[::-1],
            "feature_names": feature_names,
            "X": X,
        }


# ═══════════════════════════════════════════════════════════════
# 4. MODEL 1: WEEKLY REVENUE FORECAST
# ═══════════════════════════════════════════════════════════════

WEEKLY_FEATURE_COLS = [
    "time_index", "month_sin", "month_cos", "is_quarter_end",
    "revenue_lag_1", "revenue_lag_2", "revenue_lag_4",
    "revenue_ma_4", "revenue_ma_8", "revenue_std_4",
    "win_rate", "avg_deal_size", "deals_won", "deals_created",
    "pipeline_created_lag1", "pipeline_value_lag1",
    "svc", "pcm", "dqd",
]


def _train_xgb_regressor(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost regressor for revenue forecasting."""
    eval_set = [(X_train, y_train)]
    if X_val is not None:
        eval_set.append((X_val, y_val))

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def _train_prophet(weekly_df: pd.DataFrame, periods: int = 12):
    """Train Prophet model on weekly revenue data."""
    try:
        from prophet import Prophet

        # Prophet needs 'ds' and 'y' columns
        prophet_df = weekly_df[["week_start", "revenue_won"]].copy()
        prophet_df.columns = ["ds", "y"]

        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=1.0,
            seasonality_mode="multiplicative",
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
        )
        model.add_seasonality(name="quarterly", period=13, fourier_order=3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)

        # Forecast
        future = model.make_future_dataframe(periods=periods, freq="W-MON")
        forecast = model.predict(future)

        return model, forecast
    except ImportError:
        print("[Forecast] Prophet not installed, skipping Prophet model.")
        return None, None


def walk_forward_cv(weekly_df: pd.DataFrame, feature_cols: List[str],
                    min_train: int = 30, test_size: int = 4) -> Dict:
    """
    Walk-forward cross-validation for the XGBoost regressor.
    
    Args:
        weekly_df: weekly DataFrame with features engineered
        feature_cols: list of feature column names
        min_train: minimum training window size
        test_size: number of weeks per test fold
    
    Returns:
        dict with fold-level and aggregate metrics
    """
    df = weekly_df.dropna(subset=feature_cols + ["revenue_won"]).reset_index(drop=True)
    n = len(df)

    folds = []
    start = min_train

    while start + test_size <= n:
        train_idx = list(range(0, start))
        test_idx = list(range(start, min(start + test_size, n)))
        folds.append((train_idx, test_idx))
        start += test_size

    if not folds:
        return {"error": "Not enough data for walk-forward CV"}

    metrics = {"mae": [], "mape": [], "rmse": []}

    for train_idx, test_idx in folds:
        X_train = df.loc[train_idx, feature_cols].values
        y_train = df.loc[train_idx, "revenue_won"].values
        X_test = df.loc[test_idx, feature_cols].values
        y_test = df.loc[test_idx, "revenue_won"].values

        model = _train_xgb_regressor(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics["mae"].append(mean_absolute_error(y_test, y_pred))
        metrics["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))

        # MAPE (avoid division by zero)
        nonzero = y_test > 0
        if nonzero.any():
            metrics["mape"].append(
                mean_absolute_percentage_error(y_test[nonzero], y_pred[nonzero])
            )

    return {
        "n_folds": len(folds),
        "mae_mean": np.mean(metrics["mae"]),
        "rmse_mean": np.mean(metrics["rmse"]),
        "mape_mean": np.mean(metrics["mape"]) if metrics["mape"] else None,
        "mae_per_fold": metrics["mae"],
        "rmse_per_fold": metrics["rmse"],
        "mape_per_fold": metrics["mape"],
    }


def generate_forecast(weekly_df: pd.DataFrame, feature_cols: List[str],
                      forecast_weeks: int = 12) -> Dict:
    """
    Generate revenue forecast using Prophet + XGBoost ensemble.
    
    Returns dict with:
        forecast_df: DataFrame with week, prophet_pred, xgb_pred, ensemble
        prophet_model, prophet_forecast: Prophet objects (or None)
        xgb_model: trained XGBoost regressor
        cv_results: walk-forward CV metrics
        feature_importance: XGB feature importance for forecast model
    """
    df = weekly_df.dropna(subset=feature_cols + ["revenue_won"]).reset_index(drop=True)

    # --- XGBoost Regressor ---
    X = df[feature_cols].values
    y = df["revenue_won"].values
    xgb_model = _train_xgb_regressor(X, y)

    # XGB forecast: project features forward
    # Use last known values + trend for lag features
    last_row = df[feature_cols].iloc[-1:].copy()
    xgb_forecasts = []

    for i in range(forecast_weeks):
        pred = xgb_model.predict(last_row.values)[0]
        xgb_forecasts.append(max(0, pred))

        # Shift lags forward (approximate)
        new_row = last_row.copy()
        if "revenue_lag_1" in feature_cols:
            col_idx = feature_cols.index("revenue_lag_1")
            new_row.iloc[0, col_idx] = pred
        if "time_index" in feature_cols:
            col_idx = feature_cols.index("time_index")
            new_row.iloc[0, col_idx] += 1
        last_row = new_row

    # --- Prophet ---
    prophet_model, prophet_forecast = _train_prophet(weekly_df, periods=forecast_weeks)

    # Build forecast DataFrame
    last_date = df["week_start"].iloc[-1] if "week_start" in df.columns else weekly_df["week_start"].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(forecast_weeks)]

    forecast_df = pd.DataFrame({
        "week_start": forecast_dates,
        "xgb_forecast": xgb_forecasts,
    })

    # Add Prophet forecasts if available
    if prophet_forecast is not None:
        prophet_future = prophet_forecast.tail(forecast_weeks)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
        forecast_df["prophet_forecast"] = prophet_future["yhat"].values
        forecast_df["prophet_lower"] = prophet_future["yhat_lower"].values
        forecast_df["prophet_upper"] = prophet_future["yhat_upper"].values

        # Ensemble: Prophet 0.4 + XGB 0.6
        forecast_df["ensemble_forecast"] = (
            0.4 * forecast_df["prophet_forecast"] + 0.6 * forecast_df["xgb_forecast"]
        )
        # Confidence bands from Prophet scaled to ensemble
        scale = forecast_df["ensemble_forecast"] / forecast_df["prophet_forecast"].clip(lower=1)
        forecast_df["ensemble_lower"] = forecast_df["prophet_lower"] * scale
        forecast_df["ensemble_upper"] = forecast_df["prophet_upper"] * scale
    else:
        # XGB only — use std-based confidence bands
        residual_std = df["revenue_won"].std() * 0.5
        forecast_df["ensemble_forecast"] = forecast_df["xgb_forecast"]
        forecast_df["ensemble_lower"] = forecast_df["xgb_forecast"] - 1.96 * residual_std
        forecast_df["ensemble_upper"] = forecast_df["xgb_forecast"] + 1.96 * residual_std

    # --- Walk-Forward CV ---
    cv_results = walk_forward_cv(weekly_df, feature_cols)

    return {
        "forecast_df": forecast_df,
        "historical_df": weekly_df,
        "prophet_model": prophet_model,
        "prophet_forecast": prophet_forecast,
        "xgb_model": xgb_model,
        "cv_results": cv_results,
        "xgb_feature_importance": dict(zip(feature_cols, xgb_model.feature_importances_)),
    }


# ═══════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE (PUBLIC API)
# ═══════════════════════════════════════════════════════════════

def run_forecast_pipeline(df: pd.DataFrame, forecast_weeks: int = 12) -> Dict:
    """
    Run the full forecast pipeline.
    
    Returns a dict with all results:
        - weekly_series: aggregated weekly data
        - forecast: revenue forecast results
        - win_loss_model: trained XGBoost classifier
        - win_loss_results: CV metrics and feature importance
        - shap_analysis: SHAP values and feature ranking
        - deal_features: (df, feature_cols, display_names)
    """
    print("[Forecast] Step 1/5: Preparing weekly time series...")
    weekly = prepare_weekly_series(df)
    weekly = engineer_weekly_features(weekly)

    print("[Forecast] Step 2/5: Preparing deal-level features...")
    deal_df, feat_cols, feat_display, label_encs = prepare_deal_level(df)

    print("[Forecast] Step 3/5: Training win/loss classifier (Model 2)...")
    win_loss_model, win_loss_results = train_win_loss_model(deal_df, feat_cols)

    print("[Forecast] Step 4/5: Computing SHAP values...")
    X_deals = deal_df[feat_cols].values
    shap_analysis = compute_shap_values(win_loss_model, X_deals, feat_display)

    print("[Forecast] Step 5/5: Generating revenue forecast (Model 1)...")
    # Use only available feature columns
    available_cols = [c for c in WEEKLY_FEATURE_COLS if c in weekly.columns]
    forecast = generate_forecast(weekly, available_cols, forecast_weeks)

    print("[Forecast] Pipeline complete!")

    return {
        "weekly_series": weekly,
        "forecast": forecast,
        "win_loss_model": win_loss_model,
        "win_loss_results": win_loss_results,
        "shap_analysis": shap_analysis,
        "deal_features": {
            "df": deal_df,
            "feature_cols": feat_cols,
            "display_names": feat_display,
            "label_encoders": label_encs,
        },
    }


@st.cache_resource(ttl=3600, show_spinner=False)
def get_forecast_pipeline_cached(df: pd.DataFrame, forecast_weeks: int = 12) -> Dict:
    """Cached wrapper for full forecast pipeline."""
    return run_forecast_pipeline(df, forecast_weeks)


# ═══════════════════════════════════════════════════════════════
# DIRECT TESTING
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_loader import load_and_prepare_data

    df, _ = load_and_prepare_data()

    results = run_forecast_pipeline(df, forecast_weeks=12)

    print("\n" + "=" * 60)
    print("FORECAST PIPELINE RESULTS")
    print("=" * 60)

    # Model 2 results
    wr = results["win_loss_results"]
    print(f"\n📊 Win/Loss Classifier (Model 2):")
    print(f"   AUC-ROC: {wr['cv_auc_mean']:.3f} ± {wr['cv_auc_std']:.3f}")
    print(f"   F1 Score: {wr['cv_f1_mean']:.3f} ± {wr['cv_f1_std']:.3f}")

    # SHAP
    shap = results["shap_analysis"]
    print(f"\n🔍 Top SHAP Features:")
    for i in shap["sorted_idx"][:5]:
        print(f"   {shap['feature_names'][i]}: {shap['mean_abs_shap'][i]:.4f}")

    # Forecast
    fc = results["forecast"]
    print(f"\n📈 Revenue Forecast (next 12 weeks):")
    fdf = fc["forecast_df"]
    for _, row in fdf.head(6).iterrows():
        print(f"   {row['week_start'].strftime('%Y-%m-%d')}: "
              f"${row['ensemble_forecast']:,.0f}")

    # CV results
    cv = fc["cv_results"]
    print(f"\n✅ Walk-Forward CV:")
    print(f"   MAE: ${cv['mae_mean']:,.0f}")
    if cv.get("mape_mean"):
        print(f"   MAPE: {cv['mape_mean']:.1%}")
