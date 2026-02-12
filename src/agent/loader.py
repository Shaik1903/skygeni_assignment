import pandas as pd
from typing import Optional
from src.data_loader import load_and_prepare_data

_global_df: Optional[pd.DataFrame] = None

def get_sales_data() -> pd.DataFrame:
    """
    Get the sales data, loading it if necessary.
    Returns a pandas DataFrame (cached in memory).
    """
    global _global_df
    if _global_df is None:
        _global_df, _ = load_and_prepare_data()
    return _global_df
