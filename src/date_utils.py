from __future__ import annotations

import pandas as pd


def parse_date_series(values: pd.Series, errors: str = "raise") -> pd.Series:
    """Parse Tushare YYYYMMDD strings or integers without treating integers as ns."""
    series = pd.Series(values, index=values.index, copy=False)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors=errors)
    text = series.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    result.loc[compact] = pd.to_datetime(
        text.loc[compact], format="%Y%m%d", errors=errors
    )
    result.loc[~compact] = pd.to_datetime(text.loc[~compact], errors=errors)
    return result
