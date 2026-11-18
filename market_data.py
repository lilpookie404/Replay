from datetime import date

import pandas as pd
import yfinance as yf


def download_daily_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Return adjusted daily Open/Close prices indexed by date.

    Dates must use YYYY-MM-DD; start is inclusive and end is exclusive.
    Invalid inputs raise ValueError; download or unusable-data failures raise
    RuntimeError. Invalid rows are rejected rather than silently removed.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a nonblank string.")
    ticker = ticker.strip().upper()
    if "," in ticker or any(character.isspace() for character in ticker):
        raise ValueError("Enter a single stock ticker.")
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise ValueError("Start and end dates are required as ISO date strings (YYYY-MM-DD).")
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        raise ValueError("Start and end dates must be valid ISO dates (YYYY-MM-DD).") from None
    if start.isoformat() != start_date or end.isoformat() != end_date:
        raise ValueError("Start and end dates must use YYYY-MM-DD.")
    if start >= end:
        raise ValueError("Start date must be before end date.")

    try:
        data = yf.download(
            ticker, start=start.isoformat(), end=end.isoformat(), interval="1d",
            auto_adjust=True, multi_level_index=False, ignore_tz=True,
            progress=False, threads=False, timeout=10,
        )
    except Exception as error:
        raise RuntimeError(f"Could not download daily prices for {ticker}.") from error

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise RuntimeError(f"No daily prices returned for {ticker} in the requested date range.")
    if (isinstance(data.columns, pd.MultiIndex) or not data.columns.is_unique
            or not {"Open", "Close"}.issubset(data.columns)):
        raise RuntimeError(f"Downloaded data for {ticker} must have unique flat Open and Close columns.")
    if (not isinstance(data.index, pd.DatetimeIndex)
            or data.index.hasnans or not data.index.is_unique):
        raise RuntimeError(f"Downloaded data for {ticker} must have a valid, unique date index.")
    if any(day < start or day >= end for day in data.index.date):
        raise RuntimeError(f"Downloaded data for {ticker} is outside the requested date range.")

    prices = data[["Open", "Close"]].copy()
    for column in prices:
        values = prices[column]
        if (not pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_complex_dtype(values)
                or values.isna().any()
                or not ((values > 0) & (values < float("inf"))).all()):
            raise RuntimeError(f"Downloaded data for {ticker} lacks usable numeric Open and Close prices.")
    return prices.sort_index()
