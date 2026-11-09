from statistics import mean, pstdev

import yfinance as yf


IBM_FEATURE_NAMES = (
    "open", "high", "low", "close", "adjusted_close", "volume",
    "return_1d", "return_5d", "volatility_20d", "relative_volume",
)


def get_ibm_history():
    """Return daily (date, open, high, low, close, adjusted_close, volume) tuples."""
    history = yf.Ticker("IBM").history(
        period="max", interval="1d", auto_adjust=False, actions=False
    )
    if history.empty:
        raise RuntimeError("Yahoo Finance returned no IBM daily history.")

    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    rows = history.sort_index()[columns]
    return [
        (timestamp.date(), open_price, high, low, close, adjusted_close, volume)
        for timestamp, open_price, high, low, close, adjusted_close, volume in
        rows.itertuples(name=None)
    ]


def ibm_features(rows):
    """Build features in IBM_FEATURE_NAMES order from 21 or more oldest-first rows."""
    if len(rows) < 21:
        raise ValueError("At least 21 daily rows are required.")

    recent_rows = rows[-21:]
    closes = [row[4] for row in recent_rows]
    daily_returns = [today / yesterday - 1 for yesterday, today in zip(closes, closes[1:])]
    average_volume = mean(row[6] for row in recent_rows[:-1])
    if average_volume <= 0:
        raise ValueError("The prior 20 days must have positive average volume.")

    return (
        *recent_rows[-1][1:],
        daily_returns[-1],
        closes[-1] / closes[-6] - 1,
        pstdev(daily_returns),
        recent_rows[-1][6] / average_volume,
    )


if __name__ == "__main__":
    rows = get_ibm_history()
    print(f"{len(rows):,} daily rows from {rows[0][0]} through {rows[-1][0]}")
    print("First row:", rows[0])
    print("Last row:", rows[-1])
