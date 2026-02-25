"""
Stock Market Predictions — daily update script
Fetches S&P 500 data via yfinance, runs linear regression for short-term
(7 trading days) and long-term (30 trading days) predictions, ranks all
companies and exports top 5 per category to data/predictions.json
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
SHORT_DAYS   = 7    # trading days ahead for short-term prediction
LONG_DAYS    = 30   # trading days ahead for long-term prediction
HISTORY_DAYS = 365  # calendar days of history to fetch
TOP_N        = 5    # top picks per category
BATCH_SIZE   = 50   # tickers per yfinance batch request
BATCH_SLEEP  = 2    # seconds between batches (rate-limit courtesy)
OUTPUT_FILE  = Path("data/predictions.json")

# ── Fetch S&P 500 tickers from Wikipedia ──────────────────────────────────
def get_sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        tickers = tables[0]["Symbol"].tolist()
        # yfinance uses "-" not "." for tickers like BRK.B → BRK-B
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"  ⚠ Could not fetch S&P 500 list from Wikipedia: {e}")
        # Fallback: top 50 well-known tickers
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V",
            "UNH","XOM","LLY","JNJ","PG","MA","HD","AVGO","MRK","CVX",
            "PEP","ABBV","KO","ADBE","WMT","MCD","BAC","ACN","CRM","NFLX",
            "TMO","CSCO","COST","ABT","PFE","LIN","NKE","DIS","TXN","NEE",
            "WFC","DHR","VZ","AMGN","RTX","PM","INTC","ORCL","HON","QCOM",
        ]

# ── Build features for a single ticker ────────────────────────────────────
def build_features(prices: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": prices})
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_5"]  = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_30"]
    df["volatility"] = df["ret_1"].rolling(20).std()
    df["momentum"]   = df["close"] / df["close"].shift(20) - 1
    df["t"] = np.arange(len(df))  # time index for trend
    df.dropna(inplace=True)
    return df

# ── Train a linear regression and predict N days ahead ─────────────────────
def predict(prices: pd.Series, horizon: int) -> dict | None:
    try:
        df = build_features(prices)
        if len(df) < 40:
            return None

        feature_cols = ["t","ret_1","ret_5","ret_20","sma_ratio","volatility","momentum"]
        X = df[feature_cols].values
        y = df["close"].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        # Predict future: extrapolate features linearly
        last_row = df.iloc[-1]
        future_rows = []
        last_t = last_row["t"]
        for i in range(1, horizon + 1):
            row = {
                "t":          last_t + i,
                "ret_1":      last_row["ret_1"],
                "ret_5":      last_row["ret_5"],
                "ret_20":     last_row["ret_20"],
                "sma_ratio":  last_row["sma_ratio"],
                "volatility": last_row["volatility"],
                "momentum":   last_row["momentum"],
            }
            future_rows.append(row)

        X_future = scaler.transform(pd.DataFrame(future_rows)[feature_cols].values)
        y_future = model.predict(X_future)

        current_price  = float(prices.iloc[-1])
        predicted_price = float(y_future[-1])
        pct_change     = (predicted_price - current_price) / current_price * 100

        # R² on training data as confidence proxy (clamped 0-100)
        r2 = max(0.0, min(1.0, model.score(X_scaled, y)))

        if pct_change > 2:
            signal = "Bullish"
        elif pct_change < -2:
            signal = "Bearish"
        else:
            signal = "Neutral"

        return {
            "current_price":   round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "pct_change":      round(pct_change, 2),
            "signal":          signal,
            "confidence":      round(r2 * 100, 1),
            "predicted_prices": [round(float(p), 2) for p in y_future],
        }
    except Exception:
        return None

# ── Process one ticker ──────────────────────────────────────────────────────
def process_ticker(ticker: str, hist: pd.DataFrame, company_name: str) -> dict | None:
    try:
        if hist.empty or len(hist) < 50:
            return None

        prices = hist["Close"].dropna()
        if len(prices) < 50:
            return None

        short = predict(prices, SHORT_DAYS)
        long  = predict(prices, LONG_DAYS)
        if not short or not long:
            return None

        # Last 90 calendar days of history for the chart (≈63 trading days)
        cutoff = prices.index[-1] - pd.Timedelta(days=90)
        recent = prices[prices.index >= cutoff]

        return {
            "ticker":       ticker,
            "name":         company_name,
            "short_term":   short,
            "long_term":    long,
            "history": {
                "dates":  [d.strftime("%Y-%m-%d") for d in recent.index],
                "prices": [round(float(p), 2) for p in recent.values],
            },
        }
    except Exception:
        return None

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("📈 Stock Market Predictions — starting update")
    print(f"   Short-term horizon : {SHORT_DAYS} trading days")
    print(f"   Long-term horizon  : {LONG_DAYS} trading days")

    tickers = get_sp500_tickers()
    print(f"   Tickers to process : {len(tickers)}")

    start_date = (datetime.today() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    results: list[dict] = []

    # Process in batches
    for batch_start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   Batch {batch_num}/{total_batches}: {batch[0]} … {batch[-1]}")

        try:
            raw = yf.download(
                batch,
                start=start_date,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
        except Exception as e:
            print(f"     ✗ Download failed: {e}")
            time.sleep(BATCH_SLEEP)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    hist = raw
                else:
                    hist = raw[ticker] if ticker in raw.columns.get_level_values(0) else pd.DataFrame()

                info = {}
                try:
                    info = yf.Ticker(ticker).fast_info
                    company_name = getattr(info, "display_name", None) or ticker
                except Exception:
                    company_name = ticker

                result = process_ticker(ticker, hist, company_name)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"     ✗ {ticker}: {e}")

        time.sleep(BATCH_SLEEP)

    print(f"\n   Processed {len(results)} tickers successfully")

    if not results:
        print("   ✗ No results — aborting")
        return

    # ── Rank and pick top 5 ──────────────────────────────────────────────
    # Short-term: highest predicted % gain with Bullish signal
    short_ranked = sorted(
        [r for r in results if r["short_term"]["signal"] == "Bullish"],
        key=lambda r: r["short_term"]["pct_change"],
        reverse=True,
    )[:TOP_N]

    # Long-term: highest predicted % gain with Bullish signal
    long_ranked = sorted(
        [r for r in results if r["long_term"]["signal"] == "Bullish"],
        key=lambda r: r["long_term"]["pct_change"],
        reverse=True,
    )[:TOP_N]

    # ── Build prediction date arrays ──────────────────────────────────────
    def future_dates(horizon: int) -> list[str]:
        dates, current = [], datetime.today()
        while len(dates) < horizon:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon–Fri
                dates.append(current.strftime("%Y-%m-%d"))
        return dates

    for stock in short_ranked + long_ranked:
        stock["short_term"]["future_dates"] = future_dates(SHORT_DAYS)
        stock["long_term"]["future_dates"]  = future_dates(LONG_DAYS)

    # ── Write JSON ────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "short_term":  short_ranked,
        "long_term":   long_ranked,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✅ Saved {OUTPUT_FILE}")
    print(f"   Top 5 short-term: {[s['ticker'] for s in short_ranked]}")
    print(f"   Top 5 long-term : {[s['ticker'] for s in long_ranked]}")


if __name__ == "__main__":
    main()
