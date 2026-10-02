"""90-trading-day evaluation against buy-and-hold and coin-flip baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_agent import SentimentModel, calculate_rsi, rule_signal


@dataclass(frozen=True)
class BacktestResult:
    observations: int
    strategy_return: float
    buy_hold_return: float
    coin_flip_mean_return: float
    strategy_max_drawdown: float
    buy_hold_max_drawdown: float
    coin_flip_mean_max_drawdown: float
    signal_counts: dict[str, int]
    coin_flip_runs: int


def fetch_aapl_prices(
    trading_days: int = 90,
    evaluation_end: str = "2026-09-30",
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with: pip install -r requirements.txt") from exc
    end = pd.Timestamp(evaluation_end)
    start = end - pd.Timedelta(days=220)
    raw = yf.download(
        "AAPL",
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("No AAPL data returned; check the internet connection")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.reset_index()[["Date", "Close"]].dropna()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    return frame.tail(trading_days + 20).reset_index(drop=True)


def load_dated_news(path: str | Path) -> pd.DataFrame:
    news = pd.read_csv(path)
    required = {"date", "headline", "source"}
    if not required.issubset(news.columns):
        raise ValueError(f"News CSV must contain {sorted(required)}")
    news = news.dropna(subset=["date", "headline"]).copy()
    news["date"] = pd.to_datetime(news["date"], errors="raise").dt.normalize()
    news["headline"] = news["headline"].astype(str).str.strip()
    return news[news["headline"].ne("")].sort_values("date")


def _max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def _equity_from_position(returns: pd.Series, position: pd.Series, cost: float) -> pd.Series:
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = position.shift(1).fillna(0) * returns - turnover * cost
    return (1 + strategy_returns).cumprod()


def run_backtest(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    sentiment_model: SentimentModel,
    trading_days: int = 90,
    transaction_cost: float = 0.001,
    coin_flip_runs: int = 500,
    seed: int = 6201,
) -> tuple[BacktestResult, pd.DataFrame]:
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    frame["RSI"] = calculate_rsi(frame["Close"])
    frame["market_return"] = frame["Close"].pct_change().fillna(0)

    analysed = news.copy()
    analysed["sentiment_score"] = [
        sentiment_model.analyse(text)["score"] for text in analysed["headline"]
    ]
    daily = analysed.groupby("date", as_index=False)["sentiment_score"].mean()
    # Weekend news becomes available on the next trading day, never earlier.
    trading_calendar = frame[["Date"]].drop_duplicates().sort_values("Date")
    daily = pd.merge_asof(
        daily.sort_values("date"),
        trading_calendar,
        left_on="date",
        right_on="Date",
        direction="forward",
    ).dropna(subset=["Date"])
    daily = daily.groupby("Date", as_index=False)["sentiment_score"].mean()
    frame = frame.merge(daily, on="Date", how="left")
    # No headline means neutral, not a guessed sentiment. No forward fill avoids look-ahead.
    frame["sentiment_score"] = frame["sentiment_score"].fillna(0.0)
    frame = frame.dropna(subset=["RSI"]).tail(trading_days).reset_index(drop=True)
    if len(frame) < trading_days:
        raise ValueError(f"Need {trading_days} valid RSI observations; found {len(frame)}")

    frame["signal"] = [
        rule_signal(float(score), float(rsi))
        for score, rsi in zip(frame["sentiment_score"], frame["RSI"])
    ]
    desired = frame["signal"].map({"BUY": 1.0, "SELL": 0.0, "HOLD": np.nan})
    frame["position"] = desired.ffill().fillna(0.0)
    frame["strategy_equity"] = _equity_from_position(
        frame["market_return"], frame["position"], transaction_cost
    )
    frame["buy_hold_equity"] = (1 + frame["market_return"]).cumprod()

    rng = np.random.default_rng(seed)
    coin_returns: list[float] = []
    coin_drawdowns: list[float] = []
    for _ in range(coin_flip_runs):
        # A literal fair coin: heads = invested in AAPL, tails = cash.
        positions = pd.Series(rng.integers(0, 2, size=len(frame)), dtype=float)
        equity = _equity_from_position(frame["market_return"], positions, transaction_cost)
        coin_returns.append(float(equity.iloc[-1] - 1))
        coin_drawdowns.append(_max_drawdown(equity))

    result = BacktestResult(
        observations=len(frame),
        strategy_return=float(frame["strategy_equity"].iloc[-1] - 1),
        buy_hold_return=float(frame["buy_hold_equity"].iloc[-1] - 1),
        coin_flip_mean_return=float(np.mean(coin_returns)),
        strategy_max_drawdown=_max_drawdown(frame["strategy_equity"]),
        buy_hold_max_drawdown=_max_drawdown(frame["buy_hold_equity"]),
        coin_flip_mean_max_drawdown=float(np.mean(coin_drawdowns)),
        signal_counts={key: int(value) for key, value in frame["signal"].value_counts().items()},
        coin_flip_runs=coin_flip_runs,
    )
    return result, frame
