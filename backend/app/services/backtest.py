"""
backtest.py
Utilities for portfolio backtesting with calendarized rebalancing.

Functions:
- rebal_dates_for(prices, rebalance)
- backtest_weights(prices, weights, capital, rebalance, fee)

Return:
- portfolio_values: pd.Series indexed by prices.index (float)
- metrics: dict with keys: start_value, end_value, cagr, ann_vol, sharpe, sortino, max_drawdown, total_traded_value, turnover, trading_days
"""
from __future__ import annotations
import math
from typing import Dict, Tuple

import pandas as pd
import numpy as np


def rebal_dates_for(prices: pd.DataFrame, rebalance: str) -> pd.DatetimeIndex:
    """
    Return dates (index) when rebalancing should occur, including the first available date.
    rebalance: "monthly" or "quarterly" (other -> monthly)
    """
    if rebalance == "monthly":
        dates = prices.resample("ME").last().index
    elif rebalance == "quarterly":
        dates = prices.resample("Q").last().index
    else:
        dates = prices.resample("ME").last().index

    first = prices.index[0]
    dates = pd.DatetimeIndex([d for d in dates if d >= first])
    if len(dates) == 0 or dates[0] != first:
        dates = pd.DatetimeIndex([first]).union(dates)
    return dates


def backtest_weights(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    capital: float = 10000.0,
    rebalance: str = "monthly",
    fee: float = 0.0005,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Simulate a portfolio with periodic rebalancing.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, index = dates (datetime), columns = tickers.
    weights : Dict[str, float]
        Mapping ticker -> target weight (doesn't need to sum to 1).
    capital : float
        Initial capital in currency units.
    rebalance : str
        'monthly' or 'quarterly' - determines rebalancing dates.
    fee : float
        Proportional fee per traded value, e.g. 0.0005 = 0.05%.

    Returns
    -------
    portfolio_values : pd.Series
        Time series of portfolio value (same index as prices).
    metrics : dict
        Dictionary with performance metrics.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pd.DataFrame")

    # Prepare prices and tickers
    prices = prices.sort_index().ffill().dropna(axis=1, how="all")
    tickers = [t for t in prices.columns if t in weights and float(weights.get(t, 0)) > 0]
    weights = {t: float(weights.get(t, 0.0)) for t in tickers}

    # normalize weights if needed
    s = sum(weights.values())
    if s == 0:
        raise ValueError("weights sum to zero or no matching tickers in prices")
    if abs(s - 1.0) > 1e-8:
        weights = {t: w / s for t, w in weights.items()}

    # Restrict prices to available tickers
    prices = prices.loc[:, tickers]

    dates = prices.index
    rebal_dates = rebal_dates_for(prices, rebalance)
    rebal_dates = pd.DatetimeIndex([d for d in rebal_dates if d >= dates[0] and d <= dates[-1]])

    # Holdings in shares (allow fractional shares)
    holdings = {t: 0.0 for t in tickers}
    portfolio_values = pd.Series(index=dates, dtype=float)
    total_traded_value = 0.0

    # initial allocation at first date
    d0 = dates[0]
    px0 = prices.loc[d0]
    for t in tickers:
        target_value = capital * weights[t]
        holdings[t] = target_value / px0[t]
    start_value = sum(holdings[t] * px0[t] for t in tickers)
    portfolio_values.iloc[0] = start_value

    # initial trade accounted (buys)
    initial_traded = sum(abs(capital * weights[t]) for t in tickers)
    total_traded_value += initial_traded * (1 + fee)

    # iterate over dates
    for i in range(1, len(dates)):
        d = dates[i]
        px = prices.loc[d]
        pv = sum(holdings[t] * px[t] for t in tickers)
        # rebalancing check
        if d in rebal_dates:
            target_values = {t: pv * weights[t] for t in tickers}
            current_values = {t: holdings[t] * px[t] for t in tickers}
            trades = {t: target_values[t] - current_values[t] for t in tickers}
            traded_value = sum(abs(v) for v in trades.values())
            fee_paid = traded_value * fee if traded_value > 0 else 0.0
            total_traded_value += traded_value + fee_paid
            # update holdings (allow fractional)
            for t in tickers:
                holdings[t] = target_values[t] / px[t]
            pv = sum(holdings[t] * px[t] for t in tickers)
        portfolio_values.iloc[i] = pv

    # compute metrics
    returns = portfolio_values.pct_change().dropna()
    trading_days = len(returns)
    years = trading_days / 252.0 if trading_days > 0 else 0.0
    end_value = portfolio_values.iloc[-1]
    if years > 0 and start_value > 0:
        cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    ann_vol = returns.std() * math.sqrt(252) if len(returns) > 1 else float("nan")
    sharpe = cagr / ann_vol if ann_vol and not math.isnan(ann_vol) and ann_vol != 0 else float("nan")

    negative_rets = returns[returns < 0]
    if len(negative_rets) > 0:
        downside_std = negative_rets.std() * math.sqrt(252)
        sortino = cagr / downside_std if downside_std != 0 else float("nan")
    else:
        downside_std = 0.0
        sortino = float("nan")

    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values / running_max) - 1.0
    max_drawdown = drawdown.min()

    avg_portfolio = portfolio_values.mean() if len(portfolio_values) > 0 else float("nan")
    turnover = total_traded_value / avg_portfolio if avg_portfolio and not math.isnan(avg_portfolio) else float("nan")

    metrics = {
        "start_value": float(start_value),
        "end_value": float(end_value),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "total_traded_value": float(total_traded_value),
        "turnover": float(turnover),
        "trading_days": int(trading_days),
    }

    return portfolio_values, metrics
