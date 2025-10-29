# %% [markdown]
# # 02 - Optimization & Backtest (skeleton)
#
# Este notebook demonstra o pipeline usado no protótipo Robo-Advisor:
# - Fetch de preços (yfinance)
# - Limpeza e preparação
# - Cálculo de retornos e matrizes
# - Otimização (PyPortfolioOpt) com fallback
# - Backtesting calendarizado (monthly/quarterly)
# - Cálculo de métricas (CAGR, Sharpe, Sortino, MaxDD, Turnover)
# - Plots e exportação de resultados
#
# Observação: correr dentro do ambiente `.venv` com dependências já instaladas.

# %%
# Imports e configuração
import math
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Tentativa segura de importar PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models, EfficientFrontier
    _HAS_PFOPT = True
except Exception:
    expected_returns = None
    risk_models = None
    EfficientFrontier = None
    _HAS_PFOPT = False

# display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.6f}'.format)

# %%
# Fetch de preços
def fetch_prices(tickers, start):
    """Fetch adjusted close prices and return a clean DataFrame."""
    data = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(axis=1, how="all")
    return data

# Exemplo de uso (descomenta para correr)
# tickers = ["SPY", "AGG", "VT", "EEM", "BND"]
# start = "2015-01-01"
# prices = fetch_prices(tickers, start)
# prices.tail()

# %%
# Compute return matrices
def compute_return_matrices(prices):
    daily_returns = prices.pct_change().dropna()
    if _HAS_PFOPT:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
    else:
        mu = daily_returns.mean() * 252
        S = daily_returns.cov() * 252
    return daily_returns, mu, S

# %%
# Optimization helper
def compute_weights(prices, method="max_sharpe"):
    tickers = list(prices.columns)
    if method == "equal_weight":
        return {t: 1/len(tickers) for t in tickers}
    if not _HAS_PFOPT:
        return {t: 1/len(tickers) for t in tickers}
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    if method == "max_sharpe":
        ef.max_sharpe()
    elif method == "min_volatility":
        ef.min_volatility()
    try:
        w = ef.clean_weights()
    except Exception:
        try:
            raw = ef.weights
            w = {k: float(v) for k,v in raw.items()}
        except Exception:
            w = {t: 1/len(tickers) for t in tickers}
    return w

# %%
# Backtest helper (import from backend if exists)
try:
    from backend.app.services.backtest import backtest_weights
except Exception:
    def backtest_weights(prices, weights, capital=10000.0, rebalance="monthly", fee=0.0005):
        raise RuntimeError("backtest_weights não disponível — importa do backend/app/services/backtest.py")

# %%
# Executar exemplo: otimização + backtest
# Uncomment e ajuste os tickers/data para correr localmente.
# tickers = ["SPY", "AGG", "VT"]
# prices = fetch_prices(tickers, "2015-01-01")
# weights = compute_weights(prices, method="max_sharpe")
# pv_opt, metrics_opt = backtest_weights(prices, weights, capital=10000, rebalance="monthly")
# eq_weights = {t: 1/len(prices.columns) for t in prices.columns}
# pv_eq, metrics_eq = backtest_weights(prices, eq_weights, capital=10000, rebalance="monthly")
# print(metrics_opt)
# %%
# Visualizações exemplo (Plotly)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=pv_opt.index, y=pv_opt.values, name="Optimized"))
# fig.add_trace(go.Scatter(x=pv_eq.index, y=pv_eq.values, name="Equal-weight"))
# fig.update_layout(title="Optimized vs Equal-weight", yaxis_title="Portfolio value")
# fig.show()

