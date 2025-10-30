import math
from typing import Dict, Tuple

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# tentativa segura de importar PyPortfolioOpt; se não estiver disponível, usa fallback
try:
    from pypfopt import expected_returns, risk_models, EfficientFrontier
    _HAS_PFOPT = True
except Exception:
    expected_returns = None
    risk_models = None
    EfficientFrontier = None
    _HAS_PFOPT = False

import plotly.graph_objs as go

from frontend.shap_explain import render_shap
from backend.app.services.backtest import backtest_weights, rebal_dates_for


st.set_page_config(page_title="Robo-Advisor MVP", layout="wide")
st.title("Robo-Advisor MVP — Demo rápida")

# ---------------- BACKTESTER HELPERS ----------------
def rebal_dates_for(prices: pd.DataFrame, rebalance: str):
    if rebalance == "monthly":
        dates = prices.resample("M").last().index
    elif rebalance == "quarterly":
        dates = prices.resample("Q").last().index
    else:
        dates = prices.resample("M").last().index
    first = prices.index[0]
    dates = [d for d in dates if d >= first]
    if len(dates) == 0 or dates[0] != first:
        dates = pd.DatetimeIndex([first]).union(dates)
    return dates

def backtest_weights(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    capital: float = 10000.0,
    rebalance: str = "monthly",
    fee: float = 0.0005,
) -> Tuple[pd.Series, Dict]:
    prices = prices.sort_index().ffill().dropna(axis=1, how="all")
    tickers = [t for t in prices.columns if t in weights and weights[t] > 0]
    weights = {t: float(weights.get(t, 0.0)) for t in tickers}
    s = sum(weights.values())
    if s == 0:
        raise ValueError("Weights sum to zero.")
    if abs(s - 1.0) > 1e-8:
        weights = {t: w / s for t, w in weights.items()}

    start = prices.index[0]
    end = prices.index[-1]
    prices = prices.loc[start:end, tickers]

    dates = prices.index
    rebal_dates = rebal_dates_for(prices, rebalance)
    rebal_dates = [d for d in rebal_dates if d >= dates[0] and d <= dates[-1]]

    holdings = {t: 0.0 for t in tickers}
    portfolio_values = pd.Series(index=dates, dtype=float)
    total_traded_value = 0.0

    d0 = dates[0]
    px0 = prices.loc[d0]
    for t in tickers:
        target_value = capital * weights[t]
        holdings[t] = target_value / px0[t]
    current_value = sum(holdings[t] * px0[t] for t in tickers)
    portfolio_values.iloc[0] = current_value

    initial_traded = sum(abs(capital * weights[t]) for t in tickers)
    total_traded_value += initial_traded * (1 + fee)

    for i in range(1, len(dates)):
        d = dates[i]
        px = prices.loc[d]
        pv = sum(holdings[t] * px[t] for t in tickers)
        if d in rebal_dates:
            target_values = {t: pv * weights[t] for t in tickers}
            current_values = {t: holdings[t] * px[t] for t in tickers}
            trades = {t: target_values[t] - current_values[t] for t in tickers}
            traded_value = sum(abs(v) for v in trades.values())
            if traded_value > 0:
                fee_paid = traded_value * fee
            else:
                fee_paid = 0.0
            total_traded_value += traded_value + fee_paid
            for t in tickers:
                holdings[t] = target_values[t] / px[t]
            pv = sum(holdings[t] * px[t] for t in tickers)
        portfolio_values.iloc[i] = pv

    returns = portfolio_values.pct_change().dropna()
    trading_days = len(returns)
    years = trading_days / 252.0 if trading_days > 0 else 0.0
    start_val = portfolio_values.iloc[0]
    end_val = portfolio_values.iloc[-1]
    if years > 0:
        cagr = (end_val / start_val) ** (1.0 / years) - 1.0
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
        "start_value": float(start_val),
        "end_value": float(end_val),
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
# ---------------- END HELPERS ----------------

with st.sidebar:
    
    run_shap = st.sidebar.button('Run SHAP explainability')
st.header("Inputs")
    tickers_input = st.text_input("Tickers (comma separated)", value="SPY, AGG, VT, EEM, BND")
    start_date = st.date_input("Start date", value=pd.to_datetime("2015-01-01"))
    capital = st.number_input("Capital (EUR)", min_value=100.0, value=10000.0, step=100.0)
    rebalance = st.selectbox("Rebalance", options=["monthly", "quarterly"], index=0)
    method = st.selectbox("Optimization method", options=["max_sharpe", "min_volatility", "equal_weight"], index=0)
    btn_run = st.button("Gerar alocação e backtest")

def fetch_prices(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(axis=1, how="all")
    return data

if btn_run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.write("Tickers:", tickers)
    with st.spinner("A descarregar preços e a calcular..."):
        prices = fetch_prices(tickers, start_date)
        if prices.shape[1] == 0:
            st.error("Nenhum dado disponível para os tickers indicados.")
        else:
            st.subheader("Dados carregados")
            st.dataframe(prices.tail())

            # ---- compute weights (with fallback) ----
            if method == "equal_weight":
                weights = {t: 1/len(tickers) for t in tickers}
            else:
                if not _HAS_PFOPT:
                    st.warning("PyPortfolioOpt não disponível — a alocação será equal-weight.")
                    weights = {t: 1/len(tickers) for t in tickers}
                else:
                    mu = expected_returns.mean_historical_return(prices)
                    S = risk_models.sample_cov(prices)
                    ef = EfficientFrontier(mu, S)
                    try:
                        if method == "max_sharpe":
                            ef.max_sharpe()
                        elif method == "min_volatility":
                            ef.min_volatility()
                    except Exception as e:
                        st.warning(f"Erro na optimização ({e}) — usando equal-weight.")
                        weights = {t: 1/len(tickers) for t in tickers}
                    try:
                        weights = ef.clean_weights()
                    except Exception:
                        # if clean_weights not available or failed, calculate from raw weights
                        try:
                            raw_w = ef.weights
                            weights = {k: float(v) for k, v in raw_w.items()}
                        except Exception:
                            weights = {t: 1/len(tickers) for t in tickers}

            # Mostrar alocação
            df_weights = pd.DataFrame.from_dict(weights, orient="index", columns=["weight"])
            df_weights = df_weights[df_weights.weight > 0]
            df_weights["allocation_EUR"] = (df_weights.weight * capital).round(2)
            st.subheader("Alocação")
            st.table(df_weights.sort_values("weight", ascending=False))

            # ---------------- Backtest ----------------
            try:
                pv_series, metrics_bt = backtest_weights(prices, weights, capital=capital, rebalance=rebalance, fee=0.0005)
            except Exception as e:
                st.error(f"Erro no backtest: {e}")
                pv_series = None
                metrics_bt = {}

            if pv_series is not None:
                st.subheader("Resultados do Backtest")
                metric_display = {
                    "Start value": metrics_bt.get("start_value"),
                    "End value": metrics_bt.get("end_value"),
                    "CAGR": metrics_bt.get("cagr"),
                    "Annualized Volatility": metrics_bt.get("ann_vol"),
                    "Sharpe": metrics_bt.get("sharpe"),
                    "Sortino": metrics_bt.get("sortino"),
                    "Max Drawdown": metrics_bt.get("max_drawdown"),
                    "Turnover": metrics_bt.get("turnover"),
                }
                metrics_df = pd.DataFrame.from_dict(metric_display, orient="index", columns=["value"])

                col1, col2 = st.columns([2,1])
                with col1:
                    st.write(metrics_df)
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pv_series.index, y=pv_series.values, name="Portfolio Value"))
                    st.plotly_chart(fig, use_container_width=True)

                # baseline equal-weight for comparison
                eq_weights = {t: 1/len(prices.columns) for t in prices.columns}
                pv_eq, metrics_eq = backtest_weights(prices, eq_weights, capital=capital, rebalance=rebalance, fee=0.0005)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=pv_series.index, y=pv_series.values, name="Optimized"))
                fig2.add_trace(go.Scatter(x=pv_eq.index, y=pv_eq.values, name="Equal-weight"))
                fig2.update_layout(title="Comparação: Optimized vs Equal-weight", yaxis_title="Portfolio value (EUR)")
                st.plotly_chart(fig2, use_container_width=True)


                hist_df = pd.DataFrame({"date": pv_series.index, "portfolio_value": pv_series.values})
                csv_hist = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button("Descarregar histórico do backtest (CSV)", data=csv_hist, file_name="backtest_history.csv", mime="text/csv")




